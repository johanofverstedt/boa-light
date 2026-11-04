"""Segmentation engine: the in-memory equivalent of TotalSegmentator's
``nnUNet_predict_image`` with caching of all work that is shared between tasks.

For each task the TotalSegmentator pipeline is

    load -> [crop to ROI of rough seg] -> canonical (RAS) -> resample -> nnU-Net
         -> resample back -> undo canonical -> [undo crop] -> uint8 label image

BOA runs this from scratch for every model. Here the steps before nnU-Net are split
into ``prepare`` (pure CPU, cached by crop/spacing so e.g. ``lung_vessels`` and
``pleural_pericard_effusion`` share one crop) and ``finalize`` (pure CPU), so the
pipeline can overlap them with GPU inference of other tasks.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable

import nibabel as nib
import numpy as np

from boa_light.class_maps import class_map, class_map_5_parts, map_taskid_to_partname_ct
from boa_light.imaging import (
    add_label_map_to_nifti,
    as_closest_canonical,
    change_spacing,
    crop_to_mask,
    reorient_to_axcodes,
    undo_canonical,
    undo_crop,
)
from boa_light.models import Task

logger = logging.getLogger(__name__)

NR_VOXELS_SPLIT_THR = 512 * 512 * 900  # TotalSegmentator's threshold for the triple split
SPLIT_MARGIN = 20
CROP_ADDON_MM = (20, 20, 20)  # TotalSegmentator uses 20 mm for all cropped subtasks

# predict_fn(task_id, nifti_image, folds, step_size) -> uint8 array (x, y, z)
PredictFn = Callable[[int, nib.Nifti1Image, object, float], np.ndarray]


@dataclass
class Prepared:
    task: Task
    img_in: nib.Nifti1Image | None  # canonical (and cropped) input, original spacing
    img_in_rsp: nib.Nifti1Image | None  # input given to nnU-Net
    bbox: list | None
    resample: list | None
    empty: bool = False


class ImageContext:
    """Holds the input CT and caches derived images."""

    def __init__(self, img: nib.Nifti1Image):
        if img.ndim == 4:
            logger.warning("Input image has 4 dimensions. Only using the first volume.")
            img = nib.Nifti1Image(np.asanyarray(img.dataobj)[..., 0], img.affine, img.header)
        if img.ndim != 3:
            raise ValueError(f"Only 3D CT images are supported, got {img.ndim}D.")
        if img.get_data_dtype().fields is not None:
            raise TypeError(f"Invalid dtype {img.get_data_dtype()}.")
        # Read the voxels once (scaled, in the stored dtype when possible).
        data = np.asanyarray(img.dataobj)
        self.orig = nib.Nifti1Image(data, img.affine, img.header)
        self.orig.header.set_data_dtype(data.dtype)
        self._lock = threading.Lock()
        self._key_locks: dict = {}
        self._cache: dict = {}

    # Simple per-key memoisation that is safe to call from several threads
    def memo(self, key, fn):
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            lock = self._key_locks.setdefault(key, threading.Lock())
        with lock:
            with self._lock:
                if key in self._cache:
                    return self._cache[key]
            value = fn()
            with self._lock:
                self._cache[key] = value
            return value

    def forget(self, prefix) -> None:
        with self._lock:
            for k in [k for k in self._cache if k[0] == prefix]:
                del self._cache[k]

    @property
    def base(self) -> nib.Nifti1Image:
        """float64 copy of the input (TotalSegmentator: ``Nifti1Image(get_fdata(), affine)``)."""
        return self.memo(("base",), lambda: nib.Nifti1Image(self.orig.get_fdata(), self.orig.affine))

    @property
    def ct_data(self) -> np.ndarray:
        """CT intensities in the original orientation (float64, (x, y, z))."""
        return self.base.get_fdata()


class SegmentationEngine:
    def __init__(self, ctx: ImageContext, predict_fn: PredictFn):
        self.ctx = ctx
        self.predict_fn = predict_fn

    # ------------------------------------------------------------------ #
    # Preparation (CPU)
    # ------------------------------------------------------------------ #
    def _crop_canonical(self, crop_key, crop_mask: np.ndarray | None):
        def compute():
            img_in = self.ctx.base
            bbox = None
            if crop_mask is not None:
                img_in, bbox = crop_to_mask(img_in, crop_mask, addon_mm=CROP_ADDON_MM, dtype=np.int32)
                logger.info("  cropping from %s to %s", crop_mask.shape, img_in.shape)
            return as_closest_canonical(img_in), bbox

        return self.ctx.memo(("crop", crop_key), compute)

    def prepare(self, task: Task, crop_mask: np.ndarray | None = None, crop_key=None) -> Prepared:
        if crop_mask is not None and not np.any(crop_mask):
            logger.info("Crop for %s is empty. Returning empty segmentation.", task.name)
            return Prepared(task, None, None, None, None, empty=True)
        img_in, bbox = self._crop_canonical(crop_key, crop_mask)

        resample = list(task.resample) if task.resample is not None else None
        if task.resample_only_thickness:
            img_in = reorient_to_axcodes(img_in, "RAS")  # TotalSegmentator default axcodes
            resample = [float(z) for z in img_in.header.get_zooms()[:2]] + [resample[0]]

        if resample is not None:
            key = ("rsp", crop_key, tuple(resample))
            img_in_rsp = self.ctx.memo(
                key, lambda: change_spacing(img_in, resample, order=3, dtype=np.int32)
            )
        else:
            img_in_rsp = img_in
        return Prepared(task, img_in, img_in_rsp, bbox, resample)

    # ------------------------------------------------------------------ #
    # Inference (GPU)
    # ------------------------------------------------------------------ #
    def _predict_parts(self, task: Task, img: nib.Nifti1Image, folds, return_folds: bool = False):
        """Returns the segmentation, or (segmentation, per-fold segmentations | None)."""
        if not task.multimodel:
            if return_folds:
                return self.predict_fn(task.task_ids[0], img, folds, task.step_size, return_folds=True)
            return self.predict_fn(task.task_ids[0], img, folds, task.step_size)
        class_map_inv = {v: k for k, v in class_map[task.label_map_key].items()}
        combined = np.zeros(img.shape, dtype=np.uint8)
        for tid in task.task_ids:
            # the same input goes through all part models -> nnU-Net preprocessing can be shared
            seg = self.predict_fn(tid, img, folds, task.step_size, shared_input=True)
            for jdx, class_name in class_map_5_parts[map_taskid_to_partname_ct[tid]].items():
                combined[seg == jdx] = class_map_inv[class_name]
        return (combined, None) if return_folds else combined

    @staticmethod
    def will_split(prep: Prepared, force_split: bool = False) -> bool:
        """TotalSegmentator's rule for splitting large images into three z-parts."""
        if prep.empty:
            return False
        ss = prep.img_in_rsp.shape
        return bool((np.prod(ss) > NR_VOXELS_SPLIT_THR and ss[2] > 200 and prep.task.multimodel) or force_split)

    def infer(self, prep: Prepared, folds=None, force_split: bool = False, return_folds: bool = False):
        """Segment a prepared input.

        Returns the segmentation (or None for an empty crop). With ``return_folds`` returns
        ``(segmentation, {fold id: segmentation})``; the dict is None for single-fold models.
        """
        if prep.empty:
            return (None, None) if return_folds else None
        task = prep.task
        folds = task.folds if folds is None else folds
        img = prep.img_in_rsp
        ss = img.shape
        if not self.will_split(prep, force_split):
            return self._predict_parts(task, img, folds, return_folds)

        logger.info("Splitting into subparts...")
        data = np.asanyarray(img.dataobj)
        third = ss[2] // 3
        m = SPLIT_MARGIN
        pieces = [
            data[:, :, : third + m],
            data[:, :, third + 1 - m : third * 2 + m],
            data[:, :, third * 2 + 1 - m :],
        ]
        results = [self._predict_parts(task, nib.Nifti1Image(p, img.affine), folds, return_folds) for p in pieces]

        def stitch(segs):
            combined = np.zeros(ss, dtype=np.uint8)
            combined[:, :, :third] = segs[0][:, :, :-m]
            combined[:, :, third : third * 2] = segs[1][:, :, m - 1 : -m]
            combined[:, :, third * 2 :] = segs[2][:, :, m - 1 :]
            return combined

        if not return_folds:
            return stitch(results)
        seg = stitch([r[0] for r in results])
        if results[0][1] is None:
            return seg, None
        return seg, {f: stitch([r[1][f] for r in results]) for f in results[0][1]}

    # ------------------------------------------------------------------ #
    # Finalisation (CPU)
    # ------------------------------------------------------------------ #
    def finalize(self, prep: Prepared, seg: np.ndarray | None) -> nib.Nifti1Image:
        task = prep.task
        orig = self.ctx.orig
        label_map = class_map[task.label_map_key]
        if prep.empty or seg is None:
            out = nib.Nifti1Image(np.zeros(orig.shape[:3], dtype=np.uint8), orig.affine)
            return add_label_map_to_nifti(out, label_map)

        img_pred = nib.Nifti1Image(seg, prep.img_in_rsp.affine)
        if prep.resample is not None:
            img_pred = change_spacing(
                img_pred,
                prep.resample,
                prep.img_in.shape,
                order=0,
                dtype=np.uint8,
                force_affine=prep.img_in.affine,
            )
        img_pred = undo_canonical(img_pred, orig)
        if prep.bbox is not None:
            img_pred = undo_crop(img_pred, orig, prep.bbox)

        data = np.asanyarray(img_pred.dataobj).astype(np.uint8)
        header = orig.header.copy()
        header.set_data_dtype(np.uint8)
        out = nib.Nifti1Image(data, img_pred.affine, header)
        return add_label_map_to_nifti(out, label_map)

    # Convenience: all three steps
    def run(self, task: Task, crop_mask=None, crop_key=None, folds=None, force_split=False):
        prep = self.prepare(task, crop_mask, crop_key)
        return self.finalize(prep, self.infer(prep, folds=folds, force_split=force_split))


def roi_mask(label_img: nib.Nifti1Image, rois, label_map_key: str = "total") -> np.ndarray:
    """Binary mask of the given ROI names in a "total" label image."""
    inv = {v: k for k, v in class_map[label_map_key].items()}
    ids = [inv[r] for r in rois]
    return np.isin(np.asanyarray(label_img.dataobj), ids).astype(np.uint8)
