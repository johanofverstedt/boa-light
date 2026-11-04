"""Persistent, in-memory nnU-Net inference.

TotalSegmentator/BOA build a new ``nnUNetPredictor`` for every call, re-read the
checkpoint, start preprocessing/export process pools and pass the image through
gzipped temporary NIfTI files. Here every network is loaded once and kept for the
whole run, and images are handed over as numpy arrays.

To stay numerically identical to TotalSegmentator, the array given to nnU-Net is
exactly what nnU-Net's reader would have produced from TotalSegmentator's
temporary file: the voxel array in (z, y, x) order as float32, with the spacing
reversed, taken from the (float32) NIfTI header. This holds for both readers used
by these models (``SimpleITKIO`` for BCA, ``NibabelIOWithReorient`` for the
TotalSegmentator models), because the input is always in canonical (RAS) order.
"""

from __future__ import annotations

import logging
import threading
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np

from boa_light.models import NETWORKS
from boa_light.weights import model_folder

logger = logging.getLogger(__name__)


def resolve_device(device: str = "auto"):
    import torch

    if device in ("auto", None):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device in ("gpu",):
        device = "cuda"
    if device.startswith("gpu:"):
        device = "cuda:" + device.split(":", 1)[1]
    return torch.device(device)


class ModelPool:
    """Lazily loads nnU-Net networks and keeps them for reuse."""

    def __init__(
        self,
        device: str = "auto",
        weights_root: Path | None = None,
        keep_loaded: bool = True,
        tile_batch: int = 1,
        share_preprocessing: bool = True,
    ):
        import torch

        self.device = resolve_device(device)
        self.weights_root = weights_root
        self.keep_loaded = keep_loaded
        self.tile_batch = max(1, int(tile_batch))
        self.share_preprocessing = share_preprocessing
        self._predictors: dict[tuple, object] = {}
        self._lock = threading.Lock()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        if share_preprocessing:
            _install_crop_cache()

    # ------------------------------------------------------------------ #
    def _load(self, task_id: int, folds, step_size: float):
        import torch
        from batchgenerators.utilities.file_and_folder_operations import load_json
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

        net = NETWORKS[task_id]
        folder = model_folder(net, self.weights_root)
        if folds is None:
            folds = sorted(
                int(p.parent.name.split("_")[-1])
                for p in folder.glob("fold_*/checkpoint_final.pth")
                if p.parent.name != "fold_all"
            )
        dataset_json = load_json(str(folder / "dataset.json"))
        plans_manager = PlansManager(load_json(str(folder / "plans.json")))
        parameters = []
        configuration_name = net.configuration
        mirroring_axes = None
        for i, f in enumerate(folds):
            ckpt = torch.load(
                folder / f"fold_{f}" / "checkpoint_final.pth", map_location="cpu", weights_only=False
            )
            if i == 0:
                configuration_name = ckpt["init_args"]["configuration"]
                mirroring_axes = ckpt.get("inference_allowed_mirroring_axes")
            parameters.append(ckpt["network_weights"])
        with warnings.catch_warnings():
            # The TotalSegmentator v2.0 weights use the old plans format; nnU-Net converts it to
            # the identical architecture (a mismatch would fail the strict load_state_dict below).
            warnings.filterwarnings("ignore", message="Detected old nnU-Net plans format")
            configuration_manager = plans_manager.get_configuration(configuration_name)
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        label_manager = plans_manager.get_label_manager(dataset_json)
        # The trainers used by these models only change training (epochs / mirroring
        # augmentation), so the default nnU-Net architecture builder applies. This
        # avoids having to register TotalSegmentator's/BOA's custom trainer classes.
        network = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            label_manager.num_segmentation_heads,
            allow_init=True,
            deep_supervision=False,
        )
        network.load_state_dict(parameters[0])
        predictor = nnUNetPredictor(
            tile_step_size=step_size,
            use_gaussian=True,
            use_mirroring=False,  # TotalSegmentator/BOA never use test-time mirroring
            perform_everything_on_device=self.device.type != "cpu",
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.manual_initialization(
            network.to(self.device),
            plans_manager,
            configuration_manager,
            parameters,
            dataset_json,
            net.trainer,
            mirroring_axes,
        )
        if len(parameters) == 1:
            # Weights are already in the network; skip re-loading them on every call.
            predictor.list_of_parameters = [network.state_dict()]
        predictor.boa_light_folds = list(folds)
        if self.tile_batch > 1:
            _enable_tile_batching(predictor, self.tile_batch)
        logger.info("Loaded network %s (folds %s) on %s", task_id, list(folds), self.device)
        return predictor

    def get(self, task_id: int, folds, step_size: float):
        key = (task_id, tuple(folds) if folds is not None else None, step_size)
        with self._lock:
            if key not in self._predictors:
                self._predictors[key] = self._load(task_id, folds, step_size)
            return self._predictors[key]

    def release(self, task_id: int) -> None:
        if self.keep_loaded:
            return
        with self._lock:
            for key in [k for k in self._predictors if k[0] == task_id]:
                del self._predictors[key]
        if self.device.type == "cuda":
            import torch

            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    def predict(self, task_id: int, img: nib.Nifti1Image, folds=(0,), step_size: float = 0.5,
                return_folds: bool = False, shared_input: bool = False):
        """Segment a (canonical, resampled) NIfTI image; returns a uint8 array (x, y, z).

        With ``return_folds`` returns ``(ensemble, {fold id: segmentation})`` (None instead
        of the dict for single-fold models). The ensemble is computed exactly as nnU-Net does.

        ``shared_input=True`` marks ``img`` as being segmented by several networks in a row
        (the five "total" models): nnU-Net's crop-to-nonzero step is then computed once
        and reused (identical result).
        """
        predictor = self.get(task_id, folds, step_size)
        _CROP_CACHE["key"] = (
            (img, tuple(predictor.plans_manager.transpose_forward))
            if shared_input and self.share_preprocessing else None
        )  # fmt: skip
        try:
            return self._predict(task_id, predictor, img, return_folds)
        finally:
            _CROP_CACHE["key"] = None
            if not shared_input:
                _CROP_CACHE["entry"] = None

    def _predict(self, task_id, predictor, img, return_folds):
        data = np.asanyarray(img.dataobj).astype(np.float32, copy=False)
        array = np.ascontiguousarray(data.transpose(2, 1, 0))[None]
        spacing = [float(s) for s in img.header.get_zooms()[:3]][::-1]
        if return_folds and len(predictor.list_of_parameters) > 1:
            seg, fold_segs = self._predict_with_folds(predictor, array, spacing)
            self.release(task_id)

            def back(x):
                return np.asarray(x).transpose(2, 1, 0).astype(np.uint8)

            return back(seg), {f: back(x) for f, x in zip(predictor.boa_light_folds, fold_segs)}
        seg = None
        if self.fast_export and len(predictor.list_of_parameters) == 1 and self.device.type != "cpu":
            try:
                seg = self._predict_argmax_on_device(predictor, array, spacing)
            except Exception:  # never fail because of the optimisation
                logger.warning("On-device export failed for task %s; using nnU-Net's export", task_id, exc_info=True)
                self.fast_export = False
        if seg is None:
            seg = predictor.predict_single_npy_array(array, {"spacing": spacing}, None, None, False)
        self.release(task_id)
        out = np.asarray(seg).transpose(2, 1, 0).astype(np.uint8)
        return (out, None) if return_folds else out

    fast_export = True

    @staticmethod
    def _predict_with_folds(predictor, array: np.ndarray, spacing):
        """Ensemble prediction that also returns the segmentation of every fold.

        Mirrors ``nnUNetPredictor.predict_logits_from_preprocessed_data`` (fold logits are
        summed on the CPU in the same order and divided by the number of folds), so the
        ensemble is identical to ``predict_single_npy_array``. Each fold's logits are
        exported with the same function nnU-Net uses for the ensemble.
        """
        import torch
        from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
        from nnunetv2.inference.export_prediction import (
            convert_predicted_logits_to_segmentation_with_correct_shape,
        )

        ppa = PreprocessAdapterFromNpy(
            [array], [None], [{"spacing": spacing}], [None],
            predictor.plans_manager, predictor.dataset_json, predictor.configuration_manager,
            num_threads_in_multithreaded=1, verbose=False,
        )  # fmt: skip
        dct = next(ppa)
        props = dct["data_properties"]

        def to_seg(logits):
            return convert_predicted_logits_to_segmentation_with_correct_shape(
                logits, predictor.plans_manager, predictor.configuration_manager,
                predictor.label_manager, props, return_probabilities=False,
            )  # fmt: skip

        network = getattr(predictor.network, "_orig_mod", predictor.network)
        prediction = None
        fold_segs = []
        with torch.inference_mode():
            for params in predictor.list_of_parameters:
                network.load_state_dict(params)
                logits = predictor.predict_sliding_window_return_logits(dct["data"]).to("cpu")
                fold_segs.append(to_seg(logits.clone()))
                if prediction is None:
                    prediction = logits
                else:
                    prediction += logits
            if len(predictor.list_of_parameters) > 1:
                prediction /= len(predictor.list_of_parameters)
            seg = to_seg(prediction)
        return seg, fold_segs

    @staticmethod
    def _predict_argmax_on_device(predictor, array: np.ndarray, spacing) -> np.ndarray | None:
        """Same result as ``predict_single_npy_array`` but the argmax runs on the GPU.

        nnU-Net moves the (half precision) logits of the whole volume to the CPU and
        takes a numpy argmax over the class axis, which for "total" at 1.5 mm is a
        multi-GB array. When nnU-Net does not resample (the input is already at the
        network spacing, true for all TotalSegmentator-resampled tasks) the argmax
        can be taken on the device; ``torch.argmax`` and ``numpy.argmax`` both return
        the first maximum, so the result is identical. Returns None (before running
        the network) if the shortcut does not apply.
        """
        import torch
        from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy

        if predictor.label_manager.has_regions:
            return None
        ppa = PreprocessAdapterFromNpy(
            [array], [None], [{"spacing": spacing}], [None],
            predictor.plans_manager, predictor.dataset_json, predictor.configuration_manager,
            num_threads_in_multithreaded=1, verbose=False,
        )  # fmt: skip
        dct = next(ppa)
        props = dct["data_properties"]
        if tuple(props["shape_after_cropping_and_before_resampling"]) != tuple(dct["data"].shape[1:]):
            return None  # nnU-Net resampled the input -> logits must be resampled back (stock path)
        with torch.inference_mode():
            logits = predictor.predict_sliding_window_return_logits(dct["data"])
            seg_cropped = torch.argmax(logits, 0).to(torch.uint8).cpu().numpy()
        del logits
        seg = np.zeros(props["shape_before_cropping"], dtype=np.uint8)
        bbox = props["bbox_used_for_cropping"]
        seg[tuple(slice(int(lo), int(hi)) for lo, hi in bbox)] = seg_cropped
        return seg.transpose(predictor.plans_manager.transpose_backward)


# ---------------------------------------------------------------------------- #
# Shared preprocessing: cache nnU-Net's crop_to_nonzero for repeated inputs
# ---------------------------------------------------------------------------- #

_CROP_CACHE: dict = {"key": None, "entry": None}


def _install_crop_cache() -> None:
    """Wrap ``crop_to_nonzero`` in nnU-Net's default preprocessor with a one-entry cache.

    The five "total" networks receive the identical image; nnU-Net would compute the
    nonzero mask (incl. a full-volume ``binary_fill_holes``) and the crop five times.
    The cache is only active while ``ModelPool.predict(..., shared_input=True)`` runs,
    is keyed on the image object, the transpose and the array shape, and returns copies,
    so the per-network normalisation (done in place by nnU-Net) cannot leak between
    networks. Results are identical to running without the cache.
    """
    import copy

    try:
        import nnunetv2.preprocessing.preprocessors.default_preprocessor as dp
    except ImportError:  # pragma: no cover
        return
    if getattr(dp.crop_to_nonzero, "_boa_light_cached", False):
        return
    original = dp.crop_to_nonzero

    def crop_to_nonzero(data, seg=None, nonzero_label=-1):
        key = _CROP_CACHE["key"]
        if key is None or seg is not None:
            return original(data, seg, nonzero_label)
        full_key = (key, data.shape, str(data.dtype), nonzero_label)
        entry = _CROP_CACHE["entry"]
        if entry is not None and entry[0] == full_key:
            d, sg, bbox = entry[1]
            return d.copy(), sg.copy(), copy.deepcopy(bbox)
        d, sg, bbox = original(data, seg, nonzero_label)
        _CROP_CACHE["entry"] = (full_key, (d.copy(), sg.copy(), copy.deepcopy(bbox)))
        return d, sg, bbox

    crop_to_nonzero._boa_light_cached = True
    dp.crop_to_nonzero = crop_to_nonzero


# ---------------------------------------------------------------------------- #
# Opt-in: several sliding-window tiles per forward pass
# ---------------------------------------------------------------------------- #


def _enable_tile_batching(predictor, batch_size: int) -> None:
    """Replace nnU-Net's one-tile-at-a-time sliding window by a batched one.

    Same tiles, Gaussian weighting, half-precision accumulation (in the same tile
    order) and normalisation as nnU-Net's ``_internal_predict_sliding_window_return_logits``;
    only the network sees ``batch_size`` tiles at once. The math is the same, but GPU
    kernels for different batch sizes may differ in the last bits, so results can
    differ very slightly from BOA. Opt-in for that reason.
    """
    import types

    import torch
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian
    from nnunetv2.utilities.helpers import empty_cache

    @torch.inference_mode()
    def batched(self, data, slicers, do_on_device: bool = True):
        results_device = self.device if do_on_device else torch.device("cpu")
        empty_cache(self.device)
        data = data.to(results_device)
        predicted_logits = torch.zeros(
            (self.label_manager.num_segmentation_heads, *data.shape[1:]), dtype=torch.half, device=results_device
        )
        n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)
        if self.use_gaussian:
            gaussian = compute_gaussian(
                tuple(self.configuration_manager.patch_size), sigma_scale=1.0 / 8, value_scaling_factor=10,
                device=results_device,
            )  # fmt: skip
        else:
            gaussian = 1
        for i in range(0, len(slicers), batch_size):
            group = slicers[i : i + batch_size]
            x = torch.cat([data[sl][None] for sl in group]).to(self.device, memory_format=torch.contiguous_format)
            out = self._internal_maybe_mirror_and_predict(x)
            for j, sl in enumerate(group):
                prediction = out[j].to(results_device)
                if self.use_gaussian:
                    prediction *= gaussian
                predicted_logits[sl] += prediction
                n_predictions[sl[1:]] += gaussian
            del out, x
        torch.div(predicted_logits, n_predictions, out=predicted_logits)
        if torch.any(torch.isinf(predicted_logits)):
            raise RuntimeError("Encountered inf in predicted array (tile batching). Run without --tile-batch.")
        return predicted_logits

    predictor._internal_predict_sliding_window_return_logits = types.MethodType(batched, predictor)
