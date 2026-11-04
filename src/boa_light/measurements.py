"""Per-structure CT measurements (``total-measurements.json``) and derived metrics.

Port of BOA ``compute/measurements.py``, ``compute/ts_metrics.py`` and
``compute/geometry.py`` (Apache-2.0, (c) IKIM / SHIP-AI, UK Essen), working on the
in-memory label images instead of re-reading NIfTI files.

Differences to BOA (intentional bug fixes):
  * label maps are taken from the model's own class map only (BOA also matched
    unrelated maps with the same prefix, e.g. ``total_v1``/``total_mr`` entries
    appeared as ``v1_*``/``mr_*`` regions inside ``total``);
  * every requested open model is measured (BOA looked for e.g.
    ``lung_vessels_airways.nii.gz`` while writing ``lung_vessels.nii.gz`` and so
    silently skipped lung vessels).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import spatial
from scipy import ndimage

from boa_light.class_maps import class_map
from boa_light.imaging import lps_spacing, lps_zyx

logger = logging.getLogger(__name__)

ADIPOSE_TISSUE = (-200, -40)
CNR_ADJUSTED_REGIONS = {"total": {"aorta", "autochthon_left", "autochthon_right"}}
LUNG_LOBES = [
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
]


def _erosion_kernel(kernel_value: int = 6) -> np.ndarray:
    kernel = np.ones([kernel_value] * 3, dtype=np.uint8)
    if kernel_value % 2 == 0:  # preserve old skimage behaviour for even kernels
        kernel = np.pad(kernel, [(0, 1)] * 3)
    return kernel


def erode_region(mask: np.ndarray, kernel_value: int = 6) -> np.ndarray:
    """== ``skimage.morphology.binary_erosion`` (0.26) with BOA's padded cube kernel.

    The erosion is computed inside the bounding box of the mask plus a margin of
    the kernel size (clipped at the image border, where ``border_value=True``
    behaves identically), which gives the same result much faster.
    """
    kernel = _erosion_kernel(kernel_value)
    out = np.zeros(mask.shape, dtype=bool)
    if not mask.any():
        return out
    margin = kernel.shape[0]
    box = []
    for axis in range(mask.ndim):
        other = tuple(a for a in range(mask.ndim) if a != axis)
        idx = np.where(mask.any(axis=other))[0]
        box.append(slice(max(0, idx[0] - margin), min(mask.shape[axis], idx[-1] + 1 + margin)))
    box = tuple(box)
    out[box] = ndimage.binary_erosion(mask[box], structure=kernel, border_value=True)
    return out


def get_region_minus_fat(ct: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    return np.logical_and(
        region_mask, np.logical_or(np.less(ct, ADIPOSE_TISSUE[0]), np.greater(ct, ADIPOSE_TISSUE[1]))
    )


def autochthon_reference(ct, right, left) -> tuple[float | None, float | None]:
    mask = erode_region(get_region_minus_fat(ct, np.logical_or(right, left)))
    if np.sum(mask) == 0:
        return None, None
    return float(np.mean(ct[mask])), float(np.std(ct[mask]))


def metrics_for_region(
    ct: np.ndarray,
    mask: np.ndarray,
    autochthon_mean,
    autochthon_std,
    spacing,
    cnr_adjustment: bool = False,
    region_name: str = "",
) -> dict[str, Any]:
    m: dict[str, Any] = {}
    if not mask.any():
        return {"present": False}
    if cnr_adjustment:
        if "autochthon" in region_name:
            mask = get_region_minus_fat(ct, mask)
        mask = erode_region(mask)
    if not mask.any():
        return {"present": False}
    hu = ct[mask]
    m["present"] = True
    m["volume_ml"] = float(np.sum(mask) * (np.prod(spacing) / 1000.0))
    m["mean_hu"] = float(np.mean(hu))
    m["std_hu"] = float(np.std(hu))
    m["min_hu"] = float(np.min(hu))
    m["median_hu"] = float(np.median(hu))
    m["max_hu"] = float(np.max(hu))
    m["25th_percentile_hu"] = float(np.percentile(hu, 25))
    m["75th_percentile_hu"] = float(np.percentile(hu, 75))
    if autochthon_mean is not None and autochthon_std is not None:
        if cnr_adjustment and region_name.partition("_")[0] == "autochthon":
            m["cnr"] = None
        else:
            m["cnr"] = float((np.mean(hu) - autochthon_mean) / autochthon_std)
    else:
        m["cnr"] = None
    return m


def _label_boxes(seg: np.ndarray) -> list:
    """Bounding box (tuple of slices) per label value, from a single pass over ``seg``."""
    return ndimage.find_objects(seg.astype(np.int32, copy=False) if seg.dtype.kind not in "iu" else seg)


def metrics_for_each_region(
    ct, seg, label_map: dict[str, int], a_mean, a_std, spacing, cnr_adjustment=False
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if cnr_adjustment:  # needs full masks for the erosion
        for region, label in label_map.items():
            out[region] = metrics_for_region(ct, seg == label, a_mean, a_std, spacing, True, region)
    else:
        # Same values as BOA's full-volume masks, but each label is only searched for
        # inside its bounding box (boolean indexing keeps the C order, so even the
        # floating point sums are identical).
        boxes = _label_boxes(seg)
        n_voxel_ml = np.prod(spacing) / 1000.0
        for region, label in label_map.items():
            box = boxes[label - 1] if 0 < label <= len(boxes) else None
            if box is None:
                out[region] = {"present": False}
                continue
            hu = ct[box][seg[box] == label]
            if hu.size == 0:
                out[region] = {"present": False}
                continue
            m = {
                "present": True,
                "volume_ml": float(hu.size * n_voxel_ml),
                "mean_hu": float(np.mean(hu)),
                "std_hu": float(np.std(hu)),
                "min_hu": float(np.min(hu)),
                "median_hu": float(np.median(hu)),
                "max_hu": float(np.max(hu)),
                "25th_percentile_hu": float(np.percentile(hu, 25)),
                "75th_percentile_hu": float(np.percentile(hu, 75)),
            }
            m["cnr"] = float((np.mean(hu) - a_mean) / a_std) if a_mean is not None and a_std is not None else None
            out[region] = m
    if "autochthon_left" in label_map and "autochthon_right" in label_map:
        mask = np.isin(seg, [label_map["autochthon_left"], label_map["autochthon_right"]])
        out["autochthon"] = metrics_for_region(ct, mask, a_mean, a_std, spacing, cnr_adjustment, "autochthon")
    return out


def ct_pfav(ct, seg, label_map, a_mean, a_std, spacing) -> tuple[dict[str, Any], np.ndarray]:
    """Pulmonary fat attenuation volume measurements; also returns the fat mask."""

    def lung_measure(ids):
        mask = np.isin(seg, ids)
        fat = mask & (ct >= ADIPOSE_TISSUE[0]) & (ct <= ADIPOSE_TISSUE[1])
        return fat, metrics_for_region(ct, fat, a_mean, a_std, spacing)

    m = {}
    for region in LUNG_LOBES:
        _, m["ct_pfav_" + region] = lung_measure([label_map[region]])
    for side in ["left", "right"]:
        _, m[f"ct_pfav_lobe_{side}"] = lung_measure([label_map[ll] for ll in LUNG_LOBES if ll.endswith(side)])
    fat_mask, m["ct_pfav_lungs"] = lung_measure([label_map[ll] for ll in LUNG_LOBES])
    return m, fat_mask


def compute_measurements(
    ct_img: nib.Nifti1Image,
    segmentations: dict[str, nib.Nifti1Image],
    cnr_adjustment: bool = False,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Compute ``total-measurements.json``. Returns (json, ct_pfav mask or None)."""
    measurements: dict[str, Any] = {"segmentations": {}, "info": {}}
    ct = np.asanyarray(ct_img.dataobj)
    spacing = np.array(ct_img.header.get_zooms()[:3], dtype=np.float64)  # == SimpleITK spacing
    a_mean = a_std = None
    pfav_mask = None
    names = sorted(segmentations, key=lambda m: m != "total")
    for name in names:
        seg = np.asanyarray(segmentations[name].dataobj)
        label_map = {v: k for k, v in class_map[name].items()}
        if name == "total":
            a_mean, a_std = autochthon_reference(
                ct, seg == label_map["autochthon_right"], seg == label_map["autochthon_left"]
            )
        measurements["segmentations"][name] = metrics_for_each_region(ct, seg, label_map, a_mean, a_std, spacing)
        if name == "total":
            pfav, pfav_mask = ct_pfav(ct, seg, label_map, a_mean, a_std, spacing)
            measurements["segmentations"][name].update(pfav)
        if cnr_adjustment and name in CNR_ADJUSTED_REGIONS:
            if a_mean is None or a_std is None:
                logger.warning("Skipping CNR-adjusted measurements: autochthon reference unavailable.")
            else:
                regions = CNR_ADJUSTED_REGIONS[name]
                adj = metrics_for_each_region(
                    ct, seg, {r: v for r, v in label_map.items() if r in regions}, a_mean, a_std, spacing, True
                )
                measurements.setdefault("cnr_adjusted", {}).update(adj)
    measurements["info"]["autochthon_mean"] = a_mean
    measurements["info"]["autochthon_std"] = a_std
    return measurements, pfav_mask


# --------------------------------------------------------------------------- #
# Major/minor body axis at L3 (geometry.py + ts_metrics.py)
# --------------------------------------------------------------------------- #


@dataclass
class Point:
    x: float
    y: float

    def to_list(self) -> list[float]:
        return [self.x, self.y]


def _find_minor_point(contours, mid: Point, rot: Point, length: int, size) -> Point | None:
    point = Point(int(mid.x + rot.x * length), int(mid.y + rot.y * length))
    contour_array = cv2.drawContours(np.zeros(size), contours, contourIdx=-1, color=1, thickness=2)
    p_array = cv2.line(np.zeros(size), [point.x, point.y], [int(mid.x), int(mid.y)], 1, 2)
    opts = np.logical_and(contour_array, p_array).nonzero()
    if len(opts[0]) == 0:
        return None
    return Point(opts[1][0], opts[0][0])


def find_axes(middle_slice: np.ndarray):
    points = np.flip(np.transpose(np.where(middle_slice)))
    hull_points = points[spatial.ConvexHull(points).vertices]
    hdist = spatial.distance.cdist(hull_points, hull_points, metric="euclidean")
    p1, p2 = np.unravel_index(hdist.argmax(), hdist.shape)
    a, b = Point(*hull_points[p1]), Point(*hull_points[p2])
    mid = Point((a.x + b.x) // 2, (a.y + b.y) // 2)
    length = sum(middle_slice.shape)
    norm = Point(a.x - b.x, a.y - b.y)
    fac = math.sqrt(norm.x * norm.x + norm.y * norm.y)
    norm.x /= fac
    norm.y /= fac
    contours, _ = cv2.findContours(middle_slice.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c1 = _find_minor_point(contours, mid, Point(-norm.y, norm.x), length, middle_slice.shape)
    c2 = _find_minor_point(contours, mid, Point(norm.y, -norm.x), length, middle_slice.shape)
    if c1 is None or c2 is None:
        return None
    return a, b, c1, c2


def major_minor_axis(l3_mask, body_mask, spacing_xy) -> tuple[float | None, float | None]:
    if not l3_mask.any() or not body_mask.any():
        return None, None
    slices = np.where(l3_mask.any(axis=(1, 2)))[0]
    middle = body_mask[int(np.median(slices)), :, :]
    if not middle.any():
        return None, None
    try:
        axes = find_axes(middle)
    except Exception:  # degenerate hull
        logger.warning("Could not compute body axes at L3", exc_info=True)
        return None, None
    if axes is None:
        return None, None
    a, b, c1, c2 = axes
    avg = np.mean(spacing_xy)
    return (
        spatial.distance.euclidean(a.to_list(), b.to_list()) * avg,
        spatial.distance.euclidean(c1.to_list(), c2.to_list()) * avg,
    )


def convert_name(name: str) -> str:
    return "".join(s.capitalize() for s in name.split("_"))


def _convert_key(key: str) -> str:
    k = convert_name(key)
    if "Hu" in k:
        return k.replace("Hu", "HU")
    return "CNR" if k == "Cnr" else k


def segmentator_metrics(
    measurements: dict[str, Any],
    ct_img: nib.Nifti1Image,
    total: nib.Nifti1Image | None,
    body_parts: nib.Nifti1Image | None,
    open_models: list[str],
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Port of ``compute_segmentator_metrics``: (info rows, regions df, cnr df)."""
    segs = measurements["segmentations"]

    def cnr(region):
        if "total" in segs and segs["total"].get(region, {}).get("present"):
            return segs["total"][region]["cnr"]
        return None

    major = minor = mean_axis = None
    if total is not None and body_parts is not None:
        ct_spacing = lps_spacing(ct_img)
        total_zyx, _ = lps_zyx(total)
        parts_zyx, _ = lps_zyx(body_parts)
        inv = {v: k for k, v in class_map["total"].items()}
        major, minor = major_minor_axis(total_zyx == inv["vertebrae_L3"], parts_zyx == 1, ct_spacing[:2])
    if major is not None and minor is not None:
        major /= 10
        minor /= 10
        mean_axis = (major + minor) / 2

    records = []
    for model, regions in segs.items():
        for region, values in regions.items():
            row = {"ModelName": convert_name(model), "BodyRegion": convert_name(region)}
            row.update({_convert_key(k): v for k, v in values.items()})
            records.append(row)
    for model in open_models:
        if model not in segs:
            records.append({"ModelName": convert_name(model), "Present": False})

    cnr_records = []
    for region in ("aorta", "autochthon", "autochthon_left", "autochthon_right"):
        if region in measurements.get("cnr_adjusted", {}):
            row = {"BodyRegion": convert_name(region)}
            row.update({_convert_key(k): v for k, v in measurements["cnr_adjusted"][region].items()})
            cnr_records.append(row)

    info = []
    for name, value in [
        ("Noise", measurements["info"].get("autochthon_std")),
        ("CNRAorta", cnr("aorta")),
        ("CNRVCI", cnr("inferior_vena_cava")),
        ("CNRPortalSplenicVein", cnr("portal_vein_and_splenic_vein")),
        ("MaxAxisL3_cm", major),
        ("MinAxisL3_cm", minor),
        ("MeanAxisL3_cm", mean_axis),
    ]:
        if value is not None:
            info.append({"name": name, "value": value})
    regions_df = pd.DataFrame(records)
    if len(regions_df):
        regions_df = regions_df.sort_values(by=["ModelName", "BodyRegion"])
    return info, regions_df, pd.DataFrame(cnr_records)
