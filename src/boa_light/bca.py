"""Body Composition Analysis (BCA): post-processing, tissue derivation and measurements.

Port of BOA's ``body_composition_analysis`` package (Apache-2.0, (c) IKIM / SHIP-AI,
UK Essen) without the PDF report. SimpleITK images are replaced by numpy arrays in
the same (z, y, x) axis order SimpleITK uses, so all slice-wise computations are
identical to BOA.
"""

from __future__ import annotations

import enum
import logging
from typing import Any

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage
import skimage.measure
from skimage.morphology import remove_small_objects

from boa_light.bca_definitions import (
    TISSUE_DERIVATION_RULES,
    BodyParts,
    BodyRegion,
    HURange,
    Tissue,
)
from boa_light.class_maps import class_map
from boa_light.imaging import add_label_map_to_nifti, lps_zyx

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Post-processing of the network outputs (arrays in SimpleITK (z, y, x) order)
# --------------------------------------------------------------------------- #


def _remove_small_objects(mask: np.ndarray, threshold: int) -> None:
    """In-place removal of connected components with fewer than ``threshold`` voxels."""
    try:  # scikit-image >= 0.26
        remove_small_objects(mask, max_size=threshold - 1, connectivity=3, out=mask)
    except TypeError:  # older scikit-image
        remove_small_objects(mask, min_size=threshold, connectivity=3, out=mask)


def remove_small_labeled_objects(mask: np.ndarray, threshold: int = 3000) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=mask.dtype)
    labels = np.unique(mask)
    for label in labels[labels > 0]:
        label_mask = mask == label
        filled = np.zeros(label_mask.shape, dtype=np.uint8)
        for i in range(label_mask.shape[0]):
            contours, _ = cv2.findContours(
                label_mask[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(filled[i], contours, -1, color=1, thickness=cv2.FILLED)
        filled = filled.astype(bool)
        _remove_small_objects(filled, threshold)
        np.invert(filled, out=filled)
        _remove_small_objects(filled, threshold)
        np.invert(filled, out=filled)
        out[filled] = label
    return out


def _filter_largest_unique_segment(segmentation: np.ndarray, mask: np.ndarray) -> None:
    labels = skimage.measure.label(mask)
    props = skimage.measure.regionprops(labels)
    if len(props) > 1:
        props = sorted(props, key=lambda x: int(x.area), reverse=True)
        for prop in props[1:]:
            segmentation[np.equal(labels, prop.label)] = 255


def postprocess_regions(seg_data: np.ndarray) -> np.ndarray:
    seg_data = seg_data.copy()
    _filter_largest_unique_segment(seg_data, np.greater(seg_data, 0))
    _filter_largest_unique_segment(
        seg_data,
        np.isin(
            seg_data,
            [BodyRegion.THORACIC_CAVITY.value, BodyRegion.MEDIASTINUM.value, BodyRegion.PERICARDIUM.value],
        ),
    )
    for region in (BodyRegion.PERICARDIUM, BodyRegion.ABDOMINAL_CAVITY):
        _filter_largest_unique_segment(seg_data, np.equal(seg_data, region.value))
    return seg_data


def postprocess_nifti(img: nib.Nifti1Image, task_name: str) -> nib.Nifti1Image:
    """Apply BOA's body_parts / body_regions post-processing to a label image."""
    zyx = np.asanyarray(img.dataobj).astype(np.uint8).transpose(2, 1, 0)
    if task_name == "body_parts":
        zyx = remove_small_labeled_objects(zyx)
    elif task_name == "body_regions":
        zyx = postprocess_regions(zyx)
    header = img.header.copy()
    header.extensions.clear()
    out = nib.Nifti1Image(np.ascontiguousarray(zyx.transpose(2, 1, 0)), img.affine, header)
    return add_label_map_to_nifti(out, class_map[task_name])


def subclassify_tissues(
    ct: nib.Nifti1Image, body_regions: nib.Nifti1Image, median_filtering: bool = False
) -> nib.Nifti1Image:
    image_data = np.asanyarray(ct.dataobj)
    seg_data = np.asanyarray(body_regions.dataobj)
    if median_filtering:
        orientation = nib.aff2axcodes(body_regions.affine)
        if "I" in orientation:
            pos = orientation.index("I")
        elif "S" in orientation:
            pos = orientation.index("S")
        else:
            raise ValueError(f"The orientation {orientation} contains neither I nor S.")
        kernel = [3, 3, 3]
        kernel[pos] = 1
        image_data = scipy.ndimage.median_filter(image_data, size=kernel)
    masks = {
        r: np.logical_and(np.greater_equal(image_data, r.value[0]), np.less_equal(image_data, r.value[1]))
        for r in HURange
    }
    tissue = np.zeros(seg_data.shape, dtype=np.uint8)
    for t, (hu, region) in TISSUE_DERIVATION_RULES.items():
        tissue[masks[hu] & np.equal(seg_data, region.value)] = t.value
    header = body_regions.header.copy()
    header.set_data_dtype(np.uint8)
    header.extensions.clear()
    out = nib.Nifti1Image(tissue, body_regions.affine, header)
    return add_label_map_to_nifti(out, class_map["tissues"])


# --------------------------------------------------------------------------- #
# Measurements (port of report/builder.py without plotting)
# --------------------------------------------------------------------------- #


class AggregatableBodyPart(enum.IntFlag):
    NONE = 0
    ABDOMEN = 1
    THORAX = 2
    NECK = 4

    @staticmethod
    def from_body_regions(
        region_data: np.ndarray,
        slice_thickness: float,
        min_abdomen_length: float = 200,
        min_neck_length: float = 100,
        min_thorax_length: float = 200,
    ) -> AggregatableBodyPart:
        result = AggregatableBodyPart.NONE
        abdomen_mask = np.equal(region_data, BodyRegion.ABDOMINAL_CAVITY)
        abdomen_slices = np.where(abdomen_mask.any(axis=(1, 2)))[0]
        n_abd = abdomen_slices.max() - abdomen_slices.min() + 1 if abdomen_slices.size else 0
        if n_abd * slice_thickness >= min_abdomen_length:
            result |= AggregatableBodyPart.ABDOMEN

        mediastinum_slices = np.where(np.equal(region_data, BodyRegion.MEDIASTINUM).any(axis=(1, 2)))[0]
        n_neck = region_data.shape[0] - mediastinum_slices.max() if mediastinum_slices.size else 0
        if n_neck * slice_thickness >= min_neck_length:
            result |= AggregatableBodyPart.NECK

        thorax_mask = np.isin(
            region_data, [BodyRegion.THORACIC_CAVITY, BodyRegion.MEDIASTINUM, BodyRegion.PERICARDIUM]
        )
        thorax_slices = np.where(thorax_mask.any(axis=(1, 2)))[0]
        has_intersection = np.logical_and(abdomen_mask.any(axis=(1, 2)), thorax_mask.any(axis=(1, 2))).any()
        n_tho = thorax_slices.max() - thorax_slices.min() + 1 if thorax_slices.size else 0
        if has_intersection and n_tho * slice_thickness >= min_thorax_length:
            result |= AggregatableBodyPart.THORAX
        return result


def _tissue_name(t: Tissue) -> str:
    return t.name.capitalize() if t in {Tissue.BONE, Tissue.MUSCLE} else t.name


def create_vertebrae_info(total_zyx: np.ndarray, part: AggregatableBodyPart) -> dict[str, tuple[int, int]]:
    vertebrae_map = {
        v.removeprefix("vertebrae_"): k for k, v in class_map["total"].items() if v.startswith("vertebrae_")
    }
    present = {label: np.where((total_zyx == label).any(axis=(1, 2)))[0] for label in vertebrae_map.values()}
    info = {}
    for vid, label in vertebrae_map.items():
        idx = present[label]
        if len(idx) == 0:
            continue
        if (
            ("C" in vid and AggregatableBodyPart.NECK not in part)
            or ("T" in vid and AggregatableBodyPart.THORAX not in part)
            or ("L" in vid and AggregatableBodyPart.ABDOMEN not in part)
        ):
            continue
        info[vid] = (int(idx.min()), int(idx.max() + 1))
    return info


def _slicewise(tissue_data: np.ndarray, ml_per_voxel: float, mask: np.ndarray | None = None) -> pd.DataFrame:
    data = {}
    for t in Tissue:
        m = tissue_data == t.value
        if mask is not None:
            m = np.logical_and(mask, m)
        data[_tissue_name(t)] = m.sum(axis=(1, 2)) * ml_per_voxel
    df = pd.DataFrame(data)
    df["TAT"] = df.SAT + df.VAT + df.IMAT + df.PAT + df.EAT
    df["slice_idx"] = range(len(df))
    return df[["slice_idx", "Bone", "Muscle", "TAT", "IMAT", "SAT", "VAT", "PAT", "EAT"]]


def _describe(slicewise: pd.DataFrame, image_data: np.ndarray, tissue_data: np.ndarray) -> pd.DataFrame:
    slicewise = slicewise.drop("slice_idx", axis=1)
    m = slicewise.describe()
    m = m.drop("count")
    m.index = ["Mean", "StdDev", "Minimum", "25%", "Median", "75%", "Maximum"]
    m.loc["Total"] = slicewise.sum()
    m = m.astype(object)
    for t in Tissue:
        masked = image_data[np.equal(tissue_data, t)]
        m.loc["MeanHU", _tissue_name(t)] = np.mean(masked) if masked.size else None
    masked = image_data[np.isin(tissue_data, [Tissue.IMAT, Tissue.SAT, Tissue.VAT, Tissue.PAT, Tissue.EAT])]
    m.loc["MeanHU", "TAT"] = np.mean(masked) if masked.size else None
    return m.replace({np.nan: None})


def _secondary_findings(region_data: np.ndarray, spacing, part: AggregatableBodyPart) -> list[str]:
    def pretty(v: float) -> str:
        return f"{v / 1000:.3f} L" if v >= 1000 else f"{v:.2f} mL"

    result = []
    mid_index = region_data.shape[1] // 2
    ml = np.prod(spacing) / 1000.0
    if AggregatableBodyPart.ABDOMEN in part:
        vol = np.equal(region_data, BodyRegion.ABDOMINAL_CAVITY.value).sum() * ml
        result.append(f"Total volume of the abdominal cavity is {pretty(vol)}")
    if AggregatableBodyPart.THORAX in part:
        tc = [BodyRegion.THORACIC_CAVITY.value, BodyRegion.MEDIASTINUM.value, BodyRegion.PERICARDIUM.value]
        result.append(f"Volume of thoracic cavity is {pretty(np.isin(region_data, tc).sum() * ml)}")
        med = [BodyRegion.MEDIASTINUM.value, BodyRegion.PERICARDIUM.value]
        result.append(f"Volume of mediastinum is {pretty(np.isin(region_data, med).sum() * ml)}")
        per = np.equal(region_data, BodyRegion.PERICARDIUM).sum() * ml
        result.append(f"Volume enclosed by the pericardial sack is {pretty(per)}")
        implant = np.equal(region_data, BodyRegion.BREAST_IMPLANT)
        if implant.any():
            props = skimage.measure.regionprops(skimage.measure.label(implant))
            props = sorted([p for p in props if p.area * ml > 10], key=lambda p: int(p.centroid[2]))
            ms = [("right" if p.centroid[2] < mid_index else "left", p.area * ml) for p in props]
            if len(ms) == 1:
                result.append(
                    f"Patient has a single breast implant on the {ms[0][0]} side with volume of "
                    f"{pretty(ms[0][1])}"
                )
            elif len(ms) == 2:
                result.append(
                    f"Patient has two breast implants with volume of {pretty(ms[0][1])} ({ms[0][0]}) "
                    f"and {pretty(ms[1][1])} ({ms[1][0]})"
                )
            elif ms:
                logger.error("More than two breast implant segments found")
    return result


_ROW_RENAME = {
    "Mean": "mean", "StdDev": "std", "Minimum": "min", "25%": "q1", "Median": "q2",
    "75%": "q3", "Maximum": "max", "Total": "sum", "MeanHU": "mean_hu",
}  # fmt: skip


def compute_bca_measurements(
    ct: nib.Nifti1Image,
    body_parts: nib.Nifti1Image | None,
    body_regions: nib.Nifti1Image,
    tissues: nib.Nifti1Image,
    total: nib.Nifti1Image | None,
    examined_body_region: str | None = None,
) -> tuple[dict[str, Any], dict[str, tuple[int, int]]]:
    """Return (bca-measurements.json content, vertebrae.json content).

    Without ``body_parts`` the "no extremities" measurements are omitted.
    """
    image_data, spacing = lps_zyx(ct)
    parts_data = lps_zyx(body_parts)[0] if body_parts is not None else None
    region_data, region_spacing = lps_zyx(body_regions)
    tissue_data, _ = lps_zyx(tissues)

    if examined_body_region:
        part = AggregatableBodyPart[examined_body_region.upper()]
    else:
        part = AggregatableBodyPart.from_body_regions(region_data, region_spacing[2])
        if part == AggregatableBodyPart.NONE:
            logger.warning("No supported body part detected")

    vertebrae = {}
    if total is not None:
        total_data, _ = lps_zyx(total)
        vertebrae = create_vertebrae_info(total_data, part)

    ml_per_voxel = np.prod(spacing) / 1000.0
    df = _slicewise(tissue_data, ml_per_voxel)
    torso = df_no_limbs = None
    if parts_data is not None:
        torso = parts_data == BodyParts.TORSO
        df_no_limbs = _slicewise(tissue_data, ml_per_voxel, torso)

    # Aggregation groups
    groups = [("Whole Scan", 0, image_data.shape[0])]

    def extent(mask):
        s = np.where(mask.any(axis=(1, 2)))[0]
        return s.min(), s.max() + 1

    if AggregatableBodyPart.ABDOMEN in part:
        groups.append(("Abdominal Cavity", *extent(np.equal(region_data, BodyRegion.ABDOMINAL_CAVITY))))
    if AggregatableBodyPart.THORAX in part:
        tc = [BodyRegion.THORACIC_CAVITY, BodyRegion.MEDIASTINUM, BodyRegion.PERICARDIUM]
        groups.append(("Thoracic Cavity", *extent(np.isin(region_data, tc))))
        groups.append(("Mediastinum", *extent(np.equal(region_data, BodyRegion.MEDIASTINUM))))
        groups.append(("Pericardium", *extent(np.equal(region_data, BodyRegion.PERICARDIUM))))
    if AggregatableBodyPart.ABDOMEN in part and AggregatableBodyPart.THORAX in part:
        groups.insert(1, ("Ventral Cavity", groups[1][1], groups[2][2]))
    for name, (lo, hi) in vertebrae.items():
        groups.append((name, lo, hi))

    aggregated = {}
    for name, lo, hi in groups:
        sel = (df.slice_idx >= lo) & (df.slice_idx < hi)
        img_r, tis_r = image_data[lo:hi], tissue_data[lo:hi]
        m = _describe(df[sel], img_r, tis_r)
        entry = {
            "num_slices": int(hi - lo),
            "min_slice_idx": int(lo),
            "max_slice_idx": int(hi),
            "measurements": m.rename(index=_ROW_RENAME, columns=str.lower).to_dict(),
        }
        if df_no_limbs is not None:
            sel_nl = (df_no_limbs.slice_idx >= lo) & (df_no_limbs.slice_idx < hi)
            m_nl = _describe(df_no_limbs[sel_nl], img_r, np.where(torso[lo:hi], tis_r, 0))
            entry["measurements_no_extremities"] = m_nl.rename(index=_ROW_RENAME, columns=str.lower).to_dict()
        aggregated[name.lower().replace(" ", "_").replace("-", "_")] = entry

    def records(frame):
        return frame.rename(columns=str.lower).drop("slice_idx", axis=1).astype(float).to_dict("records")

    result = {"slices": records(df)}
    if df_no_limbs is not None:
        result["slices_no_extremities"] = records(df_no_limbs)
    result |= {
        "aggregated": aggregated,
        "body_parts": {
            "abdomen": AggregatableBodyPart.ABDOMEN in part,
            "neck": AggregatableBodyPart.NECK in part,
            "thorax": AggregatableBodyPart.THORAX in part,
        },
        "other_findings": _secondary_findings(region_data, region_spacing, part),
    }
    return result, vertebrae


def body_region_flags(body_regions: nib.Nifti1Image) -> int:
    """BOA's ``bca_regions`` statistic: bit0 abdomen, bit1 thorax, bit2 brain."""
    data = np.asanyarray(body_regions.dataobj)
    flag = 0
    if (data == BodyRegion.ABDOMINAL_CAVITY).any():
        flag |= 1
    if (data == BodyRegion.THORACIC_CAVITY).any():
        flag |= 2
    if (data == BodyRegion.BRAIN).any():
        flag |= 4
    return flag
