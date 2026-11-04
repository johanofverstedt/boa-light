"""Image geometry helpers (resampling, cropping, reorientation).

These are faithful ports of the TotalSegmentator v2.12.0 helpers vendored by BOA
(``resampling.py``, ``cropping.py``, ``alignment.py``, ``nifti_ext_header.py``;
Apache-2.0, (c) Jakob Wasserthal). They operate on ``nibabel.Nifti1Image`` objects
and perform exactly the same numerical operations, so that BOA_light reproduces
BOA/TotalSegmentator results. The only changes are: no joblib/cucim code paths,
no file I/O, and integer output buffers where TotalSegmentator used float64.
"""

from __future__ import annotations

import logging

import nibabel as nib
import numpy as np
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from scipy import ndimage

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #


def resample_img(img: np.ndarray, zoom, order: int = 0) -> np.ndarray:
    """``ndimage.zoom`` with mode="nearest" (TotalSegmentator ``resample_img``)."""
    return ndimage.zoom(img, zoom, mode="nearest", order=order)


def change_spacing(
    img_in: nib.Nifti1Image,
    new_spacing=1.25,
    target_shape=None,
    order: int = 0,
    dtype=None,
    force_affine=None,
) -> nib.Nifti1Image:
    """Resample a NIfTI image to a new spacing or to an exact target shape.

    Port of ``totalsegmentator.resampling.change_spacing`` (CPU path).
    """
    data = img_in.get_fdata()
    old_shape = np.array(data.shape)
    img_spacing = np.array(img_in.header.get_zooms())[:3]

    if isinstance(new_spacing, float):
        new_spacing = [new_spacing] * 3
    new_spacing = np.array(new_spacing)

    if target_shape is not None:
        zoom = np.array(target_shape) / old_shape
        new_spacing = img_spacing / zoom
    else:
        zoom = img_spacing / new_spacing

    if np.array_equal(img_spacing, new_spacing):
        return img_in

    new_affine = np.copy(img_in.affine)
    new_affine[:3, 0] = new_affine[:3, 0] / zoom[0]
    new_affine[:3, 1] = new_affine[:3, 1] / zoom[1]
    new_affine[:3, 2] = new_affine[:3, 2] / zoom[2]

    new_data = resample_img(data, zoom=zoom, order=order)
    if dtype is not None:
        new_data = new_data.astype(dtype)
    if force_affine is not None:
        new_affine = force_affine
    return nib.Nifti1Image(new_data, new_affine)


# --------------------------------------------------------------------------- #
# Orientation
# --------------------------------------------------------------------------- #


def as_closest_canonical(img_in: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.as_closest_canonical(img_in)


def undo_canonical(img_can: nib.Nifti1Image, img_orig: nib.Nifti1Image) -> nib.Nifti1Image:
    """Invert ``nib.as_closest_canonical`` (TotalSegmentator ``undo_canonical``)."""
    img_ornt = io_orientation(img_orig.affine)
    ras_ornt = axcodes2ornt("RAS")
    from_canonical = ornt_transform(ras_ornt, img_ornt)
    return img_can.as_reoriented(from_canonical)


def reorient_to_axcodes(img: nib.Nifti1Image, axcodes: str = "RAS") -> nib.Nifti1Image:
    """Port of BCA ``load_nibabel_image_with_axcodes``."""
    input_axcodes = "".join(nib.aff2axcodes(img.affine))
    axcodes = "".join(axcodes)
    if input_axcodes != axcodes:
        transform = ornt_transform(axcodes2ornt(input_axcodes), axcodes2ornt(axcodes))
        return img.as_reoriented(transform)
    return img


def lps_spacing(img: nib.Nifti1Image) -> np.ndarray:
    """Voxel spacing (x, y, z) of ``img`` after reorientation to LPS (no data copy)."""
    rzs = img.affine[:3, :3]
    spacing = np.sqrt(np.sum(rzs * rzs, axis=0))
    transform = ornt_transform(io_orientation(img.affine), axcodes2ornt("LPS"))
    out = np.empty(3)
    for i in range(3):
        out[int(transform[i, 0])] = spacing[i]
    return out


def lps_zyx(img: nib.Nifti1Image) -> tuple[np.ndarray, np.ndarray]:
    """Return (array, spacing) as SimpleITK would see the LPS-reoriented image.

    Equivalent to BCA's ``process_image`` followed by ``sitk.GetArrayViewFromImage``:
    the image is reoriented to LPS and the axes are reversed to (z, y, x).
    The returned spacing is in SimpleITK (x, y, z) order.
    """
    img = reorient_to_axcodes(img, "LPS")
    data = np.asanyarray(img.dataobj)
    rzs = img.affine[:3, :3]
    spacing = np.sqrt(np.sum(rzs * rzs, axis=0))
    return data.transpose(2, 1, 0), spacing


# --------------------------------------------------------------------------- #
# Cropping
# --------------------------------------------------------------------------- #


def get_bbox_from_mask(mask: np.ndarray, outside_value=0, addon=0) -> list[list[int]]:
    if isinstance(addon, int):
        addon = [addon] * 3
    s = mask.shape
    fg = mask > outside_value
    if not fg.any():
        logger.warning("Could not crop because no foreground detected")
        return [[0, s[0]], [0, s[1]], [0, s[2]]]
    bbox = []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        idx = np.where(fg.any(axis=other))[0]
        lo = max(0, int(idx.min()) - int(addon[axis]))
        hi = min(s[axis], int(idx.max()) + 1 + int(addon[axis]))
        bbox.append([lo, hi])
    return bbox


def crop_to_bbox_nifti(image: nib.Nifti1Image, bbox, dtype=None) -> nib.Nifti1Image:
    data = np.asanyarray(image.dataobj)
    cropped = data[bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]]
    affine = np.copy(image.affine)
    affine[:3, 3] = np.dot(affine, np.array([bbox[0][0], bbox[1][0], bbox[2][0], 1]))[:3]
    data_type = image.dataobj.dtype if dtype is None else dtype
    return nib.Nifti1Image(cropped.astype(data_type), affine)


def crop_to_mask(img_in: nib.Nifti1Image, mask: np.ndarray, addon_mm=(0, 0, 0), dtype=None):
    """Crop ``img_in`` to the bounding box of ``mask`` (+ margin in mm)."""
    addon = (np.array(addon_mm) / img_in.header.get_zooms()).astype(int)
    bbox = get_bbox_from_mask(mask, outside_value=0, addon=addon)
    return crop_to_bbox_nifti(img_in, bbox, dtype), bbox


def undo_crop(img: nib.Nifti1Image, ref_img: nib.Nifti1Image, bbox, dtype=np.uint8) -> nib.Nifti1Image:
    out = np.zeros(ref_img.shape[:3], dtype=dtype)
    out[bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]] = np.asanyarray(
        img.dataobj
    )
    return nib.Nifti1Image(out, ref_img.affine)


# --------------------------------------------------------------------------- #
# NIfTI label map extension (readable by TotalSegmentator / MITK / Slicer)
# --------------------------------------------------------------------------- #

_COLORS = [
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],
    [255, 128, 0], [255, 0, 128], [128, 255, 128], [0, 128, 255], [128, 128, 128], [185, 170, 155],
]  # fmt: skip


def add_label_map_to_nifti(img_in: nib.Nifti1Image, label_map: dict[int, str]) -> nib.Nifti1Image:
    xmlpre = (
        '<?xml version="1.0" encoding="UTF-8"?> <CaretExtension>  '
        "<Date><![CDATA[2013-07-14T05:45:09]]></Date>   "
        '<VolumeInformation Index="0">   <LabelTable>'
    )
    body = ""
    for label_id, label_name in label_map.items():
        rgb = _COLORS[label_id % len(_COLORS)]
        body += (
            f'<Label Key="{label_id}" Red="{rgb[0] / 255}" Green="{rgb[1] / 255}" '
            f'Blue="{rgb[2] / 255}" Alpha="1"><![CDATA[{label_name}]]></Label>\n'
        )
    xmlpost = (
        "  </LabelTable>  <StudyMetaDataLinkSet>  </StudyMetaDataLinkSet>  "
        "<VolumeType><![CDATA[Label]]></VolumeType>   </VolumeInformation></CaretExtension>"
    )
    xml = xmlpre + "\n" + body + "\n" + xmlpost + "\n              "
    img_in.header.extensions.append(nib.nifti1.Nifti1Extension(0, bytes(xml, "utf-8")))
    return img_in


def lps_shape_spacing(img: nib.Nifti1Image) -> tuple[tuple[int, int, int], np.ndarray]:
    """Shape and spacing of ``img`` after reorientation to LPS (no data copy)."""
    transform = ornt_transform(io_orientation(img.affine), axcodes2ornt("LPS"))
    shape = [0, 0, 0]
    for i in range(3):
        shape[int(transform[i, 0])] = img.shape[i]
    return tuple(shape), lps_spacing(img)
