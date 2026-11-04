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
import os

import nibabel as nib
import numpy as np
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from scipy import ndimage

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #


_RESAMPLE_WORKERS = max(1, os.cpu_count() or 1)
_PARALLEL_MIN_VOXELS = 1_000_000  # below this, threading overhead is not worth it


def set_resample_workers(n: int | None) -> None:
    """Number of threads used by :func:`parallel_zoom` (default: all CPUs)."""
    global _RESAMPLE_WORKERS
    _RESAMPLE_WORKERS = max(1, int(n)) if n else max(1, os.cpu_count() or 1)


def _chunks(n: int, parts: int) -> list[tuple[int, int]]:
    parts = max(1, min(parts, n))
    edges = np.linspace(0, n, parts + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def parallel_zoom(img: np.ndarray, zoom, order: int = 0, workers: int | None = None) -> np.ndarray:
    """Bit-identical, multi-threaded ``scipy.ndimage.zoom(img, zoom, order=order, mode="nearest")``.

    scipy's zoom is single-threaded. It consists of (1) a spline prefilter that is applied
    line by line along each axis and (2) interpolation of every output voxel from the
    filtered coefficients. Both steps release the GIL and are split into independent
    pieces here: the prefilter by blocks of lines, the interpolation by blocks of output
    slices (using ``zoom_shift`` with an integer shift, so the sampling coordinates are
    computed exactly as in the single call). The result is identical to ``ndimage.zoom``.
    Falls back to ``ndimage.zoom`` if scipy's internals are not as expected.
    """
    workers = workers or _RESAMPLE_WORKERS
    img = np.asarray(img)
    if workers <= 1 or img.ndim != 3 or img.size < _PARALLEL_MIN_VOXELS or np.iscomplexobj(img):
        return ndimage.zoom(img, zoom, mode="nearest", order=order)
    try:
        from scipy.ndimage import _nd_image, _ni_support
    except ImportError:  # pragma: no cover
        return ndimage.zoom(img, zoom, mode="nearest", order=order)

    zoom = _ni_support._normalize_sequence(zoom, img.ndim)
    output_shape = tuple(int(round(ii * jj)) for ii, jj in zip(img.shape, zoom))
    if all(z == 1 for z in zoom):  # scipy's early exit
        return img.copy()
    output = np.empty(output_shape, dtype=img.dtype)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        if order > 1:
            npad = 12  # _prepad_for_spline_filter for mode="nearest"
            filtered = np.pad(img, npad, mode="edge").astype(np.float64)
            for axis in range(3):
                split_axis = (axis + 1) % 3 if filtered.shape[(axis + 1) % 3] >= filtered.shape[(axis + 2) % 3] \
                    else (axis + 2) % 3  # fmt: skip

                def run(a, b, axis=axis, split_axis=split_axis):
                    sl = [slice(None)] * 3
                    sl[split_axis] = slice(a, b)
                    view = filtered[tuple(sl)]
                    ndimage.spline_filter1d(view, order, axis, output=view, mode="nearest")

                list(pool.map(lambda ab: run(*ab), _chunks(filtered.shape[split_axis], workers * 2)))
        else:
            npad = 0
            filtered = img
        mode = _ni_support._extend_mode_to_code("nearest")
        zoom_div = np.array(output_shape) - 1
        zoom_nom = np.array(img.shape) - 1
        zoom_arr = np.divide(zoom_nom, zoom_div, out=np.ones(3, dtype=np.float64), where=zoom_div != 0)
        zoom_arr = np.ascontiguousarray(zoom_arr)

        def interp(a, b):
            shift = np.ascontiguousarray([float(a), 0.0, 0.0])
            _nd_image.zoom_shift(filtered, zoom_arr, shift, output[a:b], order, mode, 0.0, npad, False)

        try:
            list(pool.map(lambda ab: interp(*ab), _chunks(output_shape[0], workers * 2)))
        except TypeError:  # private API changed -> plain scipy
            return ndimage.zoom(img, zoom, mode="nearest", order=order)
    return output


def resample_img(img: np.ndarray, zoom, order: int = 0) -> np.ndarray:
    """``ndimage.zoom`` with mode="nearest" (TotalSegmentator ``resample_img``), multi-threaded."""
    return parallel_zoom(img, zoom, order=order)


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
    raw = np.asanyarray(img_in.dataobj)
    if order == 0 and raw.dtype.kind in "ui" and dtype is not None and np.dtype(dtype).kind in "ui" \
            and isinstance(img_in.dataobj, np.ndarray):
        # Label maps: nearest-neighbour interpolation copies values, so resampling the
        # integer array directly gives exactly what TotalSegmentator's float64 path gives
        # (8x less memory traffic).
        data = raw
    else:
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
