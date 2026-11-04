"""DICOM series input (optional; needs ``SimpleITK`` and ``pydicom``).

Port of BOA ``compute/io.get_image_info``: the series is read with SimpleITK,
validated (axial CT, no localizer/derived images) and the DICOM tags BOA reports in
the ``info`` sheet are collected.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def _sitk_to_nib(image) -> nib.Nifti1Image:
    import SimpleITK as sitk

    points = [image.TransformContinuousIndexToPhysicalPoint(p) for p in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0))]
    c = np.array(points)
    affine = np.concatenate(
        [np.concatenate([c[0:3] - c[3:], c[3:]], axis=0), [[0.0], [0.0], [0.0], [1.0]]], axis=1
    ).T
    affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine  # LPS -> RAS
    return nib.Nifti1Image(sitk.GetArrayFromImage(image).transpose(), affine)


def _da(value) -> date | None:
    try:
        from pydicom.valuerep import DA

        return DA(value) if value else None
    except Exception:
        return None


def _first(value):
    from pydicom.multival import MultiValue

    return value[0] if isinstance(value, MultiValue) and len(value) else value


def _plain(value):
    """Convert pydicom values to JSON/Excel friendly python types."""
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def validate_dicom(dcm, num_dicoms: int, minimum_images: int = 10, axial_normal_z_min: float = 0.85) -> str | None:
    if num_dicoms < minimum_images:
        return f"The series has less than {minimum_images} instances: {num_dicoms}."
    modality = dcm.get("Modality")
    if modality is not None and modality != "CT":
        return f"The modality is not CT: {modality}"
    iop = dcm.get("ImageOrientationPatient")
    if iop is not None and len(iop) == 6:
        normal = np.cross(np.asarray(iop[:3], float), np.asarray(iop[3:], float))
        ax, ay, az = np.abs(normal)
        if not (az >= ax and az >= ay):
            return f"Image plane is not axial. IOP={list(iop)}"
        if abs(normal[2]) < axial_normal_z_min:
            return f"Axial but tilted beyond tolerance: |normal_z|={abs(normal[2]):.3f}. IOP={list(iop)}"
    hits = {"LOCALIZER", "REFORMATTED", "DERIVED", "PROJECTION IMAGE"} & set(dcm.get("ImageType") or ())
    if hits:
        return f"ImageType contains disqualifying marker(s) {hits}"
    return None


def read_dicom_series(folder: Path) -> tuple[nib.Nifti1Image, list[dict[str, Any]]]:
    try:
        import pydicom
        import SimpleITK as sitk
    except ImportError as e:  # pragma: no cover
        raise ImportError("DICOM input needs `pip install boa-light[dicom]` (SimpleITK, pydicom)") from e

    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(folder))
    if not files:
        raise ValueError(f"No DICOM series found in {folder}")
    reader.SetFileNames(files)
    image = reader.Execute()
    dcm = pydicom.dcmread(files[0], stop_before_pixels=True)
    message = validate_dicom(dcm, len(files))
    if message:
        raise ValueError(message)

    series_date, birth_date = _da(dcm.get("SeriesDate")), _da(dcm.get("PatientBirthDate"))
    age = None
    if series_date and birth_date:
        age = (
            series_date.year
            - birth_date.year
            - ((series_date.month, series_date.day) < (birth_date.month, birth_date.day))
        )
    ordered = [
        ("StudyInstanceUID", dcm.get("StudyInstanceUID")),
        ("SeriesInstanceUID", dcm.get("SeriesInstanceUID")),
        ("Date", series_date.strftime("%d.%m.%Y") if series_date else None),
        ("AgeYears", age),
        ("Gender", dcm.get("PatientSex")),
        ("AccessionNumber", dcm.get("AccessionNumber")),
        ("SeriesNumber", dcm.get("SeriesNumber")),
        ("SeriesDescription", dcm.get("SeriesDescription")),
        ("Modality", dcm.get("Modality")),
        ("CTDIvol", dcm.get("CTDIvol")),
        ("ExposureTime", dcm.get("ExposureTime")),
        ("XRayTubeCurrent", dcm.get("XRayTubeCurrent")),
        ("Exposure", dcm.get("Exposure")),
        ("KVP", dcm.get("KVP")),
        ("SpiralPitchFactor", dcm.get("SpiralPitchFactor")),
        ("ConvolutionKernel", _first(dcm.get("ConvolutionKernel"))),
        ("SliceThickness", dcm.get("SliceThickness")),
    ]
    ps = dcm.get("PixelSpacing")
    if ps is not None and len(ps) >= 2:
        ordered += [("PixelSpacingX", ps[0]), ("PixelSpacingY", ps[1])]
    else:
        ordered.append(("PixelSpacing", ps))
    ordered.append(("ScanLength", dcm.get("ScanLength")))
    info = [{"name": k, "value": _plain(v)} for k, v in ordered]
    return _sitk_to_nib(image), info
