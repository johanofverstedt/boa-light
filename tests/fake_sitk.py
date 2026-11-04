"""Minimal numpy/nibabel-backed stand-in for the parts of SimpleITK BOA uses.

Only for running BOA's own code as a reference in the tests (SimpleITK is not
needed by BOA_light itself).
"""

from __future__ import annotations

import nibabel as nib
import numpy as np

sitkLinear = 1
_FLIP = np.diag([-1.0, -1.0, 1.0])


class Image:
    def __init__(self, arr: np.ndarray):
        self._arr = np.asarray(arr)
        self._spacing = (1.0, 1.0, 1.0)
        self._origin = (0.0, 0.0, 0.0)
        self._direction = tuple(np.eye(3).flatten())

    def GetSize(self):
        return tuple(int(s) for s in self._arr.shape[::-1])

    def GetDepth(self):
        return int(self._arr.shape[0])

    def GetSpacing(self):
        return tuple(float(s) for s in self._spacing)

    def SetSpacing(self, s):
        self._spacing = tuple(float(v) for v in s)

    def GetOrigin(self):
        return self._origin

    def SetOrigin(self, o):
        self._origin = tuple(float(v) for v in o)

    def GetDirection(self):
        return self._direction

    def SetDirection(self, d):
        self._direction = tuple(float(v) for v in np.asarray(d).flatten())

    def CopyInformation(self, other: Image):
        assert self._arr.shape == other._arr.shape
        self._spacing, self._origin, self._direction = other._spacing, other._origin, other._direction

    def TransformContinuousIndexToPhysicalPoint(self, idx):
        d = np.asarray(self._direction).reshape(3, 3)
        return tuple(np.asarray(self._origin) + d @ (np.asarray(self._spacing) * np.asarray(idx, float)))

    def _affine_ras(self):
        d = np.asarray(self._direction).reshape(3, 3)
        aff = np.eye(4)
        aff[:3, :3] = _FLIP @ (d * np.asarray(self._spacing))
        aff[:3, 3] = _FLIP @ np.asarray(self._origin)
        return aff


def GetArrayFromImage(img: Image) -> np.ndarray:
    return img._arr.copy()


def GetArrayViewFromImage(img: Image) -> np.ndarray:
    return img._arr


def GetImageFromArray(arr) -> Image:
    return Image(np.asarray(arr))


def ReadImage(path) -> Image:
    nimg = nib.load(str(path))
    img = Image(np.asanyarray(nimg.dataobj).transpose(2, 1, 0))
    spacing = np.array(nimg.header.get_zooms()[:3], dtype=float)
    rzs = _FLIP @ nimg.affine[:3, :3]
    img._spacing = tuple(spacing)
    img._direction = tuple((rzs / np.sqrt((rzs**2).sum(axis=0))).flatten())
    img._origin = tuple(_FLIP @ nimg.affine[:3, 3])
    return img


def WriteImage(img: Image, path, useCompression=False, *args):
    nib.save(nib.Nifti1Image(img._arr.transpose(2, 1, 0), img._affine_ras()), str(path))
