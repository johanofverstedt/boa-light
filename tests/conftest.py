import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

# Always exercise the multi-threaded resampling in the tests (also for small images), so
# the BOA/TotalSegmentator equivalence tests cover it.
import boa_light.imaging as _imaging  # noqa: E402

_imaging._PARALLEL_MIN_VOXELS = 0
_imaging.set_resample_workers(3)

TS_EXAMPLE = os.environ.get(
    "TS_EXAMPLE_CT", str(Path(__file__).parent / "data" / "example_ct.nii.gz")
)


def _synthetic_ct() -> nib.Nifti1Image:
    """Torso-like phantom: body ellipse, fat ring, 'organs', 'spine' and 'lungs'."""
    shape = (90, 80, 70)
    x, y, z = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing="ij")
    ct = np.full(shape, -1000.0)
    body = (x / 0.9) ** 2 + (y / 0.75) ** 2 < 1
    ct[body] = -100  # fat
    ct[(x / 0.8) ** 2 + (y / 0.65) ** 2 < 1] = 40  # muscle / organs
    ct[((x - 0.35) / 0.25) ** 2 + (y / 0.3) ** 2 + ((z - 0.4) / 0.5) ** 2 < 1] = -850  # lung
    ct[((x + 0.35) / 0.25) ** 2 + (y / 0.3) ** 2 + ((z - 0.4) / 0.5) ** 2 < 1] = -850
    ct[(x / 0.08) ** 2 + ((y + 0.5) / 0.08) ** 2 < 1] = 700  # spine
    rng = np.random.default_rng(0)
    ct = ct + rng.normal(0, 15, shape)
    affine = np.diag([1.2, 1.2, 2.5, 1.0])
    affine[:3, 3] = [-50, -40, 100]
    return nib.Nifti1Image(np.round(ct).astype(np.int16), affine)


def _variants(base: nib.Nifti1Image) -> dict[str, nib.Nifti1Image]:
    data = np.asanyarray(base.dataobj)
    out = {"RAS": base}
    # LPS: flip x and y
    aff = base.affine.copy()
    flip = np.diag([-1, -1, 1, 1]).astype(float)
    flip[0, 3] = data.shape[0] - 1
    flip[1, 3] = data.shape[1] - 1
    out["LPS"] = nib.Nifti1Image(np.ascontiguousarray(data[::-1, ::-1, :]), aff @ flip)
    # Permuted axes (A, R, S) with anisotropic in-plane spacing
    perm = np.ascontiguousarray(data.transpose(1, 0, 2))
    aff_p = np.zeros((4, 4))
    aff_p[1, 0] = 1.1
    aff_p[0, 1] = 1.3
    aff_p[2, 2] = 2.0
    aff_p[3, 3] = 1
    out["ARS"] = nib.Nifti1Image(perm, aff_p)
    return out


@pytest.fixture(scope="session")
def ct_variants() -> dict[str, nib.Nifti1Image]:
    if Path(TS_EXAMPLE).is_file():
        img = nib.load(TS_EXAMPLE)
        data = np.asanyarray(img.dataobj)[:, :, 20:80].astype(np.int16)
        affine = np.diag([1.2, 1.2, 2.5, 1.0])
        base = nib.Nifti1Image(data, affine)
    else:
        base = _synthetic_ct()
    return _variants(base)


@pytest.fixture(scope="session")
def reference():
    from ts_reference import load_reference

    ref = load_reference()
    if ref is None:
        pytest.skip("BOA_SRC not set: TotalSegmentator reference not available")
    return ref
