"""Shared nnU-Net preprocessing for the five "total" networks == nnU-Net's own preprocessing.

Runs nnU-Net's real ``DefaultPreprocessor.run_case_npy`` (torch stubbed, it is not used
there) with the plans of the downloaded TotalSegmentator part models, once plainly
and once through BOA_light's crop cache, and requires bit-identical results.
Needs ``NNUNET_SRC`` (nnU-Net checkout), ``ACVL_SRC``, ``BG_SRC`` (acvl_utils,
batchgenerators checkouts) and ``BOA_LIGHT_WEIGHTS`` with networks 291-295.
"""

import copy
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

SRCS = [os.environ.get(k) for k in ("NNUNET_SRC", "ACVL_SRC", "BG_SRC")]
WEIGHTS = os.environ.get("BOA_LIGHT_WEIGHTS")
pytestmark = pytest.mark.skipif(
    not all(SRCS) or not WEIGHTS, reason="nnU-Net sources / weights not configured"
)


def test_crop_cache_identical_subprocess():
    """Run the check in a fresh interpreter (other tests install module stubs)."""
    import subprocess

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent), *sys.path])
    r = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


def nnunet():
    try:
        import torch  # noqa: F401

        real_torch = True
    except ImportError:
        real_torch = False
    if not real_torch:
        import nnunet_stub

        nnunet_stub.install()
    sys.path[:0] = SRCS
    import nnunetv2.preprocessing.preprocessors.default_preprocessor as dp
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    return dp, PlansManager


def _ct_at_1p5mm():
    from conftest import TS_EXAMPLE, _synthetic_ct

    from boa_light.imaging import as_closest_canonical, change_spacing

    img = nib.load(TS_EXAMPLE) if Path(TS_EXAMPLE).is_file() else _synthetic_ct()
    img = nib.Nifti1Image(np.asanyarray(img.dataobj).astype(np.float64), img.affine)
    return change_spacing(as_closest_canonical(img), [1.5, 1.5, 1.5], order=3, dtype=np.int32)


def _ct_at_5mm_slices():
    from conftest import TS_EXAMPLE, _synthetic_ct

    from boa_light.imaging import as_closest_canonical, change_spacing

    img = nib.load(TS_EXAMPLE) if Path(TS_EXAMPLE).is_file() else _synthetic_ct()
    img = as_closest_canonical(nib.Nifti1Image(np.asanyarray(img.dataobj).astype(np.float64), img.affine))
    z = [float(v) for v in img.header.get_zooms()[:2]]
    return change_spacing(img, z + [5.0], order=3, dtype=np.int32)


def check_crop_cache_identical(task_ids=(291, 292, 293, 294, 295), make_img=None):
    dp, PlansManager = nnunet()
    from boa_light import predictor as P
    from boa_light.models import NETWORKS
    from boa_light.weights import model_folder

    img = (make_img or _ct_at_1p5mm)()
    data = np.asanyarray(img.dataobj).astype(np.float32)
    array = np.ascontiguousarray(data.transpose(2, 1, 0))[None]
    spacing = [float(s) for s in img.header.get_zooms()[:3]][::-1]

    original = dp.crop_to_nonzero
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return original(*a, **k)

    dp.crop_to_nonzero = counting
    try:
        reference, cached = [], []
        configs = []
        for tid in task_ids:
            folder = model_folder(NETWORKS[tid], Path(WEIGHTS))
            pm = PlansManager(json.loads((folder / "plans.json").read_text()))
            cm = pm.get_configuration("3d_fullres")
            ds = json.loads((folder / "dataset.json").read_text())
            configs.append((pm, cm, ds))
            pre = cm.preprocessor_class(verbose=False)
            reference.append(pre.run_case_npy(array, None, {"spacing": spacing}, pm, cm, ds))
        n_plain = calls["n"]

        P._install_crop_cache()  # wraps the (counting) function
        calls["n"] = 0
        for pm, cm, ds in configs:
            P._CROP_CACHE["key"] = (img, tuple(pm.transpose_forward))
            pre = cm.preprocessor_class(verbose=False)
            cached.append(pre.run_case_npy(array, None, {"spacing": spacing}, pm, cm, ds))
        P._CROP_CACHE["key"] = P._CROP_CACHE["entry"] = None
    finally:
        dp.crop_to_nonzero = original

    assert n_plain == len(task_ids) and calls["n"] == 1, "crop_to_nonzero should run once for all networks"
    for (d0, s0, p0), (d1, s1, p1) in zip(reference, cached):
        assert d0.dtype == d1.dtype and np.array_equal(d0, d1)
        assert np.array_equal(s0, s1)
        assert copy.deepcopy(p0).keys() == p1.keys()
        for k in p0:
            assert np.array_equal(np.asarray(p0[k], dtype=object), np.asarray(p1[k], dtype=object)), k
    # the five networks really normalise differently (otherwise the test proves little)
    assert not all(np.array_equal(reference[0][0], r[0]) for r in reference[1:])


if __name__ == "__main__":
    import boa_light.imaging  # noqa: F401  (conftest settings are not needed here)

    check_crop_cache_identical()
    check_crop_cache_identical((543, 542), _ct_at_5mm_slices)  # body_parts + body_regions
    print("OK")
