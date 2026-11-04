"""Import BOA's vendored TotalSegmentator (v2.12.0) with heavy dependencies stubbed.

Used by the equivalence tests: TotalSegmentator's own ``nnUNet_predict_image`` /
``totalsegmentator`` are run with the nnU-Net call replaced by ``fake_network``
(through real temporary NIfTI files, as in production), and compared with
BOA_light's in-memory engine. Requires a BOA checkout: set ``BOA_SRC`` to the
repository root.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np

from fake_model import fake_network


def _stub(name: str) -> types.ModuleType:
    m = mock.MagicMock(name=name)
    m.__path__ = []  # make it a package
    m.__spec__ = None
    sys.modules[name] = m
    return m


STUB_PREFIXES = (
    "torch", "nnunetv2", "requests", "tqdm", "xmltodict", "pydicom", "rt_utils",
    "highdicom", "cupy", "cucim", "totalsegmentator.custom_trainers", "totalsegmentator.dicom_io",
    "totalsegmentator.statistics", "totalsegmentator.preview", "p_tqdm", "dicom2nifti",
    "dotenv", "matplotlib", "plotly", "jinja2", "weasyprint", "boa_contrast", "kaleido", "seaborn",
    "body_composition_analysis.report.plots", "dicomweb_client", "pydicom_seg", "smbprotocol",
)  # fmt: skip


class _StubLoader:
    def create_module(self, spec):
        m = mock.MagicMock(name=spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        pass


class _StubFinder:
    def find_spec(self, fullname, path=None, target=None):
        import importlib.machinery

        if fullname.startswith(STUB_PREFIXES):
            return importlib.machinery.ModuleSpec(fullname, _StubLoader(), is_package=True)
        return None


def load_reference():
    root = os.environ.get("BOA_SRC")
    if not root or not Path(root).is_dir():
        return None
    ext = Path(root) / "body_organ_analysis" / "_external"
    import fake_sitk

    sys.modules["SimpleITK"] = fake_sitk
    sys.meta_path.insert(0, _StubFinder())
    joblib = types.ModuleType("joblib")
    joblib.Parallel = lambda n_jobs=1: (lambda gen: [f() for f in gen])
    joblib.delayed = lambda fn: (lambda *a, **k: (lambda: fn(*a, **k)))
    sys.modules["joblib"] = joblib

    # body_composition_analysis.io (needed by nnunet.py) imports SimpleITK only for unused helpers
    sys.path.insert(0, str(ext))
    import totalsegmentator.nnunet as ts_nnunet  # noqa: E402
    import totalsegmentator.python_api as ts_api  # noqa: E402

    def fake_nnunetv2_predict(dir_in, dir_out, task_id, *args, **kwargs):
        for f in sorted(Path(dir_in).glob("*_0000.nii.gz")):
            img = nib.load(f)
            arr = img.get_fdata().astype(np.float32).transpose(2, 1, 0)
            spacing = [float(s) for s in img.header.get_zooms()[:3]][::-1]
            seg = fake_network(task_id, arr, spacing).transpose(2, 1, 0)
            case = f.name.replace("_0000.nii.gz", "")
            nib.save(nib.Nifti1Image(seg, img.affine), Path(dir_out) / f"{case}.nii.gz")

    ts_nnunet.nnUNetv2_predict = fake_nnunetv2_predict
    for fn in ["setup_nnunet", "setup_totalseg", "download_pretrained_weights", "send_usage_stats",
               "increase_prediction_counter", "set_config_key", "show_license_info"]:  # fmt: skip
        setattr(ts_api, fn, lambda *a, **k: None)
    ts_api.get_config_key = lambda *a, **k: True
    ts_api.select_device = lambda d: "cpu"
    ts_api.validate_device_type_api = lambda d: None
    # BOA itself (imports body_composition_analysis / totalsegmentator from _external)
    sys.path.insert(0, str(Path(root)))
    if not hasattr(np, "float_"):  # BOA pins numpy<2
        np.float_ = np.float64
    import body_organ_analysis  # noqa: F401
    import body_organ_analysis.commands as boa_commands
    from body_composition_analysis.infer import infer as bca_infer
    from body_composition_analysis.report import builder

    bca_infer.download_bca_weights = lambda *a, **k: None

    class _Fig:
        def to_image(self, **k):
            return b""

    builder.create_aggregation_image = lambda **k: np.zeros((2, 2, 3), np.uint8)
    builder.create_tissue_summary = lambda *a, **k: _Fig()
    builder.create_tissue_heatmaps = lambda *a, **k: []
    builder.create_equidistant_overview = lambda *a, **k: []
    builder.to_png_data_url = lambda img: ""
    return types.SimpleNamespace(nnunet=ts_nnunet, api=ts_api, boa=boa_commands)
