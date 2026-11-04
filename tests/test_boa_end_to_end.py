"""Full pipeline: BOA's ``analyze_ct`` vs ``boa_light.analyze`` with the same fake network.

Compares every segmentation, ``tissues``, ``total-measurements.json``,
``bca-measurements.json`` and the sheets of ``output.xlsx``.
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from conftest import TS_EXAMPLE
from fake_model import fake_predict_fn

from boa_light.pipeline import Options, analyze

MODELS = ["total", "bca", "lung_vessels", "liver_vessels", "pleural_pericard_effusion", "hip_implant"]


@pytest.fixture(scope="module")
def tall_ct(tmp_path_factory) -> Path:
    if Path(TS_EXAMPLE).is_file():
        data = np.asanyarray(nib.load(TS_EXAMPLE).dataobj).astype(np.int16)
    else:
        from conftest import _synthetic_ct

        data = np.asanyarray(_synthetic_ct().dataobj)
    # LPS orientation, 4 mm slices -> ~450 mm coverage so abdomen + thorax are detected
    aff = np.diag([-1.5, -1.5, 4.0, 1.0])
    p = tmp_path_factory.mktemp("in") / "ct.nii.gz"
    nib.save(nib.Nifti1Image(data, aff), p)
    return p


@pytest.fixture(scope="module")
def both(reference, tall_ct, tmp_path_factory):
    boa_out = tmp_path_factory.mktemp("boa")
    reference.boa.analyze_ct(
        input_folder=tall_ct,
        processed_output_folder=boa_out,
        excel_output_folder=boa_out,
        models=set(MODELS),
        compute_contrast_information=False,
        total_preview=False,
        device="cpu",
        bca_pdf=False,
        recompute=True,
        cnr_adjustment=True,
    )
    bl_out = tmp_path_factory.mktemp("bl")
    analyze(tall_ct, bl_out, Options(models=MODELS, contrast=False, cnr_adjustment=True), predict_fn=fake_predict_fn)
    return boa_out, bl_out


@pytest.mark.parametrize(
    "name",
    ["total", "lung_vessels", "liver_vessels", "pleural_pericard_effusion", "hip_implant",
     "body_parts", "body_regions", "tissues", "ct_pfav"],
)  # fmt: skip
def test_segmentations(both, name):
    boa_out, bl_out = both
    a = nib.load(boa_out / f"{name}.nii.gz")
    b = nib.load(bl_out / f"{name}.nii.gz")
    np.testing.assert_array_equal(np.asanyarray(a.dataobj), np.asanyarray(b.dataobj))
    np.testing.assert_allclose(a.affine, b.affine, atol=1e-4)


def _compare(a, b, path=""):
    if isinstance(a, dict):
        for k in a:
            assert k in b, f"missing key {path}/{k}"
            _compare(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _compare(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and b is not None:
        assert np.isclose(a, b, rtol=1e-9, atol=1e-9), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a} != {b}"


def test_total_measurements(both):
    boa_out, bl_out = both
    a = json.loads((boa_out / "total-measurements.json").read_text())
    b = json.loads((bl_out / "total-measurements.json").read_text())
    from boa_light.class_maps import class_map

    # BOA leaks unrelated class maps with the same prefix into "total" (v1_*, mr_*, ...): ignore those.
    for model, regions in a["segmentations"].items():
        valid = set(class_map[model].values()) | {"autochthon"} | {k for k in regions if k.startswith("ct_pfav")}
        a["segmentations"][model] = {k: v for k, v in regions.items() if k in valid}
    # BOA does not measure lung_vessels (looks for lung_vessels_airways.nii.gz): BOA_light does.
    assert "lung_vessels" not in a["segmentations"]
    assert "lung_vessels" in b["segmentations"]
    _compare(a, b)


def test_bca_measurements(both):
    boa_out, bl_out = both
    a = json.loads((boa_out / "bca-measurements.json").read_text())
    b = json.loads((bl_out / "bca-measurements.json").read_text())
    assert a["body_parts"]["abdomen"] and a["body_parts"]["thorax"], "fake BCA should hit the aggregation paths"
    _compare(a, b)
    va, vb = boa_out / "vertebrae.json", bl_out / "vertebrae.json"
    assert va.exists() == vb.exists()
    if va.exists():
        assert json.loads(va.read_text()) == json.loads(vb.read_text())


@pytest.mark.parametrize(
    "sheet", ["bca-aggregated-measurements", "bca-slice-measurements", "bca-slice-measurements_no_ext", "cnr-adjusted"]
)
def test_excel_sheets(both, sheet):
    boa_out, bl_out = both
    a = pd.read_excel(boa_out / "output.xlsx", sheet_name=sheet)
    b = pd.read_excel(bl_out / "output.xlsx", sheet_name=sheet)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_excel_info_and_regions(both):
    boa_out, bl_out = both
    a = pd.read_excel(boa_out / "output.xlsx", sheet_name="info", header=None, index_col=0)[1]
    b = pd.read_excel(bl_out / "output.xlsx", sheet_name="info", header=None, index_col=0)[1]
    for key in ["Noise", "CNRAorta", "CNRVCI", "CNRPortalSplenicVein", "MaxAxisL3_cm", "MinAxisL3_cm", "MeanAxisL3_cm"]:
        assert (key in a.index) == (key in b.index), key
        if key in a.index:
            assert np.isclose(float(a[key]), float(b[key])), key
    ra = pd.read_excel(boa_out / "output.xlsx", sheet_name="regions-statistics")
    rb = pd.read_excel(bl_out / "output.xlsx", sheet_name="regions-statistics")
    key = ["ModelName", "BodyRegion"]
    merged = ra.merge(rb, on=key, suffixes=("_a", "_b"))
    assert len(merged) > 100
    for col in ["VolumeMl", "MeanHU", "StdHU", "MedianHU", "CNR"]:
        np.testing.assert_allclose(merged[f"{col}_a"], merged[f"{col}_b"], rtol=1e-9, equal_nan=True)


# --------------------------------------------------------------------------- #
# --bca-skip-body-parts: same as BOA minus the outputs that need body_parts
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def skip_parts(both, tall_ct, tmp_path_factory):
    boa_out, _ = both
    out = tmp_path_factory.mktemp("bl_skip")
    calls = []

    def recording_predict(task_id, img, folds, step):
        calls.append(task_id)
        return fake_predict_fn(task_id, img, folds, step)

    analyze(
        tall_ct, out,
        Options(models=MODELS, contrast=False, cnr_adjustment=True, bca_skip_body_parts=True),
        predict_fn=recording_predict,
    )  # fmt: skip
    return boa_out, out, calls


def test_skip_body_parts_runs_no_body_parts_model(skip_parts):
    _, out, calls = skip_parts
    assert 543 not in calls and 542 in calls
    assert not (out / "body_parts.nii.gz").exists()
    for name in ["body_regions", "tissues", "total"]:
        a = nib.load(skip_parts[0] / f"{name}.nii.gz")
        b = nib.load(out / f"{name}.nii.gz")
        np.testing.assert_array_equal(np.asanyarray(a.dataobj), np.asanyarray(b.dataobj))


def test_skip_body_parts_bca_json(skip_parts):
    boa_out, out, _ = skip_parts
    a = json.loads((boa_out / "bca-measurements.json").read_text())
    b = json.loads((out / "bca-measurements.json").read_text())
    a.pop("slices_no_extremities")
    for group in a["aggregated"].values():
        group.pop("measurements_no_extremities")
    assert "slices_no_extremities" not in b
    assert all("measurements_no_extremities" not in g for g in b["aggregated"].values())
    _compare(a, b)


def test_skip_body_parts_excel(skip_parts):
    boa_out, out, _ = skip_parts
    sheets = pd.ExcelFile(out / "output.xlsx").sheet_names
    assert "bca-slice-measurements_no_ext" not in sheets
    a = pd.read_excel(boa_out / "output.xlsx", sheet_name="bca-aggregated-measurements")
    a = a[~a.BodyPart.str.endswith("_NoExtremities")].reset_index(drop=True)
    b = pd.read_excel(out / "output.xlsx", sheet_name="bca-aggregated-measurements")
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    for sheet in ["bca-slice-measurements", "cnr-adjusted"]:
        pd.testing.assert_frame_equal(
            pd.read_excel(boa_out / "output.xlsx", sheet_name=sheet),
            pd.read_excel(out / "output.xlsx", sheet_name=sheet),
            check_dtype=False,
        )
    ia = pd.read_excel(boa_out / "output.xlsx", sheet_name="info", header=None, index_col=0)[1]
    ib = pd.read_excel(out / "output.xlsx", sheet_name="info", header=None, index_col=0)[1]
    assert "MaxAxisL3_cm" in ia.index and "MaxAxisL3_cm" not in ib.index
    for key in ["Noise", "CNRAorta", "CNRVCI"]:
        assert np.isclose(float(ia[key]), float(ib[key]))
