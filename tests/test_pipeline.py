"""Pipeline / CLI tests that need neither torch nor a BOA checkout."""

import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from conftest import _synthetic_ct
from fake_model import fake_predict_fn

from boa_light.cli import build_parser
from boa_light.models import networks_for, resolve_models
from boa_light.pipeline import Options, analyze


def test_resolve_models():
    tasks, bca = resolve_models("all")
    assert bca
    assert tasks[0] == "total" and tasks[-2:] == ["body_parts", "body_regions"]
    assert "heartchambers_highres" not in tasks
    assert resolve_models("liver_vessels") == (["liver_vessels"], False)
    assert resolve_models("bca") == (["total", "body_parts", "body_regions"], True)
    assert resolve_models("bca", bca_skip_body_parts=True) == (["total", "body_regions"], True)
    assert resolve_models("bca+body_parts", bca_skip_body_parts=True)[0] == ["total", "body_parts", "body_regions"]
    assert 543 not in networks_for(resolve_models("all", bca_skip_body_parts=True)[0])
    with pytest.raises(ValueError):
        resolve_models("heartchambers_highres")


def test_networks_for():
    assert networks_for(["total"]) == [291, 292, 293, 294, 295]
    assert networks_for(["lung_vessels", "pleural_pericard_effusion"]) == [258, 315, 298]
    # crop masks from "total": no rough model needed
    assert 298 not in networks_for(["total", "lung_vessels"], crop_source="total")
    assert networks_for(["total"], fast_total=True) == [297]


@pytest.mark.parametrize("crop_source", ["rough", "total"])
def test_pipeline_outputs(tmp_path, crop_source):
    ct = _synthetic_ct()
    p = tmp_path / "ct.nii.gz"
    nib.save(ct, p)
    out = tmp_path / "out"
    res = analyze(p, out, Options(models="all", contrast=False, crop_source=crop_source), predict_fn=fake_predict_fn)
    for name in ["total", "lung_vessels", "cerebral_bleed", "hip_implant", "liver_vessels",
                 "pleural_pericard_effusion", "body_parts", "body_regions", "tissues", "ct_pfav"]:  # fmt: skip
        img = nib.load(out / f"{name}.nii.gz")
        assert img.shape == ct.shape
        np.testing.assert_allclose(img.affine, ct.affine, atol=1e-5)
        assert img.get_data_dtype() == np.uint8
    m = json.loads((out / "total-measurements.json").read_text())
    assert set(m["segmentations"]) == {"total", "lung_vessels", "cerebral_bleed", "hip_implant",
                                       "liver_vessels", "pleural_pericard_effusion"}  # fmt: skip
    assert (out / "bca-measurements.json").exists()
    sheets = pd.ExcelFile(out / "output.xlsx").sheet_names
    assert sheets[:2] == ["info", "regions-statistics"]
    assert "bca-aggregated-measurements" in sheets
    stats = json.loads((out / "boa_light_stats.json").read_text())
    assert "_rough" in stats["inference_time_per_model"] or crop_source == "total"
    assert res.segmentations["total"] is not None


def test_in_memory_no_output(tmp_path):
    res = analyze(_synthetic_ct(), None, Options(models="total+liver_vessels", contrast=False), predict_fn=fake_predict_fn)
    assert set(res.segmentations) == {"total", "liver_vessels"}
    assert res.measurements is not None and res.excel_path is None


def test_skip_body_parts_without_reference(tmp_path):
    res = analyze(_synthetic_ct(), tmp_path, Options(models="bca", contrast=False, bca_skip_body_parts=True),
                  predict_fn=fake_predict_fn)  # fmt: skip
    assert "body_parts" not in res.segmentations and "tissues" in res.segmentations
    assert "bca-slice-measurements_no_ext" not in pd.ExcelFile(tmp_path / "output.xlsx").sheet_names


def test_cli_parser():
    args = build_parser().parse_args(["run", "in.nii.gz", "out", "-m", "total+bca", "--fast-bca"])
    assert args.models == "total+bca" and args.fast_bca and args.device == "auto"
