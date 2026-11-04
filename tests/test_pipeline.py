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
    res = analyze(p, out, Options(models="all", measurements=True, contrast=False, crop_source=crop_source), predict_fn=fake_predict_fn)
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
    res = analyze(_synthetic_ct(), None, Options(models="total+liver_vessels", measurements=True, contrast=False), predict_fn=fake_predict_fn)
    assert set(res.segmentations) == {"total", "liver_vessels"}
    assert res.measurements is not None and res.excel_path is None


def test_skip_body_parts_without_reference(tmp_path):
    res = analyze(_synthetic_ct(), tmp_path, Options(models="bca", measurements=True, contrast=False, bca_skip_body_parts=True),
                  predict_fn=fake_predict_fn)  # fmt: skip
    assert "body_parts" not in res.segmentations and "tissues" in res.segmentations
    assert "bca-slice-measurements_no_ext" not in pd.ExcelFile(tmp_path / "output.xlsx").sheet_names


def test_default_writes_only_labels(tmp_path):
    p = tmp_path / "ct.nii.gz"
    nib.save(_synthetic_ct(), p)
    out = tmp_path / "out"
    res = analyze(p, out, Options(models="total+bca+liver_vessels"), predict_fn=fake_predict_fn)
    files = sorted(f.name for f in out.iterdir())
    assert files == sorted(f"{n}.nii.gz" for n in
                           ["total", "liver_vessels", "body_parts", "body_regions", "tissues"])  # fmt: skip
    assert res.measurements is None and res.bca_measurements is None and res.excel_path is None


@pytest.mark.parametrize("fast_bca", [False, True])
def test_export_folds(tmp_path, fast_bca):
    p = tmp_path / "ct.nii.gz"
    nib.save(_synthetic_ct(), p)
    ref = analyze(p, tmp_path / "ref", Options(models="all", fast_bca=fast_bca), predict_fn=fake_predict_fn)
    out = tmp_path / "out"
    res = analyze(p, out, Options(models="all", fast_bca=fast_bca, export_folds=True), predict_fn=fake_predict_fn)
    expected = set() if fast_bca else {"body_parts", "body_regions"}
    if np.asanyarray(res.segmentations["pleural_pericard_effusion"].dataobj).any():  # crop may be empty
        expected.add("pleural_pericard_effusion")
    assert set(res.fold_segmentations) == expected
    for name in expected:
        assert sorted(res.fold_segmentations[name]) == [0, 1, 2, 3, 4]
        for f in range(5):
            img = nib.load(out / "folds" / f"{name}_fold{f}.nii.gz")
            assert img.shape == res.segmentations[name].shape
        # the ensemble output is unchanged by --export-folds
        np.testing.assert_array_equal(
            np.asanyarray(nib.load(out / f"{name}.nii.gz").dataobj),
            np.asanyarray(nib.load(tmp_path / "ref" / f"{name}.nii.gz").dataobj),
        )
    assert ref.fold_segmentations == {}


def test_export_folds_with_split():
    from boa_light.engine import ImageContext, SegmentationEngine
    from boa_light.models import TASKS

    ct = _synthetic_ct()
    data = np.asanyarray(ct.dataobj)
    tall = nib.Nifti1Image(np.concatenate([data] * 6, axis=2), ct.affine)
    eng = SegmentationEngine(ImageContext(tall), fake_predict_fn)
    prep = eng.prepare(TASKS["body_regions"])
    seg, folds = eng.infer(prep, force_split=True, return_folds=True)
    np.testing.assert_array_equal(seg, eng.infer(prep, force_split=True))
    assert sorted(folds) == [0, 1, 2, 3, 4] and all(f.shape == seg.shape for f in folds.values())
    # all-folds model (folds=None), uncropped here
    prep = eng.prepare(TASKS["pleural_pericard_effusion"])
    seg, folds = eng.infer(prep, return_folds=True)
    assert len(folds) == 5
    # single-fold models report no folds
    assert eng.infer(eng.prepare(TASKS["liver_vessels"]), return_folds=True)[1] is None


def test_cli_parser():
    args = build_parser().parse_args(["run", "in.nii.gz", "out", "-m", "total+bca", "--fast-bca"])
    assert args.models == "total+bca" and args.fast_bca and args.device == "auto"
    assert not args.measurements
    assert build_parser().parse_args(["run", "a", "b", "--reports"]).measurements


@pytest.mark.parametrize(
    "shape,zoom,order",
    [((61, 47, 33), (1.37, 0.8, 2.5), 3), ((40, 41, 90), (2.0, 2.0, 0.21), 3),
     ((50, 40, 30), (0.5, 3.0, 1.0), 0), ((33, 70, 20), (1.1, 1.1, 1.1), 1), ((5, 6, 7), (1.0, 1.0, 1.0), 3)],
)  # fmt: skip
def test_parallel_zoom_identical(shape, zoom, order):
    from scipy import ndimage

    from boa_light.imaging import parallel_zoom

    a = np.random.default_rng(1).normal(0, 300, shape).round()
    # (scipy's own all-ones early exit returns a copy; compare to that directly so that torch
    # stubs from other tests cannot interfere with scipy's array-API dispatch)
    ref = a.copy() if all(z == 1 for z in zoom) else ndimage.zoom(a, zoom, order=order, mode="nearest")
    for workers in (2, 3, 7):
        np.testing.assert_array_equal(parallel_zoom(a, zoom, order=order, workers=workers), ref)


def test_tissue_lut_matches_boa_rules():
    from boa_light.bca import _subclassify_tissues_masks, subclassify_tissues

    rng = np.random.default_rng(3)
    ct = rng.uniform(-1100, 3100, (60, 50, 40))
    ct.flat[:12] = [-1000, 3000, -190, -30, -29, 150, -191, -31, -28, 151, -1001, 3001]
    regions = rng.choice([0, *range(1, 12), 255], ct.shape).astype(np.uint8)
    r_img = nib.Nifti1Image(regions, np.eye(4))
    for data in (ct, np.round(ct).astype(np.int16)):
        ref = _subclassify_tissues_masks(data, regions, r_img)
        new = subclassify_tissues(nib.Nifti1Image(data, np.eye(4)), r_img, workers=3)
        np.testing.assert_array_equal(np.asanyarray(ref.dataobj), np.asanyarray(new.dataobj))


@pytest.mark.parametrize("seed", range(12))
def test_bca_postprocessing_matches_boa_reference(seed):
    from scipy import ndimage

    from boa_light import bca

    rng = np.random.default_rng(seed)
    shape = tuple(int(x) for x in rng.integers(20, 50, 3))
    m = np.zeros(shape, np.uint8)
    for lab in range(1, 7):
        noise = ndimage.gaussian_filter(rng.normal(size=shape), rng.uniform(1, 3))
        m[noise > np.quantile(noise, rng.uniform(0.6, 0.97))] = lab
    if seed % 3 == 0:
        m[:, :, 5:9] = 3  # label spanning the whole image along two axes
    if seed % 4 == 0:
        m[2:6, 2:6, 2:6] = 0
        m[3, 3, 3] = 4  # tiny object in a tiny box
    thr = int(rng.choice([5, 50, 300, 3000]))
    np.testing.assert_array_equal(
        bca.remove_small_labeled_objects(m, thr, workers=3), bca._remove_small_labeled_objects_reference(m, thr)
    )
    r = m.copy()
    r[rng.random(shape) < 0.002] = 7
    r[rng.random(shape) < 0.002] = 3
    np.testing.assert_array_equal(
        bca.postprocess_regions(r), bca.postprocess_regions(r, _filter=bca._filter_largest_unique_segment_reference)
    )


def test_merge_part_labels_matches_totalsegmentator_loop():
    from boa_light.engine import _merge_part_labels_reference, merge_part_labels, part_label_lut

    rng = np.random.default_rng(5)
    a, b = np.zeros((40, 30, 20), np.uint8), np.zeros((40, 30, 20), np.uint8)
    for tid in (291, 292, 293, 294, 295):
        seg = rng.integers(0, 30, a.shape).astype(np.uint8)  # includes labels the part map lacks
        _merge_part_labels_reference(a, seg, "total", tid)
        merge_part_labels(b, seg, part_label_lut("total", tid))
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("target", [(97, 81, 230), (40, 33, 20), (61, 50, 115)])
def test_label_resampling_integer_path_identical(target):
    """order-0 resampling of uint8 labels == TotalSegmentator's float64 path."""
    from scipy import ndimage

    from boa_light.imaging import change_spacing

    rng = np.random.default_rng(2)
    seg = rng.integers(0, 118, (61, 50, 115)).astype(np.uint8)
    img = nib.Nifti1Image(seg, np.diag([1.5, 1.5, 1.5, 1]))
    new = change_spacing(img, [1.5] * 3, target, order=0, dtype=np.uint8, force_affine=np.eye(4))
    zoom = np.array(target) / np.array(seg.shape)
    if np.all(zoom == 1):  # change_spacing returns the input unchanged (equal spacing)
        ref = seg
    else:
        ref = ndimage.zoom(seg.astype(np.float64), zoom, mode="nearest", order=0).astype(np.uint8)
    assert np.asanyarray(new.dataobj).dtype == np.uint8
    np.testing.assert_array_equal(np.asanyarray(new.dataobj), ref)
