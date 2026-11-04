"""BOA_light's engine vs. BOA's vendored TotalSegmentator, with the same fake network.

TotalSegmentator runs its real code path (temporary NIfTI files, cropping via a
rough segmentation, canonical reorientation, resampling, triple split, label
merging, undo steps); only the nnU-Net forward pass is replaced. Outputs must be
voxel-identical.
"""

import nibabel as nib
import numpy as np
import pytest
from fake_model import fake_predict_fn

from boa_light.engine import ImageContext, SegmentationEngine, roi_mask
from boa_light.models import ROUGH, TASKS, TOTAL_FAST

VARIANTS = ["RAS", "LPS", "ARS"]


def _assert_same(ts_img, bl_img):
    a = np.asanyarray(ts_img.dataobj).astype(np.uint8)
    b = np.asanyarray(bl_img.dataobj)
    assert a.shape == b.shape
    assert a.any(), "reference segmentation is empty; test is not informative"
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(ts_img.affine, bl_img.affine, atol=1e-5)


def _engine(img):
    return SegmentationEngine(ImageContext(img), fake_predict_fn)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("fast", [False, True])
def test_total(reference, ct_variants, variant, fast):
    img = ct_variants[variant]
    ts = reference.api.totalsegmentator(img, None, ml=True, fast=fast, task="total", quiet=True)
    bl = _engine(img).run(TOTAL_FAST if fast else TASKS["total"])
    _assert_same(ts, bl)


@pytest.mark.parametrize("variant", ["RAS", "ARS"])
def test_total_triple_split(reference, ct_variants, variant):
    img = ct_variants[variant]
    ts = reference.api.totalsegmentator(img, None, ml=True, task="total", force_split=True, quiet=True)
    bl = _engine(img).run(TASKS["total"], force_split=True)
    _assert_same(ts, bl)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "task", ["lung_vessels", "liver_vessels", "pleural_pericard_effusion", "liver_segments", "hip_implant"]
)
def test_cropped_tasks(reference, ct_variants, variant, task, tmp_path):
    img = ct_variants[variant]
    # BOA always passes an output file (this matters for empty crops)
    ts = reference.api.totalsegmentator(img, tmp_path / "out.nii.gz", ml=True, task=task, quiet=True)
    eng = _engine(img)
    rough = eng.run(ROUGH)
    t = TASKS[task]
    bl = eng.run(t, roi_mask(rough, t.crop), crop_key=("rough", t.crop))
    if not np.asanyarray(ts.dataobj).any():
        assert not np.asanyarray(bl.dataobj).any()
        return
    _assert_same(ts, bl)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("task", ["body_parts", "body_regions"])
@pytest.mark.parametrize("force_split", [False, True])
def test_bca_networks(reference, ct_variants, variant, task, force_split):
    img = ct_variants[variant]
    if force_split:  # needs enough slices at 5 mm: stack the volume along its last axis
        data = np.asanyarray(img.dataobj)
        img = nib.Nifti1Image(np.concatenate([data] * 6, axis=2), img.affine)
    t = TASKS[task]
    ts, _, _ = reference.nnunet.nnUNet_predict_image(
        file_in=img,
        file_out=None,
        task_id=t.task_ids[0],
        task_name=task,
        resample=5.0,
        resample_only_thickness=True,
        folds=list(t.folds),
        trainer="x",
        multilabel_image=True,
        force_split=force_split,
        quiet=True,
        device="cpu",
    )
    bl = _engine(img).run(t, force_split=force_split)
    _assert_same(ts, bl)


def test_shared_crop_cache(ct_variants):
    """lung_vessels and pleural effusion use the same crop -> prepared once."""
    eng = _engine(ct_variants["RAS"])
    rough = eng.run(ROUGH)
    p1 = eng.prepare(TASKS["lung_vessels"], roi_mask(rough, TASKS["lung_vessels"].crop), ("rough", TASKS["lung_vessels"].crop))
    p2 = eng.prepare(
        TASKS["pleural_pericard_effusion"],
        roi_mask(rough, TASKS["pleural_pericard_effusion"].crop),
        ("rough", TASKS["pleural_pericard_effusion"].crop),
    )
    assert p1.img_in is p2.img_in
