#!/usr/bin/env python
"""Sanity check of the in-memory nnU-Net path on a real CT (needs torch + nnunetv2).

    python scripts/check_predictor.py CT.nii.gz [--task 298] [--device cuda]

1. Runs one network the way TotalSegmentator does (temporary NIfTI file ->
   ``nnUNetPredictor.predict_from_files``) and the BOA_light way (in-memory,
   on-device argmax), and checks that the results are voxel-identical.
2. Prints the timing of both.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from boa_light import weights
from boa_light.engine import ImageContext, SegmentationEngine
from boa_light.models import NETWORKS, ROUGH, TASKS, Task
from boa_light.predictor import ModelPool


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("ct", type=Path)
    p.add_argument("--task", type=int, default=298, help="network id (default 298 = 6 mm total, fast)")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    task = {t.task_ids[0]: t for t in [ROUGH, *TASKS.values()] if not t.multimodel}.get(args.task)
    if task is None:
        task = Task("check", "total", (args.task,), resample=(1.5, 1.5, 1.5), step_size=0.8)
    weights.download(args.task)
    pool = ModelPool(args.device)
    eng = SegmentationEngine(ImageContext(nib.load(args.ct)), pool.predict)
    prep = eng.prepare(task)
    img = prep.img_in_rsp

    # 1) BOA_light: in-memory + on-device argmax
    pool.predict(args.task, img, task.folds, task.step_size)  # warm-up / load
    t = time.perf_counter()
    seg_fast = pool.predict(args.task, img, task.folds, task.step_size)
    t_fast = time.perf_counter() - t
    # 2) in-memory, nnU-Net export
    pool.fast_export = False
    t = time.perf_counter()
    seg_mem = pool.predict(args.task, img, task.folds, task.step_size)
    t_mem = time.perf_counter() - t
    # 3) file based, like TotalSegmentator's nnUNetv2_predict
    predictor = pool.get(args.task, task.folds, task.step_size)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        nib.save(img, tmp / "s01_0000.nii.gz")
        t = time.perf_counter()
        predictor.predict_from_files(str(tmp), str(tmp), save_probabilities=False, overwrite=True,
                                     num_processes_preprocessing=1, num_processes_segmentation_export=1)  # fmt: skip
        t_file = time.perf_counter() - t
        seg_file = np.asanyarray(nib.load(tmp / "s01.nii.gz").dataobj).astype(np.uint8)

    print(f"network {args.task} ({NETWORKS[args.task].folder}) on {pool.device}, input {img.shape}")
    print(f"  in-memory + GPU argmax : {t_fast:7.2f} s")
    print(f"  in-memory + nnU-Net export: {t_mem:7.2f} s")
    print(f"  file based (TotalSegmentator style): {t_file:7.2f} s")
    ok = np.array_equal(seg_fast, seg_mem) and np.array_equal(seg_mem, seg_file)
    print("  identical:", ok, "| labels:", np.unique(seg_fast).size - 1)
    if not ok:
        print("  agreement fast/mem:", (seg_fast == seg_mem).mean(), " mem/file:", (seg_mem == seg_file).mean())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
