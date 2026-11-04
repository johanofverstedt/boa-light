#!/usr/bin/env python
"""Compare a BOA output directory with a BOA_light output directory.

Usage:
    python scripts/compare_outputs.py BOA_OUTPUT_DIR BOA_LIGHT_OUTPUT_DIR [--json report.json]

Reports, for every segmentation present in both directories, the fraction of
identical voxels and the per-label Dice; and for ``total-measurements.json`` /
``bca-measurements.json`` the largest relative difference of any numeric value.
Run BOA and BOA_light on the same CT (same device type) and expect identical
segmentations (agreement 1.0); tiny differences can come from different
nnU-Net / torch / cuDNN versions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

SEGMENTATIONS = [
    "total", "lung_vessels", "cerebral_bleed", "hip_implant", "liver_vessels",
    "pleural_pericard_effusion", "liver_segments", "body_parts", "body_regions", "tissues", "ct_pfav",
]  # fmt: skip


def dice_per_label(a: np.ndarray, b: np.ndarray) -> dict[int, float]:
    out = {}
    for label in np.union1d(np.unique(a), np.unique(b)):
        if label == 0:
            continue
        ma, mb = a == label, b == label
        denom = ma.sum() + mb.sum()
        out[int(label)] = float(2 * np.logical_and(ma, mb).sum() / denom) if denom else 1.0
    return out


def numeric_diffs(a, b, path="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) & set(b):
            numeric_diffs(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            numeric_diffs(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        rel = abs(a - b) / max(abs(a), abs(b), 1e-12)
        out.append((rel, path, a, b))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("boa_dir", type=Path)
    p.add_argument("boa_light_dir", type=Path)
    p.add_argument("--json", type=Path, help="Write the full report as JSON")
    args = p.parse_args(argv)
    report: dict = {"segmentations": {}, "measurements": {}}
    worst_agreement = 1.0

    print(f"{'segmentation':28s} {'agreement':>10s} {'min Dice':>9s}  worst labels")
    for name in SEGMENTATIONS:
        fa, fb = args.boa_dir / f"{name}.nii.gz", args.boa_light_dir / f"{name}.nii.gz"
        if not (fa.exists() and fb.exists()):
            continue
        a, b = np.asanyarray(nib.load(fa).dataobj), np.asanyarray(nib.load(fb).dataobj)
        if a.shape != b.shape:
            print(f"{name:28s} SHAPE MISMATCH {a.shape} vs {b.shape}")
            worst_agreement = 0
            continue
        agreement = float((a == b).mean())
        d = dice_per_label(a, b)
        worst = sorted(d.items(), key=lambda kv: kv[1])[:3]
        min_dice = min(d.values()) if d else 1.0
        worst_agreement = min(worst_agreement, agreement)
        print(f"{name:28s} {agreement:10.6f} {min_dice:9.4f}  {', '.join(f'{k}:{v:.3f}' for k, v in worst)}")
        report["segmentations"][name] = {"agreement": agreement, "dice": d}

    for fname in ["total-measurements.json", "bca-measurements.json"]:
        fa, fb = args.boa_dir / fname, args.boa_light_dir / fname
        if not (fa.exists() and fb.exists()):
            continue
        diffs = sorted(numeric_diffs(json.loads(fa.read_text()), json.loads(fb.read_text())), reverse=True)
        max_rel = diffs[0][0] if diffs else 0.0
        print(f"\n{fname}: {len(diffs)} shared numeric values, max relative difference {max_rel:.3g}")
        for rel, path, x, y in diffs[:5]:
            if rel > 0:
                print(f"   {rel:.3g}  {path}: {x} vs {y}")
        report["measurements"][fname] = {"n": len(diffs), "max_rel": max_rel}

    stats = args.boa_light_dir / "boa_light_stats.json"
    if stats.exists():
        s = json.loads(stats.read_text())
        print(f"\nBOA_light total time: {s.get('total_time')} s  (per stage: {s.get('timings')})")
    debug = args.boa_dir / "debug_information.txt"
    if debug.exists():
        lines = [ln for ln in debug.read_text().splitlines() if "Complete CT analysis" in ln]
        if lines:
            print("BOA:", lines[-1].split(": ", 1)[-1])
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    return 0 if worst_agreement == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
