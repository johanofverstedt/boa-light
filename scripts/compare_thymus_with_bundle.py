#!/usr/bin/env python
"""Compare BOA_light's thymus.nii.gz with the output of the original MONAI bundle.

1. Run the bundle as described in https://github.com/vprudente/open_thymus_segmentator
   (its own environment: Python 3.8, torch 1.12, MONAI 1.1):
       cd open_thymus_segmentator/app
       python -m monai.bundle run --meta_file configs/metadata.json \\
           --config_file configs/inference.yaml --dataset_dir CT.nii.gz --output_dir out_bundle
2. Run BOA_light:  boa-light run CT.nii.gz out_light -m thymus
3. python scripts/compare_thymus_with_bundle.py out_bundle/<file>.nii.gz out_light/thymus.nii.gz
"""

import sys

import nibabel as nib
import numpy as np


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    if len(argv) != 2:
        print(__doc__)
        return 2
    a_img, b_img = nib.load(argv[0]), nib.load(argv[1])
    a = np.rint(np.asanyarray(a_img.dataobj)).astype(np.uint8).squeeze()
    b = np.asanyarray(b_img.dataobj).astype(np.uint8).squeeze()
    if a.shape != b.shape:
        print(f"shape mismatch: bundle {a.shape} vs boa-light {b.shape}")
        return 1
    if not np.allclose(a_img.affine, b_img.affine, atol=1e-3):
        print("warning: affines differ\n", a_img.affine, "\n", b_img.affine)
    fa, fb = a > 0, b > 0
    inter = np.logical_and(fa, fb).sum()
    denom = fa.sum() + fb.sum()
    dice = 2 * inter / denom if denom else 1.0
    vox_ml = float(np.prod(b_img.header.get_zooms()[:3])) / 1000
    print(f"voxel agreement: {(a == b).mean():.6f}")
    print(f"Dice:            {dice:.6f}")
    print(f"volume bundle:   {fa.sum() * vox_ml:.2f} mL, boa-light: {fb.sum() * vox_ml:.2f} mL")
    print(f"differing voxels: {int((a != b).sum())}")
    return 0 if (a == b).all() else 1


if __name__ == "__main__":
    sys.exit(main())
