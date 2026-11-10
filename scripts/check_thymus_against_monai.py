#!/usr/bin/env python
"""Check BOA_light's MONAI-free thymus pipeline against MONAI itself, piece by piece.

Needs torch and MONAI (only for this check; BOA_light does not use MONAI):

    python -m pip install monai
    python scripts/check_thymus_against_monai.py [CT.nii.gz] [--weights model_v2_2_7997.pt]

Compares, on a real CT if given (else a synthetic volume):
  1. the UNet: identical state-dict keys/shapes and identical outputs for the same weights;
  2. Spacingd(1.5, bilinear, align_corners, diagonal): output affine, shape and values;
  3. windowing + NormalizeIntensity + GaussianSmooth(0.4) + ScaleIntensity(-1, 1);
  4. sliding window inference (96^3, batch 4, overlap 0.375) with the same network;
  5. KeepLargestConnectedComponent and the nearest-neighbour inversion of Spacing.
Exit code 0 if everything matches (up to float32 rounding where noted).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from boa_light import thymus as T


def report(name, ok, detail=""):
    print(f"[{'OK ' if ok else 'FAIL'}] {name} {detail}")
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("ct", nargs="?", type=Path)
    p.add_argument("--weights", type=Path, help="bundle checkpoint (optional, else random weights)")
    args = p.parse_args(argv)

    from monai.data import MetaTensor
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import UNet
    from monai.transforms import (
        GaussianSmooth,
        KeepLargestConnectedComponent,
        NormalizeIntensity,
        ScaleIntensity,
        Spacing,
    )

    torch.manual_seed(0)
    ok = True

    # 1. network --------------------------------------------------------------
    ref_net = UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(32, 64, 128, 256),
                   strides=(2, 2, 2), num_res_units=4).eval()  # fmt: skip
    our_net = T.build_network().eval()
    if args.weights:
        sd = T._state_dict_from_checkpoint(torch.load(args.weights, map_location="cpu", weights_only=False))
        ref_net.load_state_dict(sd)
    sd_ref = ref_net.state_dict()
    keys_ok = list(sd_ref) == list(our_net.state_dict()) and all(
        sd_ref[k].shape == v.shape for k, v in our_net.state_dict().items()
    )
    ok &= report("UNet state-dict keys and shapes", keys_ok)
    our_net.load_state_dict(sd_ref, strict=True)
    x = torch.randn(2, 1, 96, 96, 96)
    with torch.no_grad():
        d = (ref_net(x) - our_net(x)).abs().max().item()
    ok &= report("UNet forward", d == 0.0, f"(max abs diff {d:.3g})")

    # image -----------------------------------------------------------------
    if args.ct:
        img = nib.load(args.ct)
    else:
        rng = np.random.default_rng(0)
        aff = np.diag([-0.8, -0.8, 2.5, 1.0])
        aff[:3, 3] = [100, 80, -50]
        img = nib.Nifti1Image(np.round(rng.normal(0, 300, (180, 160, 70))).astype(np.int16), aff)
    data = torch.from_numpy(np.asanyarray(img.dataobj).astype(np.float32))[None]
    affine = np.asarray(img.affine, dtype=np.float64)

    # 2. Spacing ----------------------------------------------------------------
    mt = MetaTensor(data.clone(), affine=torch.as_tensor(affine))
    spacing = Spacing(pixdim=T.PIXDIM, mode="bilinear", align_corners=True, diagonal=True)
    ref_sp = spacing(mt)
    new_aff, new_shape = T.spacing_target(affine, data.shape[1:])
    ours_sp = T.resample(data, affine, new_aff, new_shape, mode="bilinear")
    ok &= report("Spacing affine", np.allclose(np.asarray(ref_sp.affine), new_aff, atol=1e-6))
    ok &= report("Spacing shape", tuple(ref_sp.shape[1:]) == tuple(new_shape), f"{tuple(new_shape)}")
    d = (ref_sp.as_tensor() - ours_sp).abs().max().item() if ref_sp.shape == ours_sp.shape else float("inf")
    ok &= report("Spacing values", d <= 1e-4, f"(max abs diff {d:.3g})")

    # 3. intensity ------------------------------------------------------------------
    r = ref_sp.as_tensor().clip(*T.WINDOW_ABDOMEN)
    r = NormalizeIntensity()(r)
    r = GaussianSmooth(sigma=T.GAUSSIAN_SIGMA)(r)
    r = ScaleIntensity(minv=-1, maxv=1)(r)
    r = torch.as_tensor(r)
    o = T.preprocess_intensity(ours_sp)
    d = (r - o).abs().max().item()
    ok &= report("intensity preprocessing", d <= 1e-5, f"(max abs diff {d:.3g})")

    # 4. sliding window --------------------------------------------------------------
    inp = o[None]
    with torch.no_grad():
        ref_pred = sliding_window_inference(inp, T.ROI_SIZE, T.SW_BATCH_SIZE, ref_net, overlap=T.OVERLAP)
        our_pred = T.sliding_window(inp, our_net)
    d = (torch.as_tensor(ref_pred) - our_pred).abs().max().item()
    ok &= report("sliding window", d <= 1e-5, f"(max abs diff {d:.3g})")
    small = inp[..., :50, :60, :40]  # smaller than the ROI -> padding path
    with torch.no_grad():
        d = (torch.as_tensor(sliding_window_inference(small, T.ROI_SIZE, 4, ref_net, overlap=T.OVERLAP))
             - T.sliding_window(small, our_net)).abs().max().item()  # fmt: skip
    ok &= report("sliding window (padded small input)", d <= 1e-5, f"(max abs diff {d:.3g})")

    # 5. post-processing ----------------------------------------------------------------
    labels = torch.argmax(our_pred[0], dim=0, keepdim=True)
    if labels.sum() == 0:  # random weights may give nothing: use a thresholded image instead
        labels = (o > 0.2).to(torch.int64)
    ref_lcc = KeepLargestConnectedComponent()(labels.clone())
    our_lcc = T.keep_largest_component(labels.numpy())
    ok &= report("KeepLargestConnectedComponent", np.array_equal(np.asarray(ref_lcc), our_lcc))
    lab_mt = MetaTensor(torch.from_numpy(our_lcc).float(), affine=ref_sp.affine)
    lab_mt.applied_operations = ref_sp.applied_operations
    ref_inv = spacing.inverse(lab_mt)  # Invertd uses nearest interpolation for labels
    our_inv = T.resample(torch.from_numpy(our_lcc).float(), new_aff, affine, data.shape[1:], mode="nearest")
    # monai's own inverse uses the mode recorded at forward time (bilinear); Invertd switches to
    # nearest. Compare on the rounded label values.
    ref_inv_n = Spacing(pixdim=T.PIXDIM, mode="nearest", align_corners=True, diagonal=True)
    mt2 = MetaTensor(data.clone(), affine=torch.as_tensor(affine))
    sp2 = ref_inv_n(mt2)
    lab_mt2 = MetaTensor(torch.from_numpy(our_lcc).float(), affine=sp2.affine)
    lab_mt2.applied_operations = sp2.applied_operations
    ref_inv = ref_inv_n.inverse(lab_mt2)
    same = tuple(ref_inv.shape) == tuple(our_inv.shape) and torch.equal(
        torch.round(torch.as_tensor(ref_inv)), torch.round(our_inv)
    )
    ok &= report("inverse Spacing (nearest)", same)

    print("ALL OK" if ok else "MISMATCHES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
