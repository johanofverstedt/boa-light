"""Anterior mediastinal fat (thymus region) segmentation with the Open Thymus Segmentator.

Model: https://github.com/vprudente/open_thymus_segmentator (Apache-2.0,
(c) 2023 Vasco Prudente & Keno Bressem), a MONAI bundle. To keep BOA_light light,
MONAI is *not* a dependency: the parts of MONAI the bundle uses are re-implemented
here with torch/numpy/scipy, following MONAI's code (Apache-2.0, (c) MONAI Consortium)
operation by operation:

* ``monai.networks.nets.UNet`` (channels 32-64-128-256, strides 2, 4 residual
  units, PReLU, instance norm), with identical module names so the bundle's
  checkpoint loads with ``strict=True``;
* ``Spacingd(pixdim=1.5, mode="bilinear", align_corners=True, diagonal=True)``
  (affine/shape computation of ``zoom_affine``/``compute_shape_offset`` and the
  ``affine_grid``/``grid_sample`` resampling in float64 with border padding);
* the bundle's ``ApplyWindowingd("abdomen")`` (clip to [-150, 250] HU),
  ``NormalizeIntensityd``, ``GaussianSmoothd(sigma=0.4)`` (erf kernel, zero-padded
  separable convolution), ``ScaleIntensityd(-1, 1)``;
* ``SlidingWindowInferer(roi 96^3, sw_batch_size 4, overlap 0.375)`` with
  constant blending, under AMP on CUDA like the bundle's evaluator;
* ``AsDiscreted(argmax)``, ``KeepLargestConnectedComponentd`` (26-connectivity)
  and ``Invertd`` of the spacing step (nearest neighbour) back to the input grid.

Weights (Zenodo record 18355851) are downloaded on first use, or given with
``--thymus-weights``. ``scripts/compare_thymus_with_bundle.py`` compares the result
with an output of the original bundle.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import tempfile
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np

from boa_light.weights import weights_dir

logger = logging.getLogger(__name__)

LABELS = {1: "anterior_mediastinal_fat"}
ZENODO_RECORD = "18355851"
CHECKPOINT_NAME = "model_v2_2_7997.pt"  # ckpt_path in the bundle's inference.yaml
WINDOW_ABDOMEN = (-150.0, 250.0)  # ApplyWindowing("abdomen"): level 50, width 400
PIXDIM = (1.5, 1.5, 1.5)
GAUSSIAN_SIGMA = 0.4
ROI_SIZE = (96, 96, 96)
SW_BATCH_SIZE = 4
OVERLAP = 0.375
AFFINE_TOL = 1e-3  # monai.data.utils.AFFINE_TOL


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #


def thymus_weights_dir(root: Path | None = None) -> Path:
    return Path(root or weights_dir()) / "open_thymus_segmentator"


def find_weights(root: Path | None = None) -> Path | None:
    d = thymus_weights_dir(root)
    preferred = d / CHECKPOINT_NAME
    if preferred.is_file():
        return preferred
    candidates = sorted(d.glob("*.pt")) + sorted(d.glob("*.pth"))
    return candidates[0] if len(candidates) == 1 else None


def _zenodo_files() -> list[dict]:
    url = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
        record = json.load(r)
    files = record.get("files") or []
    if isinstance(files, dict):  # newer API: {"entries": {...}}
        files = list((files.get("entries") or {}).values())
    return files


def download_weights(root: Path | None = None, force: bool = False) -> Path:
    """Download the checkpoint from Zenodo (record 18355851) if it is not present."""
    existing = find_weights(root)
    if existing is not None and not force:
        return existing
    d = thymus_weights_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    files = _zenodo_files()
    ckpts = [f for f in files if str(f.get("key", "")).endswith((".pt", ".pth"))]
    chosen = next((f for f in ckpts if f.get("key") == CHECKPOINT_NAME), None)
    if chosen is None:
        if len(ckpts) != 1:
            names = ", ".join(f.get("key", "?") for f in ckpts) or "none"
            raise RuntimeError(
                f"Could not pick the thymus checkpoint from Zenodo record {ZENODO_RECORD} (files: {names}). "
                f"Download it manually and pass --thymus-weights PATH."
            )
        chosen = ckpts[0]
    key = chosen["key"]
    links = chosen.get("links") or {}
    urls = [u for u in (links.get("content"), links.get("self"), links.get("download")) if u]
    urls.append(f"https://zenodo.org/records/{ZENODO_RECORD}/files/{key}?download=1")
    checksum = str(chosen.get("checksum", ""))
    expected_md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None

    logger.info("Downloading thymus weights %s from Zenodo", key)
    last_error: Exception | None = None
    for url in urls:
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(dir=d, delete=False, suffix=".part") as tmp:
                tmp_name = tmp.name
                md5 = hashlib.md5()  # noqa: S324 (integrity check only)
                with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310
                    while chunk := r.read(1 << 20):
                        md5.update(chunk)
                        tmp.write(chunk)
            if expected_md5 and md5.hexdigest() != expected_md5:
                raise RuntimeError(f"checksum mismatch for {url}")
            if Path(tmp_name).stat().st_size < 1000:
                raise RuntimeError(f"{url} did not return a checkpoint")
            target = d / key
            shutil.move(tmp_name, target)
            return target
        except Exception as e:  # try the next URL
            last_error = e
            if tmp_name:
                Path(tmp_name).unlink(missing_ok=True)
    raise RuntimeError(f"Downloading the thymus weights failed: {last_error}")


# --------------------------------------------------------------------------- #
# Network: minimal re-implementation of monai.networks.nets.UNet
# --------------------------------------------------------------------------- #


def build_network(in_channels=1, out_channels=2, channels=(32, 64, 128, 256), strides=(2, 2, 2), num_res_units=4):
    """``monai.networks.nets.UNet(spatial_dims=3, ...)`` with MONAI's defaults
    (kernel 3, up-kernel 3, PReLU, instance norm without affine, dropout 0, bias,
    "NDA" ordering). Module names match MONAI's, so state dicts are interchangeable."""
    import torch
    from torch import nn

    class ADN(nn.Sequential):  # ordering "NDA": InstanceNorm3d -> Dropout(0) -> PReLU
        def __init__(self, ch):
            super().__init__()
            self.add_module("N", nn.InstanceNorm3d(ch))
            self.add_module("D", nn.Dropout(p=0.0))
            self.add_module("A", nn.PReLU())

    class Convolution(nn.Sequential):
        def __init__(self, inc, outc, stride=1, conv_only=False, transposed=False):
            super().__init__()
            if transposed:  # output_padding = stride - 1 (stride_minus_kernel_padding(1, stride))
                conv = nn.ConvTranspose3d(inc, outc, 3, stride=stride, padding=1, output_padding=stride - 1)
            else:
                conv = nn.Conv3d(inc, outc, 3, stride=stride, padding=1)
            self.add_module("conv", conv)
            if not conv_only:
                self.add_module("adn", ADN(outc))

    class ResidualUnit(nn.Module):
        def __init__(self, inc, outc, stride=1, subunits=2, last_conv_only=False):
            super().__init__()
            self.conv = nn.Sequential()
            self.residual = nn.Identity()
            sch, sst = inc, stride
            subunits = max(1, subunits)
            for su in range(subunits):
                conv_only = last_conv_only and su == subunits - 1
                self.conv.add_module(f"unit{su:d}", Convolution(sch, outc, sst, conv_only=conv_only))
                sch, sst = outc, 1
            if stride != 1 or inc != outc:
                if stride == 1:
                    self.residual = nn.Conv3d(inc, outc, 1, stride, 0)
                else:
                    self.residual = nn.Conv3d(inc, outc, 3, stride, 1)

        def forward(self, x):
            res = self.residual(x)
            cx = self.conv(x)
            return cx + res

    class SkipConnection(nn.Module):
        def __init__(self, submodule):
            super().__init__()
            self.submodule = submodule

        def forward(self, x):
            return torch.cat([x, self.submodule(x)], dim=1)

    def down(inc, outc, stride):
        if num_res_units > 0:
            return ResidualUnit(inc, outc, stride, subunits=num_res_units)
        return Convolution(inc, outc, stride)

    def up(inc, outc, stride, is_top):
        conv = Convolution(inc, outc, stride, conv_only=is_top and num_res_units == 0, transposed=True)
        if num_res_units > 0:
            ru = ResidualUnit(outc, outc, 1, subunits=1, last_conv_only=is_top)
            return nn.Sequential(conv, ru)
        return conv

    def create_block(inc, outc, chs, sts, is_top):
        c, s = chs[0], sts[0]
        if len(chs) > 2:
            sub = create_block(c, c, chs[1:], sts[1:], False)
            upc = c * 2
        else:
            sub = down(c, chs[1], 1)  # bottom layer
            upc = c + chs[1]
        return nn.Sequential(down(inc, c, s), SkipConnection(sub), up(upc, outc, s, is_top))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = create_block(in_channels, out_channels, list(channels), list(strides), True)

        def forward(self, x):
            return self.model(x)

    return UNet()


def _state_dict_from_checkpoint(ckpt) -> dict:
    """The bundle's CheckpointLoader loads ``{"network": ...}`` or a bare state dict."""
    import torch

    if isinstance(ckpt, torch.nn.Module):
        return ckpt.state_dict()
    if isinstance(ckpt, dict):
        for key in ("network", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    sd = dict(ckpt)
    if sd and all(k.startswith("module.") for k in sd):
        sd = {k[len("module.") :]: v for k, v in sd.items()}
    return sd


# --------------------------------------------------------------------------- #
# Geometry: MONAI Spacing (diagonal, align_corners) and its inverse
# --------------------------------------------------------------------------- #


def _affine_to_spacing(affine: np.ndarray) -> np.ndarray:
    a = np.asarray(affine, dtype=np.float64)[:3, :3]
    sp = np.sqrt(np.sum(a * a, axis=0))
    sp[sp == 0] = 1.0
    return sp


def spacing_target(affine: np.ndarray, shape, pixdim=PIXDIM) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Output affine and shape of ``Spacing(pixdim, diagonal=True)`` (``zoom_affine`` +
    ``compute_shape_offset`` with ``scale_extent=False``)."""
    in_aff = np.asarray(affine, dtype=np.float64)
    scale = np.asarray(pixdim, dtype=np.float64)
    scale[scale == 0] = 1.0
    out_aff = np.diag(np.append(scale, 1.0))  # diagonal=True
    shp = np.asarray(shape, dtype=np.float64)
    corners = np.asarray(np.meshgrid(*[(0.0, d - 1.0) for d in shp], indexing="ij")).reshape((3, -1))
    corners = np.concatenate((corners, np.ones_like(corners[:1])))
    corners_out = np.linalg.solve(out_aff, in_aff) @ corners
    corners_world = in_aff @ corners
    all_dist = corners_out[:-1].copy()
    corners_out = corners_out[:-1] / corners_out[-1]
    out_shape = np.round(np.ptp(corners_out, axis=1) + 1.0)
    offset = None
    for i in range(corners_world.shape[1]):
        min_corner = np.min(all_dist - all_dist[:, i : i + 1], 1)
        if np.allclose(min_corner, 0.0, rtol=AFFINE_TOL):
            offset = corners_world[:-1, i]
            break
    if offset is None:
        offset = in_aff[:-1, :-1] @ (shp / 2.0) + in_aff[:-1, -1] - out_aff[:-1, :-1] @ (out_shape / 2.0)
    out_aff[:3, -1] = offset[:3]
    return out_aff, tuple(int(v) for v in out_shape.astype(int))


def _normalize_transform(shape):
    """``monai.networks.utils.normalize_transform`` for align_corners=True."""
    import torch

    norm = torch.tensor([float(s) for s in shape], dtype=torch.float64)
    norm[norm <= 1.0] = 2.0
    norm = 2.0 / (norm - 1.0)
    norm = torch.diag(torch.cat((norm, torch.ones((1,), dtype=torch.float64))))
    norm[:-1, -1] = -1.0
    return norm.unsqueeze(0)


def resample(img, src_affine: np.ndarray, dst_affine: np.ndarray, spatial_size, mode: str):
    """``SpatialResample`` as used by ``Spacing`` (padding "border", align_corners,
    float64 computation, float32 output). ``img`` is a (C, X, Y, Z) tensor."""
    import torch
    import torch.nn.functional as F

    in_size = tuple(int(s) for s in img.shape[1:])
    spatial_size = tuple(int(s) for s in spatial_size)
    src = np.asarray(src_affine, dtype=np.float64)
    dst = np.asarray(dst_affine, dtype=np.float64)
    xform = np.linalg.solve(src, dst)
    same_size = np.allclose(np.asarray(spatial_size), np.asarray(in_size))
    if same_size and (np.allclose(src, dst, atol=AFFINE_TOL) or np.allclose(xform, np.eye(4), atol=AFFINE_TOL)):
        return img.to(torch.float32).clone()
    img64 = img.to(torch.float64)
    theta = torch.as_tensor(xform, dtype=torch.float64)[None]
    src_x = _normalize_transform(in_size)
    dst_x = _normalize_transform(spatial_size)
    theta = src_x @ theta @ torch.as_tensor(np.linalg.inv(dst_x.numpy()), dtype=torch.float64)
    rev = torch.as_tensor([2, 1, 0])
    theta[:, :3] = theta[:, rev]
    theta[:, :, :3] = theta[:, :, rev]
    grid = F.affine_grid(theta=theta[:, :3], size=[1, img.shape[0], *spatial_size], align_corners=True)
    out = F.grid_sample(img64[None].contiguous(), grid, mode=mode, padding_mode="border", align_corners=True)
    return out[0].to(torch.float32)


# --------------------------------------------------------------------------- #
# Intensity transforms
# --------------------------------------------------------------------------- #


def _gaussian_1d(sigma: float, truncated: float = 4.0):
    import torch

    sigma_t = torch.as_tensor(sigma, dtype=torch.float)
    tail = int(max(float(sigma_t) * truncated, 0.5) + 0.5)
    x = torch.arange(-tail, tail + 1, dtype=torch.float)
    t = 0.70710678 / torch.abs(sigma_t)
    out = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
    return out.clamp(min=0)


def _separable_filter(x, kernels, d: int):
    """``monai.networks.layers.simplelayers._separable_filtering_conv`` (zero padding)."""
    import torch.nn.functional as F

    if d < 0:
        return x
    s = [1] * len(x.shape)
    s[d + 2] = -1
    k = kernels[d].reshape(s)
    if k.numel() == 1 and k[0] == 1:
        return _separable_filter(x, kernels, d - 1)
    n_ch = x.shape[1]
    k = k.repeat([n_ch, 1] + [1] * 3)
    pad = [0, 0, 0]
    pad[d] = (kernels[d].shape[0] - 1) // 2
    pad_arg = sum([[p, p] for p in reversed(pad)], [])
    padded = F.pad(x, pad_arg, mode="constant")
    return F.conv3d(input=_separable_filter(padded, kernels, d - 1), weight=k, groups=n_ch)


def preprocess_intensity(x):
    """Windowing, NormalizeIntensity, GaussianSmooth, ScaleIntensity on a (1, X, Y, Z) tensor."""
    import torch

    x = x.to(torch.float32).clip(WINDOW_ABDOMEN[0], WINDOW_ABDOMEN[1])
    mean = torch.mean(x.float()).item()
    std = torch.std(x.float(), unbiased=False).item()
    if std == 0.0:
        std = 1.0
    x = (x - mean) / std
    kernels = [_gaussian_1d(GAUSSIAN_SIGMA).to(x) for _ in range(3)]
    x = _separable_filter(x.float().unsqueeze(0), kernels, 2).squeeze(0)
    mina, maxa = x.min(), x.max()
    if mina == maxa:
        return x * -1.0
    return ((x - mina) / (maxa - mina)) * 2.0 + -1.0


# --------------------------------------------------------------------------- #
# Sliding window inference (constant blending), as monai.inferers.sliding_window_inference
# --------------------------------------------------------------------------- #


def _scan_interval(image_size, roi_size, overlap):
    out = []
    for i in range(3):
        if roi_size[i] == image_size[i]:
            out.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            out.append(interval if interval > 0 else 1)
    return out


def _patch_starts(image_size, patch_size, interval):
    starts = []
    for d in range(3):
        num = int(math.ceil(float(image_size[d]) / interval[d]))
        scan = next((i for i in range(num) if i * interval[d] + patch_size[d] >= image_size[d]), None)
        n = scan + 1 if scan is not None else 1
        ds = []
        for i in range(n):
            s = i * interval[d]
            s -= max(s + patch_size[d] - image_size[d], 0)
            ds.append(s)
        starts.append(ds)
    return np.asarray([x.flatten() for x in np.meshgrid(*starts, indexing="ij")]).T


def sliding_window(inputs, network, roi_size=ROI_SIZE, sw_batch_size=SW_BATCH_SIZE, overlap=OVERLAP):
    """``inputs``: (1, C, X, Y, Z) on the compute device. Returns (1, K, X, Y, Z)."""
    import torch
    import torch.nn.functional as F

    compute_dtype = inputs.dtype
    image_size_ = list(inputs.shape[2:])
    roi = [int(r) for r in roi_size]
    image_size = [max(image_size_[i], roi[i]) for i in range(3)]
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    if any(pad_size):
        inputs = F.pad(inputs, pad=pad_size, mode="constant", value=0.0)
    patch = [min(image_size[i], roi[i]) for i in range(3)]
    starts = _patch_starts(image_size, patch, _scan_interval(image_size, roi, overlap))
    slices = [tuple(slice(int(s), int(s) + patch[d]) for d, s in enumerate(x)) for x in starts]
    w = torch.ones([1, 1, *patch], dtype=compute_dtype, device=inputs.device)
    output = count = None
    for g in range(0, len(slices), sw_batch_size):
        group = slices[g : g + sw_batch_size]
        win = torch.cat([inputs[(slice(0, 1), slice(None), *s)] for s in group])
        seg = network(win)
        if output is None:
            output = torch.zeros([1, seg.shape[1], *image_size], dtype=compute_dtype, device=inputs.device)
            count = torch.zeros([1, 1, *image_size], dtype=compute_dtype, device=inputs.device)
            for s in slices:
                count[(slice(None), slice(None), *s)] += w
        seg = seg * w if seg.dtype == w.dtype else seg.mul_(w)
        for p, s in zip(seg, group):
            output[(slice(0, 1), slice(None), *s)] += p
    output /= count
    if any(pad_size):
        final = []
        for sp in range(3):
            si = 3 - sp - 1
            final.insert(0, slice(pad_size[sp * 2], pad_size[sp * 2] + image_size_[si]))
        output = output[(slice(None), slice(None), *final)]
    return output


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #


def keep_largest_component(labels: np.ndarray) -> np.ndarray:
    """``KeepLargestConnectedComponent`` for a single foreground label (26-connectivity)."""
    from scipy import ndimage

    fg = labels > 0
    lab, n = ndimage.label(fg, structure=np.ones((3, 3, 3), dtype=bool))
    if n <= 1:
        return labels
    sizes = np.bincount(lab.ravel())[1:]
    keep = int(np.argmax(sizes)) + 1
    out = labels.copy()
    out[fg & (lab != keep)] = 0
    return out


# --------------------------------------------------------------------------- #
# Runner (prepare on CPU -> infer on GPU -> finalize on CPU)
# --------------------------------------------------------------------------- #


class ThymusSegmenter:
    def __init__(self, device: str = "auto", weights: Path | None = None, weights_root: Path | None = None,
                 download: bool = True):  # fmt: skip
        import torch

        from boa_light.predictor import resolve_device

        self.device = resolve_device(device)
        if weights is None:
            weights = find_weights(weights_root)
            if weights is None:
                if not download:
                    raise FileNotFoundError(
                        f"Thymus weights not found in {thymus_weights_dir(weights_root)}; "
                        "run 'boa-light download -m thymus' or pass --thymus-weights PATH"
                    )
                weights = download_weights(weights_root)
        self.weights = Path(weights)
        self.network = build_network()
        ckpt = torch.load(self.weights, map_location="cpu", weights_only=False)
        self.network.load_state_dict(_state_dict_from_checkpoint(ckpt), strict=True)
        self.network.to(self.device).eval()

    def prepare(self, img: nib.Nifti1Image) -> dict:
        import torch

        data = torch.from_numpy(np.asanyarray(img.dataobj).astype(np.float32))[None]  # EnsureChannelFirst
        affine = np.asarray(img.affine, dtype=np.float64)
        new_affine, new_shape = spacing_target(affine, data.shape[1:])
        x = resample(data, affine, new_affine, new_shape, mode="bilinear")
        x = preprocess_intensity(x)
        return {"image": x, "affine": new_affine, "orig_affine": affine, "orig_shape": tuple(data.shape[1:])}

    def infer(self, prepared: dict):
        import torch

        x = prepared["image"][None].to(self.device)
        with torch.no_grad(), torch.autocast("cuda", enabled=self.device.type == "cuda"):
            pred = sliding_window(x, self.network)
        return pred[0].float().cpu()

    def finalize(self, prepared: dict, pred) -> np.ndarray:
        import torch

        labels = torch.argmax(pred, dim=0, keepdim=True)  # AsDiscrete(argmax=True)
        labels = torch.from_numpy(keep_largest_component(labels.numpy()))
        inv = resample(labels.to(torch.float32), prepared["affine"], prepared["orig_affine"],
                       prepared["orig_shape"], mode="nearest")  # fmt: skip
        return np.rint(inv[0].numpy()).astype(np.uint8)
