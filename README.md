# BOA_light

A small, pip-installable and faster way to run
[BOA (Body and Organ Analysis)](https://github.com/UMEssen/Body-and-Organ-Analysis) including
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator) on a CT scan, without Docker.
It uses **only openly licensed models**, has **no PDF report**, and produces the same
segmentations as BOA. By default it writes **only the label images**. With `--measurements` it also
writes BOA's `*-measurements.json` files and `output.xlsx`.

See **[INSTALL.md](INSTALL.md)** for step-by-step installation in a conda environment. In short:

```bash
conda create -n boa_light python=3.11 && conda activate boa_light
conda install pip setuptools
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124   # matching your CUDA
python -m pip install --no-build-isolation "./[dicom]"
boa-light download                      # fetch weights once (~5 GB; reuses ~/.totalsegmentator)
boa-light run ct.nii.gz out/            # all open models, like BOA's "--models all" (labels only)
boa-light run ct.nii.gz out/ --measurements   # + BOA's measurements and Excel report
boa-light run dicom_dir/ out/ -m total+bca --fast-bca
```

From Python:

```python
from boa_light import analyze, Options
res = analyze("ct.nii.gz", "out/", Options(models="total+bca", measurements=True))
res.segmentations["total"]      # nibabel image, also written to out/total.nii.gz
res.measurements                # == out/total-measurements.json (None without measurements=True)
```

## Models

| model | output | networks |
|---|---|---|
| `total` | `total.nii.gz` (117 structures) | 291–295 at 1.5 mm (`--fast-total`: 297 at 3 mm) |
| `bca` | `body_parts`, `body_regions`, `tissues` (+ `bca-measurements.json` with `--measurements`) | 543, 542 (5 folds; `--fast-bca`: 1 fold); adds `total`. `--bca-skip-body-parts` drops 543 |
| `lung_vessels` | `lung_vessels.nii.gz` | 258 (cropped to the lungs) |
| `cerebral_bleed` | `cerebral_bleed.nii.gz` | 150 (cropped to the brain) |
| `hip_implant` | `hip_implant.nii.gz` | 260 (cropped to hips/femurs) |
| `liver_vessels` | `liver_vessels.nii.gz` | 8 (cropped to the liver) |
| `pleural_pericard_effusion` | `pleural_pericard_effusion.nii.gz` | 315, all folds (cropped to the lungs) |
| `liver_segments` | `liver_segments.nii.gz` | 570 (opt-in, not in `all`) |

`all` = `total+bca+lung_vessels+cerebral_bleed+hip_implant+liver_vessels+pleural_pericard_effusion`,
which is BOA's `all` without the licensed `heartchambers_highres`. `boa-light list` shows the models
and which weights are installed.

Outputs with `--measurements`: `ct_pfav.nii.gz`, `vertebrae.json`, `output.xlsx` (sheets `info`, `regions-statistics`,
`cnr-adjusted` (with `--cnr-adjustment`), `bca-aggregated-measurements`, `bca-slice-measurements`,
`bca-slice-measurements_no_ext`), and `boa_light_stats.json` with timings. Contrast phase prediction runs
if the optional `boa-contrast` package is installed (`pip install "boa-light[contrast]"`).

## What makes it faster

BOA calls TotalSegmentator once per model, and each call starts again from the file on disk.
BOA_light runs everything in one process and reuses all shared work:

* **The rough crop segmentation runs once.** Every cropped subtask in BOA (lung vessels, bleed, hip
  implant, liver vessels, effusion) runs its own 6 mm "total" network to find its bounding box.
  BOA_light runs it once and shares it, and `--crop-source total` skips it entirely by using the full
  `total` result.
* **Each image is prepared once.** The canonical reorientation and the 1.5 mm, 6 mm and 5 mm-slice
  resamplings are computed once. Tasks with the same crop (lung vessels and effusion; liver vessels
  and liver segments) share the cropped image.
* **No temporary files or process pools.** Images go to nnU-Net as arrays, not gzipped temporary
  NIfTIs, and no new process pools are started for each prediction.
* **Networks stay loaded.** Each network is built and its checkpoint read only once.
* **The argmax runs on the GPU.** nnU-Net moves the half-precision logits of the whole volume
  (several GB for `total`) to the CPU for a numpy argmax. BOA_light takes the argmax on the GPU,
  which gives an identical result.
* **CPU and GPU work overlap.** Preparation, resampling back, saving, BCA post-processing and
  measurements run in worker threads while the GPU segments the next task.
* **Faster measurements.** Per-structure statistics are computed inside each label's bounding box,
  with identical values.
* **Smaller install.** No PDF report (weasyprint/Chromium/plotly/vtk), no Docker, no Celery or PACS
  stack.

## Faithfulness to BOA

The numerical pipeline is a step-by-step port of TotalSegmentator v2.12.0 / BOA 1.0.3: same networks,
folds, step sizes, crop margins (20 mm), spline resampling, int32 casts, triple split for large images,
label merging, canonical/undo steps and BCA post-processing.

The test suite runs **BOA's real code** (`analyze_ct`, with temporary files, the TotalSegmentator
API, the BCA pipeline and the Excel writer) and BOA_light on the same CTs, with the nnU-Net forward
pass replaced by a deterministic stand-in network. It covers RAS, LPS and permuted-axis inputs. All
segmentations are voxel-identical, the measurement JSONs agree to about 1e-16, and the Excel sheets
are equal. On a 488×404×224 CT with the stand-in network, BOA's non-network overhead is 132 s and
BOA_light's is 54 s on 2 CPU cores. The real gains on a GPU are expected to be larger, because the
redundant network runs and the CPU argmax are gone too.

Intentional differences from BOA (bug fixes):

* `total-measurements.json` only contains each model's own structures. BOA also matches other class
  maps that share the name prefix, so entries such as `v1_spleen` and `mr_spleen` appear under `total`.
* `lung_vessels` is measured. BOA looks for `lung_vessels_airways.nii.gz`, never finds it, and skips it.
* `bca-measurements.json` additionally contains the `other_findings` sentences that used to be in the PDF.

## Validate on your machine

```bash
# 1) the in-memory / GPU-argmax nnU-Net path equals nnU-Net's own file-based prediction
python scripts/check_predictor.py ct.nii.gz --task 298
python scripts/check_predictor.py ct.nii.gz --task 291
# 2) compare with a BOA run on the same CT (docker image shipai/boa-cli, --models all)
boa-light -v run ct.nii.gz out_light/
python scripts/compare_outputs.py out_boa/ out_light/
```

Developer tests (no torch needed): `pip install -e ".[test]" && pytest`. To include the BOA equivalence
tests, also set `BOA_SRC=/path/to/Body-and-Organ-Analysis` and optionally
`TS_EXAMPLE_CT=/path/to/TotalSegmentator/tests/reference_files/example_ct.nii.gz`.

## Options

`boa-light run --help`. Key flags: `--device auto|cuda|cuda:1|cpu`, `--fast-total`, `--fast-bca`,
`--crop-source rough|total`, `--export-folds` (also save each fold of the ensemble models to `folds/<model>_fold<k>.nii.gz`), `--bca-skip-body-parts` (skip the `body_parts` model; drops the
"no extremities" BCA measurements and the L3 body axes), `--cnr-adjustment`, `--save-format nii.gz|nii|none`, `--threads N`, `-q/--quiet` (no progress messages),
`--low-memory` (unload networks after use), `--no-fast-export` (nnU-Net's CPU argmax). Weights live in
`$BOA_LIGHT_WEIGHTS`, `$TOTALSEG_WEIGHTS_PATH` or `~/.totalsegmentator/nnunet/results`.

## Citation

Please cite BOA (Haubold et al., Invest. Radiol. 2023, doi:10.1097/RLI.0000000000001040),
TotalSegmentator (Wasserthal et al., Radiol. AI 2023, doi:10.1148/ryai.230024) and nnU-Net
(Isensee et al., Nat. Methods 2021, doi:10.1038/s41592-020-01008-z).
License: Apache-2.0 (see `LICENSE` and `NOTICE`).
