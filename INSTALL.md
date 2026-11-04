# Installing BOA_light

These steps create a dedicated conda environment and install BOA_light into it.

## 1. Create and activate a conda environment

```bash
conda create -n boa_light python=3.11
conda activate boa_light
```

## 2. Give the environment its own pip and setuptools

```bash
conda install pip setuptools
```

Check that both `python` and `pip` now come from the environment, not the system:

```bash
which python pip      # both must point into .../envs/boa_light/bin/
```

If they point to `/usr/bin/...`, conda is not active in this shell: run `conda init`, open a new
terminal and `conda activate boa_light` again. (Otherwise pip fails with
`error: externally-managed-environment`.)

## 3. Install PyTorch

Install torch before BOA_light, so you get the build that matches your hardware. Pick the
command for your CUDA version at <https://pytorch.org/get-started/locally/>, for example:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124   # NVIDIA GPU, CUDA 12.4
# or
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU only
```

Check that the GPU is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 4. Install BOA_light

From the unpacked `boa_light` folder:

```bash
python -m pip install --no-build-isolation ./
# optional extras:
python -m pip install --no-build-isolation "./[dicom]"           # DICOM folder input (SimpleITK, pydicom)
python -m pip install --no-build-isolation "./[dicom,contrast]"  # + contrast phase prediction (boa-contrast)
```

To update to a newer version later, unpack it and reinstall with `--force-reinstall`
(otherwise pip may keep the installed copy):

```bash
python -m pip install --no-build-isolation --force-reinstall --no-deps ./
boa-light --version
```

If you change the code yourself, install it in editable mode instead (`pip install -e`); edits then take
effect without reinstalling.

Always use `python -m pip` (not bare `pip`). This guarantees the pip of the active environment
is used.

## 5. Download the model weights (once)

```bash
boa-light download          # ~5 GB; reuses ~/.totalsegmentator/nnunet/results if present
boa-light list              # shows models and which weights are installed
```

To keep the weights somewhere else, set `export BOA_LIGHT_WEIGHTS=/path/to/weights` (or
TotalSegmentator's `TOTALSEG_WEIGHTS_PATH`) before downloading and running.

## 6. Run

```bash
boa-light run ct.nii.gz out/                     # all open models, label images only
boa-light run ct.nii.gz out/ --measurements      # + measurements, JSON and Excel reports
boa-light -v run dicom_folder/ out/ -m total+bca
```

## Optional: check the installation

```bash
python scripts/check_predictor.py ct.nii.gz --task 298   # in-memory GPU path == nnU-Net's own
python -m pip install --no-build-isolation "./[test]" && python -m pytest tests
```

## Uninstall

```bash
conda deactivate
conda env remove -n boa_light
```
