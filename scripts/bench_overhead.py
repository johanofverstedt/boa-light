#!/usr/bin/env python
"""Measure the non-network overhead of BOA vs BOA_light (no torch needed).

Both pipelines are run with the same instant "fake network" (tests/fake_model.py),
so the timing difference is purely the orchestration: loading, cropping,
resampling, temporary files, saving and measurements.

    BOA_SRC=/path/to/Body-and-Organ-Analysis python scripts/bench_overhead.py CT.nii.gz OUT_DIR [models]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from fake_model import fake_predict_fn  # noqa: E402
from ts_reference import load_reference  # noqa: E402

from boa_light.pipeline import Options, analyze  # noqa: E402

ct, out = Path(sys.argv[1]), Path(sys.argv[2])
models = sys.argv[3] if len(sys.argv) > 3 else "total+bca+lung_vessels+cerebral_bleed+hip_implant+liver_vessels+pleural_pericard_effusion"
ref = load_reference()

t = time.perf_counter()
analyze(ct, out / "boa_light", Options(models=models, contrast=False), predict_fn=fake_predict_fn)
t_bl = time.perf_counter() - t
print(f"BOA_light: {t_bl:.1f} s", flush=True)

t = time.perf_counter()
ref.boa.analyze_ct(
    input_folder=ct, processed_output_folder=out / "boa", excel_output_folder=out / "boa",
    models=set(models.split("+")), compute_contrast_information=False, total_preview=False,
    device="cpu", bca_pdf=False, recompute=True,
)  # fmt: skip
t_boa = time.perf_counter() - t
print(f"BOA:       {t_boa:.1f} s")
print(f"overhead speed-up: {t_boa / t_bl:.2f}x")
