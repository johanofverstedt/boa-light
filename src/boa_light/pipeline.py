"""The BOA_light analysis pipeline."""

from __future__ import annotations

import json
import logging
import platform
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import nibabel as nib
import numpy as np
import pandas as pd

from boa_light import __version__
from boa_light.bca import body_region_flags, compute_bca_measurements, postprocess_nifti, subclassify_tissues
from boa_light.imaging import lps_shape_spacing
from boa_light.engine import ImageContext, PredictFn, Prepared, SegmentationEngine, roi_mask
from boa_light.measurements import compute_measurements, segmentator_metrics
from boa_light.models import CROP_TASKS, ROUGH, TASKS, TOTAL_FAST, networks_for, resolve_models

logger = logging.getLogger(__name__)


@dataclass
class Options:
    models: str | list[str] = "all"
    measurements: bool = False  # False: only label images. True: also JSON/Excel reports (BOA outputs)
    device: str = "auto"
    fast_total: bool = False
    fast_bca: bool = False
    crop_source: str = "rough"  # "rough" (identical to BOA) | "total" (no extra model)
    cnr_adjustment: bool = False
    contrast: bool = True  # run boa-contrast if installed
    bca_median_filtering: bool = False
    bca_examined_body_region: str | None = None
    bca_skip_body_parts: bool = False  # no "no extremities" measurements / L3 axes
    save_format: str = "nii.gz"  # "nii.gz" | "nii" | "none"
    excel: bool = True
    threads: int = 4
    keep_models_loaded: bool = True
    fast_export: bool = True  # argmax on the GPU (identical result, much faster)
    progress: bool = True  # print progress messages to stderr
    download: bool = True
    weights_root: Path | None = None


@dataclass
class Result:
    segmentations: dict[str, nib.Nifti1Image] = field(default_factory=dict)
    measurements: dict[str, Any] | None = None
    bca_measurements: dict[str, Any] | None = None
    info: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    excel_path: Path | None = None


class _Timer:
    def __init__(self, stats: dict, key: str):
        self.stats, self.key = stats, key

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t
        self.stats.setdefault("timings", {})[self.key] = round(dt, 3)
        logger.info("%s: DONE in %.2fs", self.key, dt)


class _Progress:
    """Short human readable status lines on stderr (independent of logging)."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.t0 = time.perf_counter()

    def __call__(self, msg: str, end: str = "\n") -> None:
        if self.enabled:
            elapsed = time.perf_counter() - self.t0
            sys.stderr.write(f"[{elapsed:7.1f}s] {msg}{end}")
            sys.stderr.flush()


_STEP_NAMES = {"_rough": "rough 6 mm total (crop masks)"}


def load_input(path: Path) -> tuple[nib.Nifti1Image, list[dict[str, Any]]]:
    path = Path(path)
    if path.is_file() and ".nii" in path.name.lower():
        return nib.load(path), []
    if path.is_dir():
        from boa_light.dicom import read_dicom_series

        return read_dicom_series(path)
    raise FileNotFoundError(f"Input must be a NIfTI file or a DICOM directory: {path}")


def _save(img: nib.Nifti1Image, path: Path) -> Path:
    nib.save(img, path)
    return path


def analyze(
    input_image: Path | nib.Nifti1Image,
    output_dir: Path | None,
    options: Options | None = None,
    predict_fn: PredictFn | None = None,
) -> Result:
    """Run all requested models on one CT and compute the BOA measurements.

    ``predict_fn`` can replace the nnU-Net model pool (used for testing).
    """
    opt = options or Options()
    progress = _Progress(opt.progress)
    res = Result()
    stats = res.stats
    t_start = time.perf_counter()
    tasks, run_bca = resolve_models(opt.models, opt.bca_skip_body_parts)
    stats.update(
        {
            "boa_light_version": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "models": tasks,
            "bca": run_bca,
            "fast_total": opt.fast_total,
            "fast_bca": opt.fast_bca,
            "crop_source": opt.crop_source,
        }
    )
    save = output_dir is not None and opt.save_format != "none"
    ext = ".nii" if opt.save_format == "nii" else ".nii.gz"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- input
    progress(f"Loading {input_image if not isinstance(input_image, nib.Nifti1Image) else 'image'}")
    with _Timer(stats, "load_input"):
        is_dicom = False
        if isinstance(input_image, nib.Nifti1Image):
            img, dicom_info = input_image, []
            ct_path = None
        else:
            is_dicom = Path(input_image).is_dir()
            img, dicom_info = load_input(Path(input_image))
            ct_path = None if is_dicom else Path(input_image)
        ctx = ImageContext(img)
    shape = ctx.orig.shape
    zooms = ctx.orig.header.get_zooms()[:3]
    logger.info("Image size: %s, spacing: %s, axcodes: %s", shape, zooms, nib.aff2axcodes(ctx.orig.affine))
    ct_data = np.asanyarray(ctx.orig.dataobj)
    if np.any(ct_data < -1024) or np.any(ct_data > 3071):
        logger.warning(
            "Unexpected CT values found in input image: got %s-%s, expected -1024-3071.",
            ct_data.min(),
            ct_data.max(),
        )
    progress(f"  image {shape}, spacing {tuple(round(float(z), 3) for z in zooms)} mm")
    stats["num_voxels"] = int(np.prod(shape))
    stats["num_slices"] = int(shape[2])
    if save and is_dicom and opt.measurements:
        _save(ctx.orig, output_dir / f"image{ext}")
        ct_path = output_dir / f"image{ext}"

    # ---------------------------------------------------------------- models
    pool = None
    if predict_fn is None:
        from boa_light import weights
        from boa_light.predictor import ModelPool

        ids = networks_for(tasks, opt.fast_total, opt.crop_source)
        if opt.download:
            from boa_light.weights import is_available

            missing = [i for i in ids if not is_available(i, opt.weights_root)]
            if missing:
                progress(f"Downloading weights for {len(missing)} network(s): {missing}")
            with _Timer(stats, "download_weights"):
                weights.ensure(ids, opt.weights_root)
        pool = ModelPool(opt.device, opt.weights_root, keep_loaded=opt.keep_models_loaded)
        pool.fast_export = opt.fast_export
        predict_fn = pool.predict
        stats["device"] = str(pool.device)
    current = {"name": "", "i": 0, "n": 0}
    base_predict = predict_fn

    def predict_with_progress(task_id, img, folds, step_size):
        t = time.perf_counter()
        out = base_predict(task_id, img, folds, step_size)
        current["i"] += 1
        if current["n"] > 1:  # multi-network task or split image
            progress(f"      network {task_id} ({current['i']}/{current['n']}) {time.perf_counter() - t:.1f}s")
        return out

    engine = SegmentationEngine(ctx, predict_with_progress)

    def task_obj(name):
        return TOTAL_FAST if (name == "total" and opt.fast_total) else TASKS[name]

    crop_tasks = [n for n in tasks if n in CROP_TASKS]
    use_total_crop = opt.crop_source == "total" and "total" in tasks
    run_rough = bool(crop_tasks) and not use_total_crop
    lps_shape, lps_zooms = lps_shape_spacing(ctx.orig)
    bca_force_split = round(lps_shape[2] / 5.0 * lps_zooms[2]) > 400  # BOA: force_split_threshold=400

    # GPU order: rough (small, unlocks crop preparation) -> total -> crops -> BCA
    order: list[str] = (["_rough"] if run_rough else []) + [n for n in tasks if n == "total"]
    order += crop_tasks + [n for n in tasks if n in ("body_parts", "body_regions")]

    executor = ThreadPoolExecutor(max_workers=max(1, opt.threads))
    prep_futures: dict[str, Future] = {}
    final_futures: dict[str, Future] = {}
    save_futures: list[Future] = []
    gpu_time: dict[str, float] = {}

    prep_lock = threading.Lock()

    def submit_prep(name: str, fn: Callable[[], Prepared]):
        with prep_lock:
            if name not in prep_futures:
                prep_futures[name] = executor.submit(fn)

    def crop_prep(name: str, source_img: nib.Nifti1Image, source: str):
        task = TASKS[name]
        mask = roi_mask(source_img, task.crop)
        return engine.prepare(task, mask, crop_key=(source, task.crop))

    def submit_crop_preps(source_future: Future, source: str):
        def cb(_f):
            if _f.exception() is None:
                for n in crop_tasks:
                    submit_prep(n, lambda n=n: crop_prep(n, _f.result(), source))

        source_future.add_done_callback(cb)

    # CPU preparation of everything that does not depend on another model
    if run_rough:
        submit_prep("_rough", lambda: engine.prepare(ROUGH))
    for n in tasks:
        if n == "total" or n in ("body_parts", "body_regions"):
            submit_prep(n, lambda n=n: engine.prepare(task_obj(n)))

    def finalize(name: str, prep: Prepared, seg):
        img_out = engine.finalize(prep, seg)
        if name in ("body_parts", "body_regions"):
            img_out = postprocess_nifti(img_out, name)
        if save and not name.startswith("_"):
            save_futures.append(executor.submit(_save, img_out, output_dir / f"{name}{ext}"))
        return img_out

    try:
        progress(f"Running {len(order)} segmentation step(s): {', '.join(_STEP_NAMES.get(n, n) for n in order)}")
        with _Timer(stats, "segmentation"):
            for step, name in enumerate(order, 1):
                if name in crop_tasks:
                    # Normally already submitted by the callback; this also surfaces errors
                    source = "total" if use_total_crop else "rough"
                    src_img = final_futures["total" if use_total_crop else "_rough"].result()
                    submit_prep(name, lambda n=name, s=src_img: crop_prep(n, s, source))
                task = ROUGH if name == "_rough" else task_obj(name)
                n_folds = (1 if opt.fast_bca else 5) if task.kind == "bca" else 1
                detail = f"{len(task.task_ids)} networks" if task.multimodel else (
                    f"{n_folds}-fold ensemble" if n_folds > 1 else "1 network")
                progress(f"  [{step}/{len(order)}] {_STEP_NAMES.get(name, name)} ({detail}) ...")
                prep = prep_futures[name].result()
                if prep.empty:
                    progress(f"      crop region not found -> empty {name}")
                split = engine.will_split(prep, bca_force_split and task.kind == "bca")
                if split:
                    progress("      large image: predicting in 3 overlapping parts")
                current.update(name=name, i=0, n=len(task.task_ids) * (3 if split else 1))
                t0 = time.perf_counter()
                if name == "_rough":
                    seg = engine.infer(prep)
                elif name in ("body_parts", "body_regions"):
                    folds = (0,) if opt.fast_bca else None
                    seg = engine.infer(prep, folds=folds, force_split=bca_force_split)
                else:
                    seg = engine.infer(prep)
                gpu_time[name] = round(time.perf_counter() - t0, 3)
                logger.info("Model %s inferred in %.2fs", name, gpu_time[name])
                remaining = len(order) - step
                progress(f"  [{step}/{len(order)}] {_STEP_NAMES.get(name, name)} done in {gpu_time[name]:.1f}s"
                         + (f", {remaining} step(s) remaining" if remaining else ""))  # fmt: skip
                final_futures[name] = executor.submit(finalize, name, prep, seg)
                if name == "_rough":
                    submit_crop_preps(final_futures[name], "rough")
                elif name == "total" and use_total_crop:
                    submit_crop_preps(final_futures[name], "total")
            progress("Finishing segmentations (resampling back, post-processing, saving) ...")
            for name in tasks:
                res.segmentations[name] = final_futures[name].result()
        stats["inference_time_per_model"] = gpu_time
        # Free cached intermediate images
        for key in ("rsp", "crop", "base"):
            ctx.forget(key)

        # ------------------------------------------------------------ metrics
        measured = {n: res.segmentations[n] for n in tasks if TASKS[n].kind == "ts"} if opt.measurements else {}
        if measured:
            progress("Computing measurements ...")
            with _Timer(stats, "measurements"):
                res.measurements, pfav = compute_measurements(ctx.orig, measured, opt.cnr_adjustment)
                if pfav is not None and save:
                    hdr = ctx.orig.header.copy()
                    hdr.set_data_dtype(np.uint8)
                    pfav_img = nib.Nifti1Image(pfav.astype(np.uint8), ctx.orig.affine, hdr)
                    save_futures.append(executor.submit(_save, pfav_img, output_dir / f"ct_pfav{ext}"))

        bca_future = None
        if run_bca:

            progress(
                "Computing body composition (tissues, BCA measurements) ..."
                if opt.measurements else "Deriving tissues ..."
            )  # fmt: skip

            def run_bca_fn():
                with _Timer(stats, "bca"):
                    tissues = subclassify_tissues(
                        ctx.orig, res.segmentations["body_regions"], opt.bca_median_filtering
                    )
                    res.segmentations["tissues"] = tissues
                    if save:
                        save_futures.append(executor.submit(_save, tissues, output_dir / f"tissues{ext}"))
                    stats["bca_regions"] = body_region_flags(res.segmentations["body_regions"])
                    if not opt.measurements:
                        return None, None
                    bca_json, vertebrae = compute_bca_measurements(
                        ctx.orig,
                        res.segmentations.get("body_parts"),
                        res.segmentations["body_regions"],
                        tissues,
                        res.segmentations.get("total"),
                        opt.bca_examined_body_region,
                    )
                    return bca_json, vertebrae

            bca_future = executor.submit(run_bca_fn)

        info: list[dict[str, Any]] = [{"name": "BOALightVersion", "value": __version__}, *dicom_info]
        regions_df = cnr_df = None
        if res.measurements is not None:
            with _Timer(stats, "segmentator_metrics"):
                extra_info, regions_df, cnr_df = segmentator_metrics(
                    res.measurements,
                    ctx.orig,
                    res.segmentations.get("total"),
                    res.segmentations.get("body_parts"),
                    [n for n in tasks if TASKS[n].kind == "ts" and n != "total"],
                )
                info += extra_info

        if bca_future is not None:
            res.bca_measurements, vertebrae = bca_future.result()
            if output_dir is not None and res.bca_measurements is not None:
                (output_dir / "bca-measurements.json").write_text(json.dumps(res.bca_measurements, indent=2))
                if vertebrae:
                    (output_dir / "vertebrae.json").write_text(json.dumps(vertebrae, indent=2))

        if output_dir is not None and res.measurements is not None:
            (output_dir / "total-measurements.json").write_text(json.dumps(res.measurements, indent=2))

        for f in save_futures:  # all segmentations on disk
            f.result()

        if opt.measurements and opt.contrast and "total" in tasks and save:
            progress("Predicting contrast phase (if boa-contrast is installed) ...")
            info += _contrast(ct_path or output_dir / f"image{ext}", ctx, output_dir, ext, stats)

        res.info = info
        if output_dir is not None and opt.measurements and opt.excel:
            progress("Writing output.xlsx ...")
            with _Timer(stats, "excel"):
                res.excel_path = write_excel(output_dir / "output.xlsx", info, regions_df, cnr_df, res.bca_measurements)
    except BaseException:
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown(wait=True)

    stats["total_time"] = round(time.perf_counter() - t_start, 3)
    if output_dir is not None and opt.measurements:
        (output_dir / "boa_light_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    logger.info("Complete CT analysis: DONE in %.2fs", stats["total_time"])
    progress(f"Done in {stats['total_time']:.1f}s" + (f" -> {output_dir}" if output_dir is not None else ""))
    return res


def _contrast(ct_path: Path, ctx: ImageContext, output_dir: Path, ext: str, stats) -> list[dict]:
    try:
        from boa_contrast import predict  # optional dependency
    except ImportError:
        logger.info("boa-contrast not installed; skipping contrast phase prediction.")
        return []
    if not Path(ct_path).exists():
        _save(ctx.orig, output_dir / f"image{ext}")
        ct_path = output_dir / f"image{ext}"
    kwargs = {} if ext == ".nii.gz" else {"total_segmentation_name": f"total{ext}"}
    try:
        with _Timer(stats, "contrast"):
            out = predict(ct_path=ct_path, segmentation_folder=output_dir, one_mask_per_file=False, **kwargs)
        stats["iv_contrast_phase"] = out["phase_ensemble_prediction"]
        stats["git_contrast"] = out["git_ensemble_prediction"]
        return [
            {"name": "PredictedContrastPhase", "value": out["phase_ensemble_predicted_class"]},
            {"name": "PredictedContrastInGIT", "value": out["git_ensemble_predicted_class"]},
        ]
    except Exception:
        logger.warning("Contrast phase prediction failed", exc_info=True)
        return []


# ---------------------------------------------------------------------------- #
# Excel output (same sheets as BOA's output.xlsx)
# ---------------------------------------------------------------------------- #

BCA_BODY_REGIONS = [
    "Whole Scan", "Abdominal Cavity", "Thoracic Cavity", "Ventral Cavity", "Mediastinum", "Pericardium",
    "L5", "L4", "L3", "L2", "L1", "T12", "T11", "T10", "T9", "T8", "T7", "T6", "T5", "T4", "T3", "T2", "T1",
    "C7", "C6", "C5", "C4", "C3", "C2", "C1",
]  # fmt: skip


def bca_tables(bca: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Port of BOA ``compute_bca_metrics``."""
    from boa_light.measurements import convert_name

    example = bca["aggregated"]["whole_scan"]["measurements"]
    rows = list(example["bone"].keys())
    cols = list(example.keys())
    rename_index = {r: r.split("_")[0].capitalize() + ("_mL" if "hu" not in r else "_HU") for r in rows}
    rename_cols = {c: (c.upper() if c not in ["bone", "muscle"] else c.capitalize()) for c in cols}
    rename_cols["index"] = "AggregationType"
    dfs = [pd.DataFrame(columns=["BodyPart", "Present", "AggregationType"])]
    has_nl = "slices_no_extremities" in bca
    for name in BCA_BODY_REGIONS:
        key = name.lower().replace(" ", "_").replace("-", "_")
        if key not in bca["aggregated"]:
            dfs.append(
                pd.DataFrame(
                    [{"BodyPart": convert_name(key), "Present": False}]
                    + ([{"BodyPart": f"{convert_name(key)}_NoExtremities", "Present": False}] if has_nl else [])
                )
            )
            continue
        for meas in ["measurements", "measurements_no_extremities"]:
            if meas not in bca["aggregated"][key]:
                continue
            df = (
                pd.DataFrame.from_dict(bca["aggregated"][key][meas])
                .rename(index=rename_index)
                .reset_index()
                .rename(columns=rename_cols)
            )
            df["Present"] = True
            part = convert_name(meas.replace("measurements", ""))
            df["BodyPart"] = convert_name(key) + ("_" + part if part else "")
            dfs.append(df)
    agg = pd.concat([d for d in dfs if len(d)] or dfs[:1])
    agg = agg.reindex(columns=list(dict.fromkeys(["BodyPart", "Present", "AggregationType", *agg.columns])))
    rename_cols["index"] = "SliceNumber"
    out = []
    for key in ("slices", "slices_no_extremities"):
        if key not in bca:
            out.append(None)
            continue
        df = pd.DataFrame(bca[key])
        df.index = df.index + 1
        out.append(df.reset_index().rename(columns=rename_cols))
    return agg, out[0], out[1]


def write_excel(path: Path, info, regions_df, cnr_df, bca) -> Path:
    info_df = pd.DataFrame(info).set_index("name")
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        info_df.to_excel(writer, sheet_name="info", header=False)
        if regions_df is not None and len(regions_df):
            regions_df.to_excel(writer, sheet_name="regions-statistics", index=False)
        if cnr_df is not None and len(cnr_df):
            cnr_df.to_excel(writer, sheet_name="cnr-adjusted", startrow=1, index=False)
            ws = writer.sheets["cnr-adjusted"]
            fmt = writer.book.add_format({"bold": True, "bg_color": "#FFF2CC", "align": "center", "text_wrap": True})
            ws.merge_range(
                0,
                0,
                0,
                max(1, len(cnr_df.columns) - 1),
                "These results were yielded by a modified version of BOA, adjusted for image quality assessment.",
                fmt,
            )
        if bca is not None:
            agg, slices, slices_nl = bca_tables(bca)
            agg.to_excel(writer, sheet_name="bca-aggregated-measurements", index=False)
            slices.to_excel(writer, sheet_name="bca-slice-measurements", index=False)
            if slices_nl is not None:
                slices_nl.to_excel(writer, sheet_name="bca-slice-measurements_no_ext", index=False)
    return path
