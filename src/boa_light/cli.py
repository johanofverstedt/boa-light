"""Command line interface: ``boa-light``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from boa_light import __version__
from boa_light.models import ALL_DEFAULT, NETWORKS, SELECTABLE, TASKS, networks_for, resolve_models


def _run(args) -> int:
    from boa_light.pipeline import Options, analyze

    opt = Options(
        models=args.models,
        measurements=args.measurements,
        device=args.device,
        fast_total=args.fast_total,
        fast_bca=args.fast_bca,
        crop_source=args.crop_source,
        cnr_adjustment=args.cnr_adjustment,
        contrast=not args.skip_contrast_information,
        bca_median_filtering=args.bca_median_filtering,
        bca_examined_body_region=args.bca_examined_body_region,
        bca_skip_body_parts=args.bca_skip_body_parts,
        save_format=args.save_format,
        threads=args.threads,
        keep_models_loaded=not args.low_memory,
        download=not args.no_download,
        fast_export=not args.no_fast_export,
        progress=not args.quiet,
        export_folds=args.export_folds,
        thymus_weights=args.thymus_weights,
        tile_batch=args.tile_batch,
        resample_threads=args.resample_threads,
    )
    analyze(args.input, args.output, opt)
    return 0


def _download(args) -> int:
    from boa_light import weights

    tasks, _ = resolve_models(args.models, args.bca_skip_body_parts)
    ids = networks_for(tasks, args.fast_total) if not args.everything else list(NETWORKS)
    if args.fast_total and not args.everything:
        ids = list(dict.fromkeys(ids + networks_for(tasks, False)))
    print(f"Weights directory: {weights.weights_dir()}")
    for tid in ids:
        state = "present" if weights.is_available(tid) else "downloading"
        print(f"  {tid:4d} {NETWORKS[tid].folder}: {state}")
        weights.download(tid)
    if "thymus" in tasks or args.everything:
        from boa_light import thymus

        path = thymus.find_weights()
        print(f"  thymus (Zenodo {thymus.ZENODO_RECORD}): {'present' if path else 'downloading'}")
        thymus.download_weights()
    return 0


def _list(_args) -> int:
    from boa_light import weights

    print("Selectable models (default 'all' = " + "+".join(ALL_DEFAULT) + "):")
    print(f"  {'bca':28s} Body Composition Analysis (adds total, body_parts, body_regions, tissues)")
    for name, t in TASKS.items():
        print(f"  {name:28s} {t.description}")
    print(f"\nWeights directory: {weights.weights_dir()}")
    for tid, net in NETWORKS.items():
        print(f"  {tid:4d} {'[x]' if weights.is_available(tid) else '[ ]'} {net.folder}")
    from boa_light import thymus

    print(f"  thymus {'[x]' if thymus.find_weights() else '[ ]'} {thymus.thymus_weights_dir()}")
    return 0


def _models_help() -> str:
    lines = ["models (combine with '+', e.g. -m total+bca+liver_vessels):"]
    lines.append(f"  {'all':27s} default: {'+'.join(ALL_DEFAULT)}")
    lines.append(f"  {'bca':27s} Body Composition Analysis: adds total, body_parts, body_regions and")
    lines.append(f"  {'':27s} derives tissues.nii.gz + bca-measurements.json")
    for name, t in TASKS.items():
        extra = " (not in 'all')" if name not in ALL_DEFAULT and t.kind != "bca" else ""
        lines.append(f"  {name:27s} {t.description}{extra}")
    return "\n".join(lines)


RUN_EXAMPLES = """examples:
  boa-light run ct.nii.gz out/                         all open models, label images only
  boa-light run ct.nii.gz out/ --measurements          + all BOA measurements/reports (JSON, Excel)
  boa-light run dicom_dir/ out/ -m total+bca           DICOM input (needs the [dicom] extra)
  boa-light run ct.nii.gz out/ -m bca --bca-skip-body-parts
                                                       BCA without the body_parts model
  boa-light run ct.nii.gz out/ -m total --fast-total   quick 3 mm TotalSegmentator only
  boa-light run ct.nii.gz out/ -m all+thymus           also the thymus region (anterior mediastinal fat)
  boa-light run ct.nii.gz out/ --fast-bca --crop-source total
                                                       fastest full run (not identical to BOA)

outputs (in OUTPUT): <model>.nii.gz label images (+ tissues.nii.gz with bca);
  with --measurements also ct_pfav.nii.gz, total-measurements.json, bca-measurements.json,
  vertebrae.json, output.xlsx and boa_light_stats.json (timings)

weights are stored in $BOA_LIGHT_WEIGHTS, $TOTALSEG_WEIGHTS_PATH or
  ~/.totalsegmentator/nnunet/results (shared with TotalSegmentator)"""


def build_parser() -> argparse.ArgumentParser:
    fmt = argparse.RawDescriptionHelpFormatter
    p = argparse.ArgumentParser(
        prog="boa-light",
        formatter_class=fmt,
        description="Fast, lightweight BOA (Body and Organ Analysis) / TotalSegmentator runner "
        "using only openly licensed models.",
        epilog="commands:\n"
        "  boa-light run --help        options for analysing a CT\n"
        "  boa-light download --help   pre-fetch model weights\n"
        "  boa-light list              models and installed weights\n\n" + _models_help(),
    )
    p.add_argument("--version", action="version", version=f"boa-light {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="detailed log output")
    sub = p.add_subparsers(dest="command", required=True, metavar="{run,download,list}")

    # ------------------------------------------------------------------ run
    r = sub.add_parser(
        "run",
        help="analyse one CT (NIfTI file or DICOM directory)",
        formatter_class=fmt,
        description="Segment one CT with the selected models and compute BOA's measurements.",
        epilog=_models_help() + "\n\n" + RUN_EXAMPLES,
    )
    r.add_argument("input", type=Path, help="input .nii / .nii.gz file or DICOM series directory")
    r.add_argument("output", type=Path, help="output directory (created if missing)")
    r.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS, help="detailed log output")
    r.add_argument("-q", "--quiet", action="store_true", help="no progress messages")

    g = r.add_argument_group("model selection")
    g.add_argument(
        "-m", "--models", default="all", metavar="MODELS",
        help="'+'-separated list of models, or 'all' (default). See the model list below.",
    )  # fmt: skip
    g.add_argument(
        "--thymus-weights", type=Path, default=None, metavar="PATH",
        help="checkpoint (.pt) of the Open Thymus Segmentator, if not downloaded automatically "
        "from Zenodo (record 18355851)",
    )  # fmt: skip
    g.add_argument(
        "--bca-skip-body-parts", action="store_true",
        help="run bca without the body_parts model (saves a 5-fold network). Omits only the "
        "'no extremities' BCA measurements (bca-slice-measurements_no_ext sheet, *_NoExtremities "
        "rows) and the L3 body axes (MaxAxisL3_cm, MinAxisL3_cm, MeanAxisL3_cm)",
    )  # fmt: skip

    g.add_argument(
        "--export-folds", action="store_true",
        help="for ensemble models (body_parts, body_regions: 5 folds; pleural_pericard_effusion: all "
        "available folds) also save the label map of every fold to OUTPUT/folds/<model>_fold<k>.nii.gz. "
        "No extra network passes; the ensemble result is unchanged. Ignored with --fast-bca for BCA",
    )  # fmt: skip

    g = r.add_argument_group("speed / accuracy trade-offs (default: identical to BOA)")
    g.add_argument("--fast-total", action="store_true",
                   help="use the single 3 mm TotalSegmentator model instead of the five 1.5 mm models")
    g.add_argument("--fast-bca", action="store_true",
                   help="use fold 0 only instead of the 5-fold ensembles for body_parts/body_regions")
    g.add_argument(
        "--crop-source", choices=["rough", "total"], default="rough",
        help="where the ROI masks for the cropped models (lung_vessels, liver_vessels, ...) come "
        "from: 'rough' = 6 mm total model, run once (identical to BOA, default); 'total' = the full "
        "'total' segmentation (one network less, slightly different crops)",
    )  # fmt: skip

    g.add_argument(
        "--tile-batch", type=int, default=1, metavar="N",
        help="run N sliding-window tiles per network call (default 1 = nnU-Net behaviour). 2-4 can be "
        "faster on large GPUs and needs more GPU memory; results may differ from BOA in rare voxels "
        "because GPU kernels for other batch sizes round differently",
    )  # fmt: skip

    g = r.add_argument_group("measurements and reports (off by default: only label images are written)")
    g.add_argument(
        "--measurements", "--reports", action="store_true",
        help="also compute BOA's measurements and reports: total-measurements.json, ct_pfav.nii.gz, "
        "bca-measurements.json, vertebrae.json, output.xlsx, boa_light_stats.json, contrast phase "
        "(and image.nii.gz for DICOM input). The options below only apply with --measurements",
    )  # fmt: skip
    g.add_argument("--cnr-adjustment", action="store_true",
                   help="add the 'cnr-adjusted' sheet (eroded aorta/autochthon measurements)")
    g.add_argument("--skip-contrast-information", action="store_true",
                   help="skip contrast phase prediction (only runs if boa-contrast is installed)")
    g.add_argument("--bca-median-filtering", action="store_true",
                   help="in-plane 3x3 median filter on the CT before tissue classification "
                   "(also affects tissues.nii.gz)")
    g.add_argument("--bca-examined-body-region", choices=["abdomen", "neck", "thorax"], default=None,
                   help="force the BCA aggregation region instead of detecting it")

    g = r.add_argument_group("hardware and output")
    g.add_argument("-d", "--device", default="auto",
                   help="auto (default: cuda if available), cuda, cuda:<id>, cpu, mps")
    g.add_argument("--save-format", choices=["nii.gz", "nii", "none"], default="nii.gz",
                   help="segmentation file format; 'nii' is faster to write, 'none' writes no images")
    g.add_argument("--threads", type=int, default=4, help="CPU worker threads for overlapping tasks (default 4)")
    g.add_argument("--resample-threads", type=int, default=None,
                   help="threads per image resampling (default: all CPUs; results are identical for any value)")
    g.add_argument("--low-memory", action="store_true",
                   help="unload each network after use (less GPU memory, slower)")
    g.add_argument("--no-download", action="store_true", help="fail instead of downloading missing weights")
    g.add_argument("--no-fast-export", action="store_true",
                   help="use nnU-Net's own CPU argmax instead of the (identical, faster) GPU argmax")
    r.set_defaults(func=_run)

    # ------------------------------------------------------------- download
    d = sub.add_parser(
        "download",
        help="download model weights",
        formatter_class=fmt,
        description="Download the weights needed for the given models (only missing ones).",
        epilog=_models_help() + "\n\nexample:\n  boa-light download -m total+bca --bca-skip-body-parts",
    )
    d.add_argument("-m", "--models", default="all", metavar="MODELS", help="as for 'run' (default: all)")
    d.add_argument("--fast-total", action="store_true", help="also fetch the 3 mm 'total' model")
    d.add_argument("--bca-skip-body-parts", action="store_true", help="do not fetch the body_parts model")
    d.add_argument("--everything", action="store_true", help="fetch every supported network")
    d.set_defaults(func=_download)

    # ----------------------------------------------------------------- list
    ls = sub.add_parser("list", help="list models and installed weights")
    ls.set_defaults(func=_list)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
