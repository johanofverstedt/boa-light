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
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="boa-light",
        description="Fast, lightweight BOA (Body and Organ Analysis) / TotalSegmentator runner "
        "using only openly licensed models.",
    )
    p.add_argument("--version", action="version", version=f"boa-light {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="Print progress logs")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Analyse one CT (NIfTI file or DICOM directory)")
    r.add_argument("input", type=Path, help="Input .nii/.nii.gz file or DICOM series directory")
    r.add_argument("output", type=Path, help="Output directory")
    r.add_argument(
        "-m", "--models", default="all",
        help=f"'+'-separated models or 'all' (default). Available: {', '.join(SELECTABLE)}",
    )  # fmt: skip
    r.add_argument("-d", "--device", default="auto", help="auto (default), cuda, cuda:<id>, cpu, mps")
    r.add_argument("--fast-total", action="store_true", help="Use the 3 mm TotalSegmentator model")
    r.add_argument("--fast-bca", action="store_true", help="Use one fold instead of the 5-fold BCA ensemble")
    r.add_argument(
        "--crop-source", choices=["rough", "total"], default="rough",
        help="ROI masks for the cropped subtasks: 'rough' = 6 mm model, computed once (identical to "
        "BOA, default); 'total' = reuse the full 'total' segmentation (no extra model)",
    )  # fmt: skip
    r.add_argument("--cnr-adjustment", action="store_true", help="Add CNR-adjusted aorta/autochthon sheet")
    r.add_argument("--skip-contrast-information", action="store_true", help="Skip boa-contrast prediction")
    r.add_argument("--bca-median-filtering", action="store_true")
    r.add_argument("--bca-examined-body-region", choices=["abdomen", "neck", "thorax"], default=None)
    r.add_argument(
        "--bca-skip-body-parts", action="store_true",
        help="Do not run the body_parts model for bca: omits the 'no extremities' BCA measurements "
        "and the L3 body axes (MaxAxisL3_cm etc.)",
    )  # fmt: skip
    r.add_argument("--save-format", choices=["nii.gz", "nii", "none"], default="nii.gz")
    r.add_argument("-q", "--quiet", action="store_true", help="No progress messages")
    r.add_argument("--threads", type=int, default=4, help="CPU worker threads (default 4)")
    r.add_argument("--low-memory", action="store_true", help="Unload each network after use")
    r.add_argument("--no-download", action="store_true", help="Fail instead of downloading missing weights")
    r.add_argument(
        "--no-fast-export", action="store_true",
        help="Use nnU-Net's own CPU argmax instead of the (equivalent) GPU argmax",
    )  # fmt: skip
    r.set_defaults(func=_run)

    d = sub.add_parser("download", help="Download model weights")
    d.add_argument("-m", "--models", default="all")
    d.add_argument("--fast-total", action="store_true", help="Also fetch the 3 mm 'total' model")
    d.add_argument("--bca-skip-body-parts", action="store_true", help="Do not fetch the body_parts model")
    d.add_argument("--everything", action="store_true", help="Fetch every supported network")
    d.set_defaults(func=_download)

    ls = sub.add_parser("list", help="List models and installed weights")
    ls.set_defaults(func=_list)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
