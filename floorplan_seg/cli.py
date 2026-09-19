"""Command-line entry point: one render in, annotated image and JSON out."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

from .config import PipelineConfig, PreprocessConfig, SeedConfig, SemanticsConfig
from .export import annotated_image, to_record, write_json
from .pipeline import segment_floorplan
from .semantics import SemanticsError
from .viz import overlay_labels

logger = logging.getLogger("floorplan_seg")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    p = argparse.ArgumentParser(
        prog="floorplan_seg",
        description="Approximate the room layout of a 3D floor-plan render.",
    )
    p.add_argument("image", type=Path, help="input render (webp/png/jpg)")
    p.add_argument("-o", "--output-dir", type=Path, default=Path("output"),
                   help="where to write <stem>.png and <stem>.json (default: output/)")
    p.add_argument("--debug", action="store_true",
                   help="also write intermediate masks (plan, walls, barrier, regions)")

    tune = p.add_argument_group("manual tuning (documented light correction)")
    tune.add_argument("--wall-delta-e", type=float, default=PreprocessConfig.wall_delta_e,
                      help="CIELAB distance to the estimated wall colour (default %(default)s)")
    tune.add_argument("--wall-dilate", type=int, default=PreprocessConfig.wall_dilate,
                      help="barrier dilation in px, covers the shaded wall face (default %(default)s)")
    tune.add_argument("--h-maxima", type=float, default=SeedConfig.h_maxima,
                      help="marker significance; raise to get fewer rooms (default %(default)s)")
    tune.add_argument("--merge-width", type=float, default=SeedConfig.passage_merge_width,
                      help="boundaries wider than this px are not doorways and get merged "
                           "(default %(default)s)")
    tune.add_argument("--min-area", type=float, default=SeedConfig.min_region_area_frac,
                      help="drop regions below this fraction of the plan (default %(default)s)")
    tune.add_argument("--simplify", type=float, default=0.004,
                      help="polygon simplification as fraction of perimeter (default %(default)s)")

    sem = p.add_argument_group("optional semantic stage (vision-language model)")
    sem.add_argument("--semantics", action="store_true",
                     help="name rooms and split open-plan areas via OpenAI; "
                          "needs OPENAI_API_KEY")
    sem.add_argument("--model", default=SemanticsConfig.model,
                     help="chat-completions model (default %(default)s)")
    return p


def _validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments and raise ValueError if invalid."""
    errors = []
    if args.wall_delta_e < 0:
        errors.append(f"--wall-delta-e must be non-negative, got {args.wall_delta_e}")
    if args.wall_dilate < 0:
        errors.append(f"--wall-dilate must be non-negative, got {args.wall_dilate}")
    if args.h_maxima <= 0:
        errors.append(f"--h-maxima must be positive, got {args.h_maxima}")
    if args.merge_width < 0:
        errors.append(f"--merge-width must be non-negative, got {args.merge_width}")
    if not 0 <= args.min_area <= 1:
        errors.append(f"--min-area must be between 0 and 1, got {args.min_area}")
    if not 0 <= args.simplify <= 1:
        errors.append(f"--simplify must be between 0 and 1, got {args.simplify}")
    if errors:
        raise ValueError("Invalid arguments:\n  " + "\n  ".join(errors))


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        preprocess=PreprocessConfig(
            wall_delta_e=args.wall_delta_e, wall_dilate=args.wall_dilate
        ),
        seeds=SeedConfig(
            h_maxima=args.h_maxima,
            passage_merge_width=args.merge_width,
            min_region_area_frac=args.min_area,
        ),
        semantics=SemanticsConfig(enabled=args.semantics, model=args.model),
    )


def _write_debug(seg, out_dir: Path, stem: str) -> None:
    dbg = out_dir / f"{stem}_debug"
    dbg.mkdir(parents=True, exist_ok=True)
    for name, img in [
        ("plan.png", (seg.plan * 255).astype(np.uint8)),
        ("wall.png", (seg.wall * 255).astype(np.uint8)),
        ("barrier.png", (seg.barrier * 255).astype(np.uint8)),
        ("regions.png", overlay_labels(seg.image, seg.labels)),
    ]:
        path = dbg / name
        if not cv.imwrite(str(path), img):
            logger.warning("failed to write debug image: %s", path)


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline on one image.

    Returns
    -------
    int
        Process exit code: 0 on success, 1 on a handled failure.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        _validate_args(args)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    if args.semantics:
        from dotenv import load_dotenv

        load_dotenv()

    try:
        seg = segment_floorplan(args.image, _config_from_args(args))
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except SemanticsError as exc:
        logger.error("semantic stage failed: %s", exc)
        logger.error("re-run without --semantics for the geometric result")
        return 1

    record = to_record(seg, simplify=args.simplify)
    stem = args.image.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{stem}.json"
    png_path = args.output_dir / f"{stem}.png"
    write_json(record, json_path)
    if not cv.imwrite(str(png_path), annotated_image(seg, record)):
        logger.error("failed to write output image: %s", png_path)
        return 1
    if args.debug:
        _write_debug(seg, args.output_dir, stem)

    logger.info("%d rooms -> %s, %s", len(record["rooms"]), png_path, json_path)
    for r in record["rooms"]:
        name = r["name"] or f"room {r['id']}"
        logger.info("  %-18s %6.1f%%  %7d px", name, 100 * r["relative_area"], r["area_px"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
