"""Command-line entry point: one render in, annotated image and JSON out."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import cv2 as cv
import numpy as np

from .config import PipelineConfig, PreprocessConfig, SeedConfig, SemanticsConfig
from .export import DEFAULT_SIMPLIFY, annotated_image, to_record, write_json
from .pipeline import label_rooms, segment_floorplan
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
                      help="distance to the estimated wall colour in OpenCV 8-bit Lab units "
                           "(default %(default)s)")
    tune.add_argument("--wall-dilate", type=int, default=PreprocessConfig.wall_dilate,
                      help="side of the square barrier dilation kernel in px; 3 adds a 1 px rim "
                           "that covers the shaded wall face, 0 disables (default %(default)s)")
    tune.add_argument("--h-maxima", type=float, default=SeedConfig.h_maxima,
                      help="marker significance; raise to get fewer rooms (default %(default)s)")
    tune.add_argument("--merge-width", type=float, default=SeedConfig.passage_merge_width,
                      help="boundaries wider than this px are not doorways and get merged "
                           "(default %(default)s)")
    tune.add_argument("--min-area", type=float, default=SeedConfig.min_region_area_frac,
                      help="drop regions below this fraction of the plan (default %(default)s)")
    tune.add_argument("--simplify", type=float, default=DEFAULT_SIMPLIFY,
                      help="polygon simplification as fraction of perimeter (default %(default)s)")

    sem = p.add_argument_group("optional semantic stage (vision-language model)")
    sem.add_argument("--semantics", action="store_true",
                     help="name rooms and split open-plan areas via OpenAI; "
                          "needs OPENAI_API_KEY")
    sem.add_argument("--model", default=SemanticsConfig.model,
                     help="chat-completions model (default %(default)s)")
    return p


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


def _imwrite(path: Path, image: np.ndarray) -> None:
    """Write an image, raising instead of returning ``False`` on failure."""
    if not cv.imwrite(str(path), image):
        raise OSError(f"could not write image: {path}")


def _write_debug(seg, out_dir: Path, stem: str) -> None:
    dbg = out_dir / f"{stem}_debug"
    dbg.mkdir(parents=True, exist_ok=True)
    _imwrite(dbg / "plan.png", (seg.plan * 255).astype(np.uint8))
    _imwrite(dbg / "wall.png", (seg.wall * 255).astype(np.uint8))
    _imwrite(dbg / "barrier.png", (seg.barrier * 255).astype(np.uint8))
    _imwrite(dbg / "regions.png", overlay_labels(seg.image, seg.labels))


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline on one image.

    Returns
    -------
    int
        Process exit code: 0 on success, 1 on a handled failure. When only
        the optional semantic stage fails, the geometric result is still
        written and the exit code is 1.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.semantics:
        from dotenv import load_dotenv

        load_dotenv()

    cfg = _config_from_args(args)
    try:
        # Geometry first, on its own, so that a failing semantic stage
        # cannot take an already computed result down with it.
        geometric_cfg = replace(cfg, semantics=SemanticsConfig(enabled=False))
        seg = segment_floorplan(args.image, geometric_cfg)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except ValueError as exc:
        logger.error("segmentation failed: %s", exc)
        return 1

    exit_code = 0
    if args.semantics:
        try:
            seg = label_rooms(seg, cfg.semantics)
        except SemanticsError as exc:
            logger.error("semantic stage failed: %s", exc)
            logger.error("writing the geometric result without room names")
            exit_code = 1

    record = to_record(seg, simplify=args.simplify)
    stem = args.image.stem
    json_path = args.output_dir / f"{stem}.json"
    png_path = args.output_dir / f"{stem}.png"
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(record, json_path)
        _imwrite(png_path, annotated_image(seg, record))
        if args.debug:
            _write_debug(seg, args.output_dir, stem)
    except OSError as exc:
        logger.error("could not write results: %s", exc)
        return 1

    logger.info("%d rooms -> %s, %s", len(record["rooms"]), png_path, json_path)
    for r in record["rooms"]:
        name = r["name"] or f"room {r['id']}"
        logger.info("  %-18s %6.1f%%  %7d px", name, 100 * r["relative_area"], r["area_px"])
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
