"""Regenerate the assets and metrics of the floor-finish split approach.

Usage
-----
    uv run python research/superpixel-finish-split/run.py            # all samples
    uv run python research/superpixel-finish-split/run.py --image limestone

Writes ``assets/*.webp`` and ``assets/metrics.json`` next to this file. Nothing
outside ``research/`` is touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2 as cv
import numpy as np
from skimage.segmentation import find_boundaries

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from approach import FinishSplitConfig, split_open_plan  # noqa: E402
from research import common  # noqa: E402

ASSETS = HERE / "assets"


def _superpixel_figure(image, sp, region):
    """Render with the SLIC lattice drawn inside the candidate region."""
    out = image.copy()
    edges = find_boundaries(sp, mode="thick") & region
    out[edges] = (40, 40, 40)
    out[~region] = cv.addWeighted(
        image, 0.3, np.full_like(image, 255), 0.7, 0
    )[~region]
    return out


def _seed_figure(image, seeds, report, region):
    """Accepted finish seeds tinted; everything else faded out."""
    out = cv.addWeighted(image, 0.35, np.full_like(image, 255), 0.65, 0)
    out[region] = cv.addWeighted(
        image[region].reshape(-1, 1, 3), 0.75,
        np.full((region.sum(), 1, 3), 255, np.uint8), 0.25, 0
    ).reshape(-1, 3)
    colours = common.palette(seeds)
    tags = {}
    for lbl in colours:
        accepted = [r for r in report if r["finish"] == lbl and r["accepted"]]
        share = sum(r["area_frac"] for r in accepted)
        contact = max((r["wall_contact"] for r in accepted), default=0.0)
        tags[lbl] = f"finish {lbl} | {share:.0%} | wall {contact:.0%}"
    return common.overlay(out, seeds, colours, tags, alpha=0.5, font_scale=0.45)


def _semantic_reference(path: Path) -> list[dict] | None:
    """Relative areas of the committed ``--semantics`` run, when present."""
    record = common.REPO_ROOT / "output" / "semantic" / f"{path.stem}.json"
    if not record.is_file():
        return None
    with record.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [
        {"name": r["name"], "relative_area": r["relative_area"]} for r in data["rooms"]
    ]


def process(path: Path, cfg: FinishSplitConfig, write_assets: bool = True) -> dict:
    """Run baseline and approach on one render, write its assets."""
    name = common.short_name(path)
    seg = common.baseline(path)
    labels, reports, debug = split_open_plan(
        seg.image, seg.labels, seg.plan, seg.barrier, cfg
    )

    distance = common.distance_map(common.free_space(seg))
    base_palette = common.palette(seg.labels)
    new_palette = common.matched_palette(seg.labels, labels)

    base_overlay = common.overlay(seg.image, seg.labels, base_palette)
    new_overlay = common.overlay(seg.image, labels, new_palette)
    diff = common.difference_view(seg.image, seg.labels, labels)

    crop_input, crop_base, crop_new, crop_diff = common.crop_to(
        seg.plan, seg.image, base_overlay, new_overlay, diff
    )
    if write_assets:
        common.save(crop_input, ASSETS / f"{name}-01-input.webp")
        common.save(crop_base, ASSETS / f"{name}-02-baseline.webp")
        common.save(crop_new, ASSETS / f"{name}-03-approach.webp")
        common.save(
            common.row(
                [crop_base, crop_new],
                [f"baseline watershed - {int(seg.labels.max())} regions",
                 f"+ floor-finish split - {int(labels.max())} regions"],
            ),
            ASSETS / f"{name}-04-side-by-side.webp",
        )
        common.save(crop_diff, ASSETS / f"{name}-05-difference.webp")

    # Intermediates for the largest candidate region only: they are what the
    # decision is made on, and cropping keeps the files small.
    if debug and write_assets:
        region_id = max(debug, key=lambda r: int((seg.labels == r).sum()))
        region = seg.labels == region_id
        dbg = debug[region_id]
        seed_report = next(r.seeds for r in reports if r.region_id == region_id)
        figures = [
            _superpixel_figure(seg.image, dbg["superpixels"], region),
            common.overlay(
                seg.image,
                dbg["finishes"],
                tags={int(v): "" for v in np.unique(dbg["finishes"])},
                alpha=0.55,
            ),
            _seed_figure(seg.image, dbg["seeds"], seed_report, region),
        ]
        crops = common.crop_to(region, *figures, pad=60)
        common.save(crops[0], ASSETS / f"{name}-06-superpixels.webp")
        common.save(crops[1], ASSETS / f"{name}-07-finish-clusters.webp")
        common.save(crops[2], ASSETS / f"{name}-08-finish-seeds.webp")

        report = next(r for r in reports if r.region_id == region_id)
        if report.decision.startswith("declined") and "proposed_zones" in dbg:
            rejected = common.overlay(seg.image, dbg["proposed_zones"], alpha=0.45)
            common.save(
                common.caption(
                    common.crop_to(region, rejected, pad=24)[0],
                    f"rejected: cut ratio {report.cut_ratio} > {cfg.max_cut_ratio}",
                ),
                ASSETS / f"{name}-09-rejected-cut.webp",
            )

    return {
        "image": path.name,
        "short_name": name,
        "baseline_regions": int(seg.labels.max()),
        "approach_regions": int(labels.max()),
        "baseline_cuts": asdict(common.cut_stats(seg.labels, distance)),
        "approach_cuts": asdict(common.cut_stats(labels, distance)),
        "agreement_vs_baseline": common.agreement(seg.labels, labels),
        "baseline_rooms": common.region_table(seg.labels, seg.plan),
        "approach_rooms": common.region_table(labels, seg.plan),
        "candidate_regions": [asdict(r) for r in reports],
        "committed_semantic_run": _semantic_reference(path),
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point: process the selected renders and write ``metrics.json``."""
    defaults = FinishSplitConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--image", action="append", default=None,
        help="substring of a file name in data/; repeatable (default: all)",
    )
    parser.add_argument("--min-region-frac", type=float, default=defaults.min_region_frac,
                        help="candidate open-plan size guard (default %(default)s)")
    parser.add_argument("--max-cut-ratio", type=float, default=defaults.max_cut_ratio,
                        help="cut-shape guard (default %(default)s)")
    parser.add_argument("--max-zones", type=int, default=defaults.max_zones,
                        help="finishes a region may be cut into (default %(default)s)")
    ablation = parser.add_argument_group("feature ablation (see ALGORITHM.md)")
    ablation.add_argument("--lightness-weight", type=float, default=defaults.lightness_weight,
                          help="weight of median L (default %(default)s)")
    ablation.add_argument("--chroma-weight", type=float, default=defaults.chroma_weight,
                          help="weight of median a/b (default %(default)s)")
    ablation.add_argument("--contrast-weight", type=float, default=defaults.contrast_weight,
                          help="weight of the illumination-normalised contrast "
                               "(0 disables it, default %(default)s)")
    ablation.add_argument("--coherence-weight", type=float, default=defaults.coherence_weight,
                          help="weight of the orientation coherence (0 disables it, "
                               "default %(default)s)")
    parser.add_argument("--no-assets", action="store_true",
                        help="only print the table; do not write assets")
    args = parser.parse_args(argv)

    images = common.sample_images()
    if args.image:
        images = [p for p in images if any(s.lower() in p.name.lower() for s in args.image)]
    if not images:
        print("no matching images in data/", file=sys.stderr)
        return 1

    cfg = FinishSplitConfig(
        min_region_frac=args.min_region_frac,
        max_cut_ratio=args.max_cut_ratio,
        max_zones=args.max_zones,
        lightness_weight=args.lightness_weight,
        chroma_weight=args.chroma_weight,
        contrast_weight=args.contrast_weight,
        coherence_weight=args.coherence_weight,
    )
    results = [process(path, cfg, write_assets=not args.no_assets) for path in images]
    if not args.no_assets:
        common.write_json(
            {"config": asdict(cfg), "results": results}, ASSETS / "metrics.json"
        )

    rows = [
        {
            "render": r["short_name"],
            "baseline": r["baseline_regions"],
            "approach": r["approach_regions"],
            "decisions": "; ".join(
                f"#{c['region_id']} ({c['region_frac']:.0%} of plan) {c['decision']}"
                f"{' ' + str([f'{z:.0%}' for z in c['zones']]) if c['zones'] else ''}"
                f" cut={c['cut_ratio']}"
                for c in r["candidate_regions"]
            ),
            "agreement": f"{r['agreement_vs_baseline']['pixel_agreement']:.0%}",
        }
        for r in results
    ]
    print(common.markdown_table(rows, ["render", "baseline", "approach", "decisions", "agreement"]))
    if not args.no_assets:
        print(f"\nassets -> {ASSETS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
