"""Regenerate the assets and metrics of the medial-axis doorway approach.

Usage
-----
    uv run python research/medial-axis-doorways/run.py             # all samples
    uv run python research/medial-axis-doorways/run.py --image limestone

Writes ``assets/*.webp`` and ``assets/metrics.json`` next to this file. Nothing
outside ``research/`` is touched.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, replace
from pathlib import Path

import cv2 as cv
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from approach import RidgeCutConfig, passages, region_labels  # noqa: E402
from floorplan_seg.config import SeedConfig  # noqa: E402
from floorplan_seg.seeds import region_labels as baseline_region_labels  # noqa: E402
from research import common  # noqa: E402

ASSETS = HERE / "assets"

#: Parameter sweeps for the sensitivity figure. Both methods are moved by the
#: same relative amount, -35 % / default / +35 %, so the comparison is about
#: how much the *output* moves, not about how wide the sweep was.
BASELINE_SWEEP = (6.5, 10.0, 13.5)
RIDGE_SWEEP = (0.325, 0.5, 0.675)


def _medial_axis_figure(image, trace):
    """Full medial axis in blue, the part that survives the cut in red."""
    out = cv.addWeighted(image, 0.45, np.full_like(image, 255), 0.55, 0)
    thick = np.ones((2, 2), np.uint8)
    dropped = cv.dilate((trace["skeleton"] & ~trace["ridge"]).astype(np.uint8), thick).astype(bool)
    kept = cv.dilate(trace["ridge"].astype(np.uint8), thick).astype(bool)
    out[dropped] = (220, 120, 0)
    out[kept] = (30, 30, 220)
    return out


def _width_figure(image, trace, labels, found):
    """Free-space width as a colour map, with the measured passages marked."""
    distance = trace["distance"]
    free = trace["free"]
    scaled = np.zeros(distance.shape, np.uint8)
    top = max(float(distance.max()), 1.0)
    scaled[free] = np.clip(255.0 * distance[free] / top, 0, 255).astype(np.uint8)
    out = cv.applyColorMap(scaled, cv.COLORMAP_TURBO)
    out[~free] = (255, 255, 255)
    out[cv.dilate(trace["ridge"].astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)] = (
        20, 20, 20
    )
    out[common.boundaries(labels)] = (255, 255, 255)
    for p in found:
        centre = (int(p["x"]), int(p["y"]))
        radius = max(6, int(p["width_px"] / 2))
        cv.circle(out, centre, radius, (255, 255, 255), 3, cv.LINE_AA)
        cv.circle(out, centre, radius, (0, 0, 220), 2, cv.LINE_AA)
        text = f"{p['width_px']:.0f}px"
        (tw, th), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        origin = (centre[0] - tw // 2, centre[1] - radius - 6)
        cv.rectangle(out, (origin[0] - 3, origin[1] - th - 3),
                     (origin[0] + tw + 3, origin[1] + 3), (255, 255, 255), cv.FILLED)
        cv.putText(out, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv.LINE_AA)
    return out


def _tie_break_stability(seg, cfg, trials: int) -> dict:
    """Region counts of the ridge variant across medial-axis tie-break seeds."""
    counts = []
    for seed in range(trials):
        labels, _ = region_labels(
            seg.plan, seg.barrier, seg.wall, replace(cfg, marker_source="ridge", rng_seed=seed)
        )
        counts.append(int(labels.max()))
    return {"trials": trials, "region_counts": counts, "distinct": sorted(set(counts))}


def _sensitivity(seg, cfg):
    """Region counts and overlays for both methods across their sweeps."""
    rows, panels = [], []
    for h in BASELINE_SWEEP:
        labels, _ = baseline_region_labels(seg.plan, seg.barrier, SeedConfig(h_maxima=h))
        rows.append({"method": "baseline", "parameter": f"h={h:g}", "regions": int(labels.max())})
        panels.append((f"baseline h-maxima={h:g}: {int(labels.max())} regions", labels))
    for scale in RIDGE_SWEEP:
        labels, _ = region_labels(
            seg.plan, seg.barrier, seg.wall,
            RidgeCutConfig(passage_scale=scale, merge_width=cfg.merge_width,
                           min_region_area_frac=cfg.min_region_area_frac),
        )
        rows.append({"method": "medial axis", "parameter": f"scale={scale:g}",
                     "regions": int(labels.max())})
        panels.append((f"medial axis scale={scale:g}: {int(labels.max())} regions", labels))
    return rows, panels


def process(
    path: Path, cfg: RidgeCutConfig, write_assets: bool = True, seed_trials: int = 6
) -> dict:
    """Run baseline and approach on one render, write its assets."""
    name = common.short_name(path)
    seg = common.baseline(path)
    labels, trace = region_labels(seg.plan, seg.barrier, seg.wall, cfg)
    other = "cores" if cfg.marker_source == "ridge" else "ridge"
    variant_labels, _ = region_labels(
        seg.plan, seg.barrier, seg.wall, replace(cfg, marker_source=other)
    )

    distance = trace["distance"]
    found = passages(labels, distance)
    base_palette = common.palette(seg.labels)
    new_palette = common.matched_palette(seg.labels, labels)

    base_overlay = common.overlay(seg.image, seg.labels, base_palette)
    new_overlay = common.overlay(seg.image, labels, new_palette)

    sweep_rows, sweep_panels = _sensitivity(seg, cfg)

    if write_assets:
        crops = common.crop_to(
            seg.plan,
            seg.image,
            base_overlay,
            new_overlay,
            common.difference_view(seg.image, seg.labels, labels),
            _medial_axis_figure(seg.image, trace),
            _width_figure(seg.image, trace, labels, found),
        )
        crop_input, crop_base, crop_new, crop_diff, crop_axis, crop_width = crops
        common.save(crop_input, ASSETS / f"{name}-01-input.webp")
        common.save(crop_base, ASSETS / f"{name}-02-baseline.webp")
        common.save(crop_new, ASSETS / f"{name}-03-approach.webp")
        common.save(
            common.row(
                [crop_base, crop_new],
                [f"baseline h-maxima watershed - {int(seg.labels.max())} regions",
                 f"medial-axis passages - {int(labels.max())} regions"],
            ),
            ASSETS / f"{name}-04-side-by-side.webp",
        )
        common.save(crop_diff, ASSETS / f"{name}-05-difference.webp")
        common.save(crop_axis, ASSETS / f"{name}-06-medial-axis.webp")
        common.save(crop_width, ASSETS / f"{name}-07-width-map.webp")

        sweep_images, sweep_titles = [], []
        for title, sweep_labels in sweep_panels:
            panel = common.overlay(
                seg.image, sweep_labels, alpha=0.4,
                tags={int(v): "" for v in np.unique(sweep_labels[sweep_labels > 0])},
            )
            sweep_images.append(common.crop_to(seg.plan, panel)[0])
            sweep_titles.append(title)
        half = len(sweep_images) // 2
        common.save(
            np.vstack([
                common.row(sweep_images[:half], sweep_titles[:half]),
                common.row(sweep_images[half:], sweep_titles[half:]),
            ]),
            ASSETS / f"{name}-08-sensitivity.webp",
        )

    return {
        "image": path.name,
        "short_name": name,
        "marker_source": trace["marker_source"],
        "wall_thickness_px": trace["wall_thickness_px"],
        "passage_half_width_px": trace["passage_half_width_px"],
        "passage_markers": trace["passage_markers"],
        "markers": trace["markers"],
        "free_space_covered": round(
            float((labels > 0).sum() / max((seg.plan & ~seg.barrier).sum(), 1)), 4
        ),
        "baseline_free_space_covered": round(
            float((seg.labels > 0).sum() / max((seg.plan & ~seg.barrier).sum(), 1)), 4
        ),
        "regions_before_postprocess": trace["regions_before_postprocess"],
        "baseline_regions": int(seg.labels.max()),
        "approach_regions": int(labels.max()),
        "baseline_cuts": asdict(common.cut_stats(seg.labels, distance)),
        "approach_cuts": asdict(common.cut_stats(labels, distance)),
        "agreement_vs_baseline": common.agreement(seg.labels, labels),
        "passages": found,
        "baseline_rooms": common.region_table(seg.labels, seg.plan),
        "approach_rooms": common.region_table(labels, seg.plan),
        "sensitivity": sweep_rows,
        "marker_variant": {
            "source": other,
            "regions": int(variant_labels.max()),
            "vs_primary": common.agreement(labels, variant_labels),
        },
        "tie_break_stability": _tie_break_stability(seg, cfg, seed_trials),
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point: process the selected renders and write ``metrics.json``."""
    defaults = RidgeCutConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--image", action="append", default=None,
        help="substring of a file name in data/; repeatable (default: all)",
    )
    parser.add_argument("--passage-scale", type=float, default=defaults.passage_scale,
                        help="passage width as a multiple of the wall thickness "
                             "(default %(default)s)")
    parser.add_argument("--passage-half-width", type=float, default=None,
                        help="absolute passage half-width in px; overrides the scale")
    parser.add_argument("--markers", choices=("ridge", "cores"), default=defaults.marker_source,
                        help="cut the medial axis or threshold the distance map "
                             "(default %(default)s)")
    parser.add_argument("--seed-trials", type=int, default=6,
                        help="medial-axis tie-break seeds to try (default %(default)s)")
    parser.add_argument("--no-assets", action="store_true",
                        help="only print the table; do not write assets")
    args = parser.parse_args(argv)

    images = common.sample_images()
    if args.image:
        images = [p for p in images if any(s.lower() in p.name.lower() for s in args.image)]
    if not images:
        print("no matching images in data/", file=sys.stderr)
        return 1

    cfg = RidgeCutConfig(
        passage_scale=args.passage_scale,
        passage_half_width=args.passage_half_width,
        marker_source=args.markers,
    )
    results = [
        process(path, cfg, write_assets=not args.no_assets, seed_trials=args.seed_trials)
        for path in images
    ]
    if not args.no_assets:
        common.write_json({"config": asdict(cfg), "results": results}, ASSETS / "metrics.json")

    rows = [
        {
            "render": r["short_name"],
            "wall px": r["wall_thickness_px"],
            "passage cut px": r["passage_half_width_px"],
            "passage markers": r["passage_markers"],
            "baseline": r["baseline_regions"],
            "approach": r["approach_regions"],
            "free space kept": f"{r['baseline_free_space_covered']:.3f} -> "
                               f"{r['free_space_covered']:.3f}",
            "pixels kept": f"{r['agreement_vs_baseline']['pixel_agreement']:.0%}",
            "mean IoU": f"{r['agreement_vs_baseline']['mean_iou']:.2f}",
            "regions over sweep": "/".join(
                str(s["regions"]) for s in r["sensitivity"] if s["method"] == "baseline"
            ) + " vs " + "/".join(
                str(s["regions"]) for s in r["sensitivity"] if s["method"] != "baseline"
            ),
            "cores variant": r["marker_variant"]["regions"],
            "tie-break counts": "/".join(
                str(c) for c in r["tie_break_stability"]["distinct"]
            ),
        }
        for r in results
    ]
    print(common.markdown_table(rows, list(rows[0])))
    print(f"\nassets -> {ASSETS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
