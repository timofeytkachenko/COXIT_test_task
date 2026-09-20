"""Rooms from the medial axis of the free space, cut at passages.

The production pipeline seeds its watershed with an **h-maxima** transform of
the distance map: a room is a hill that rises at least ``h`` pixels above the
surrounding saddle. That works, but ``h`` is a contrast in a distance map,
which is hard to reason about, and it fires on constrictions that are not
doors, so a second pass has to merge regions whose shared boundary turned out
to be wide.

This approach replaces the marker stage with the topology of the free space:

1. take the **medial axis** of the free space, with the distance to the
   nearest wall attached to every skeleton pixel;
2. keep only the skeleton pixels that are at least ``passage_half_width``
   away from a wall. A skeleton pixel carries the radius of the largest disc
   that fits in the free space, so this deletes exactly the stretches of the
   skeleton that run through a passage narrower than one door;
3. what the skeleton falls apart into is the set of rooms. Each surviving
   piece becomes one marker, after pruning specks;
4. flood the free space from those markers with the same distance-transform
   watershed as the baseline, so the boundary still lands in the middle of the
   doorway.

The threshold is a **width in pixels**, and it is derived per image from the
measured wall thickness, so it does not have to be retuned when the render
resolution changes. Everything after the markers - the wide-boundary merge and
the small-region absorption - is the production code, unchanged, so a
difference against the baseline can only come from the markers.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from skimage.morphology import medial_axis
from skimage.segmentation import watershed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from research.common import postprocess_like_baseline  # noqa: E402

BoolMask = NDArray[np.bool_]
LabelMap = NDArray[np.int32]


@dataclass(frozen=True)
class RidgeCutConfig:
    """Parameters of the medial-axis cut.

    Attributes
    ----------
    passage_scale
        Cut the skeleton where the free space is narrower than this multiple
        of the wall thickness measured in the same image. A door opening in
        these renders is about one wall thickness wide, so 0.5 (a *half*
        width, because the skeleton carries a radius) sits on the door.
    passage_half_width
        Absolute threshold in pixels. When given it overrides
        ``passage_scale``; useful for the sensitivity sweep.
    marker_source
        ``"ridge"`` cuts the medial axis, ``"cores"`` thresholds the distance
        map directly (``distance >= half_width``, which is an erosion by a
        disc of that radius). Same threshold, same meaning, but the core
        variant needs no skeleton and is therefore free of the medial axis'
        random tie-break. They agree on all three samples at the default
        threshold; see ALGORITHM.md.
    min_ridge_px
        Skeleton fragments shorter than this are dropped instead of seeding a
        room. They appear where a passage is long and slightly bulged.
    min_core_px
        Same, for the ``"cores"`` variant, in pixels of area.
    min_orphan_frac
        A free-space component that holds no marker gets one of its own when
        it covers at least this share of the plan, so its area is not dropped;
        see :func:`_seed_unmarked_components`.
    rng_seed
        Seed for ``skimage.morphology.medial_axis``, which breaks ties between
        equidistant pixels at random. Left to the default the skeleton - and
        with it the region count - changes between runs; see the
        reproducibility note in ALGORITHM.md.
    merge_width, min_region_area_frac
        Passed to the production post-processing, matching ``SeedConfig``.
    """

    passage_scale: float = 0.5
    passage_half_width: float | None = None
    marker_source: str = "ridge"
    min_ridge_px: int = 25
    min_core_px: int = 25
    min_orphan_frac: float = 0.0005
    rng_seed: int = 0
    merge_width: float = 16.0
    min_region_area_frac: float = 0.004


def wall_thickness(wall: BoolMask) -> float:
    """Estimate wall thickness in pixels from the wall mask.

    The distance transform inside the wall network peaks at the centreline of
    the thickest walls; the 90th percentile of those radii, doubled, is a
    robust stand-in for "how thick is a wall here" that ignores the thin
    speckle the colour threshold leaves behind.
    """
    radii = cv.distanceTransform(wall.astype(np.uint8), cv.DIST_L2, 5)
    inside = radii[radii > 0]
    if inside.size == 0:
        raise ValueError("empty wall mask; cannot estimate wall thickness")
    return 2.0 * float(np.percentile(inside, 90))


def ridge_markers(
    free: BoolMask, half_width: float, min_ridge_px: int, rng_seed: int = 0
) -> tuple[LabelMap, BoolMask, BoolMask, NDArray[np.float64]]:
    """Cut the medial axis at passages and label what is left.

    Parameters
    ----------
    free
        Free-space mask (plan minus barrier).
    half_width
        Skeleton pixels closer than this to a wall are removed.
    min_ridge_px
        Minimum size of a surviving skeleton fragment.
    rng_seed
        Tie-break seed of the medial-axis transform.

    Returns
    -------
    markers, ridge, skeleton, distance
        Labelled markers, the surviving ridge, the full medial axis and the
        distance transform of ``free``.
    """
    skeleton, distance = medial_axis(free, return_distance=True, rng=rng_seed)
    ridge = skeleton & (distance >= half_width)
    labelled, n = ndi.label(ridge, structure=np.ones((3, 3), int))
    if n:
        sizes = np.bincount(labelled.ravel())
        keep = np.nonzero(sizes >= min_ridge_px)[0]
        keep = keep[keep > 0]
        ridge = np.isin(labelled, keep)
    markers, _ = ndi.label(ridge, structure=np.ones((3, 3), int))
    return markers.astype(np.int32), ridge, skeleton, distance


def core_markers(
    distance: NDArray[np.float64], half_width: float, min_core_px: int
) -> tuple[LabelMap, BoolMask]:
    """Label the parts of the free space that are wider than one passage.

    ``distance >= half_width`` is exactly the erosion of the free space by a
    disc of that radius, so this is the same cut as :func:`ridge_markers`
    without the intermediate skeleton - and without its random tie-break.
    """
    cores = distance >= half_width
    labelled, n = ndi.label(cores, structure=np.ones((3, 3), int))
    if n:
        sizes = np.bincount(labelled.ravel())
        keep = np.nonzero(sizes >= min_core_px)[0]
        cores = np.isin(labelled, keep[keep > 0])
    markers, _ = ndi.label(cores, structure=np.ones((3, 3), int))
    return markers.astype(np.int32), cores


def _seed_unmarked_components(
    free: BoolMask, markers: LabelMap, min_px: float
) -> LabelMap:
    """Give every free-space component without a marker one of its own.

    A component that is narrower than a passage everywhere - a niche, a
    shallow closet, the sliver of floor a beige palette leaves along a wall -
    holds no ridge and no core, and the watershed would leave it unlabelled,
    quietly dropping its area from the result. On the low-contrast sample that
    was 2.7 % of the plan. Those components get a marker here and the
    production small-region absorption decides afterwards whether they are
    rooms of their own or belong to a neighbour.

    Components below ``min_px`` are left out: they are antialiasing crumbs
    along the barrier, they carry 0.2-0.3 % of the free space between them,
    and each one costs a full pass of the absorption loop.
    """
    components, n = ndi.label(free, structure=np.ones((3, 3), int))
    if n == 0:
        return markers
    sizes = np.bincount(components.ravel(), minlength=n + 1)
    marked = set(np.unique(components[markers > 0]).tolist()) - {0}
    orphans = np.isin(
        components,
        [c for c in range(1, n + 1) if c not in marked and sizes[c] >= min_px],
    )
    if not orphans.any():
        return markers
    extra, _ = ndi.label(orphans, structure=np.ones((3, 3), int))
    out = markers.copy()
    out[orphans] = extra[orphans] + int(markers.max())
    return out


def region_labels(
    plan: BoolMask, barrier: BoolMask, wall: BoolMask, cfg: RidgeCutConfig | None = None
) -> tuple[LabelMap, dict[str, Any]]:
    """Segment free space into rooms by cutting the medial axis at passages.

    Drop-in alternative to :func:`floorplan_seg.seeds.region_labels`, taking
    the wall mask as well because the passage threshold is derived from it.

    Parameters
    ----------
    plan, barrier, wall
        Masks from :mod:`floorplan_seg.preprocess`.
    cfg
        Method parameters; defaults are used when omitted.

    Returns
    -------
    labels, trace
        The final label map (1..N) and a trace of the intermediate counts,
        thresholds and masks for figures and metrics. ``passage_markers``
        counts the markers the passage cut produced, ``markers`` counts them
        after every marker-less free component has been given one.
    """
    cfg = cfg or RidgeCutConfig()
    free = plan & ~barrier
    thickness = wall_thickness(wall)
    half_width = (
        cfg.passage_half_width
        if cfg.passage_half_width is not None
        else cfg.passage_scale * thickness
    )

    if cfg.marker_source not in ("ridge", "cores"):
        raise ValueError(f"marker_source must be 'ridge' or 'cores', got {cfg.marker_source!r}")

    if cfg.marker_source == "ridge":
        markers, ridge, skeleton, distance = ridge_markers(
            free, half_width, cfg.min_ridge_px, cfg.rng_seed
        )
        marker_mask = ridge
    else:
        # No skeleton at all on this path: that is where its speed and its
        # determinism come from. The medial-axis figure is then empty.
        distance = ndi.distance_transform_edt(free)
        markers, marker_mask = core_markers(distance, half_width, cfg.min_core_px)
        ridge = skeleton = np.zeros(free.shape, bool)
    passage_markers = int(markers.max())
    markers = _seed_unmarked_components(
        free, markers, cfg.min_orphan_frac * float(plan.sum())
    )
    if markers.max() == 0:  # degenerate plan: keep the free space as one room
        markers = free.astype(np.int32)

    raw = watershed(-distance, markers, mask=free).astype(np.int32)
    labels = postprocess_like_baseline(
        raw, distance, plan, cfg.merge_width, cfg.min_region_area_frac
    )
    trace = {
        "marker_source": cfg.marker_source,
        "wall_thickness_px": round(thickness, 1),
        "passage_half_width_px": round(float(half_width), 1),
        "passage_markers": passage_markers,
        "markers": int(markers.max()),
        "regions_before_postprocess": len(np.unique(raw[raw > 0])),
        "regions": int(labels.max()),
        "ridge": ridge,
        "markers_mask": marker_mask,
        "skeleton": skeleton,
        "distance": distance,
        "free": free,
    }
    return labels, trace


def passages(labels: LabelMap, distance: NDArray[np.float64]) -> list[dict[str, Any]]:
    """Locate and measure the passage each pair of neighbouring rooms shares.

    The widest point of a shared boundary is the pass the watershed came
    through; its free width is twice the distance to the nearest wall there.

    Returns
    -------
    list of dict
        ``{"rooms": (a, b), "x": int, "y": int, "width_px": float}``, sorted
        by width.
    """
    found: dict[tuple[int, int], tuple[float, int, int]] = {}
    for dy, dx in ((1, 0), (0, 1)):
        a = labels[: labels.shape[0] - dy, : labels.shape[1] - dx]
        b = labels[dy:, dx:]
        touching = (a > 0) & (b > 0) & (a != b)
        if not touching.any():
            continue
        ys, xs = np.nonzero(touching)
        widths = 2.0 * distance[ys + dy, xs + dx]
        for y, x, width in zip(ys, xs, widths):
            key = (int(min(a[y, x], b[y, x])), int(max(a[y, x], b[y, x])))
            if key not in found or width > found[key][0]:
                found[key] = (float(width), int(x), int(y))
    return sorted(
        (
            {"rooms": list(key), "x": value[1], "y": value[2], "width_px": round(value[0], 1)}
            for key, value in found.items()
        ),
        key=lambda p: p["width_px"],
    )
