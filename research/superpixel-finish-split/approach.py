"""Split open-plan regions by floor finish, with superpixels and no network.

The production pipeline leaves kitchen + dining + living as one region because
there is no wall between them, and the only shipped way to cut it is a
vision-language model (``--semantics``). This module tries to do the same cut
offline, from the render alone: a floor finish change (tile vs plank vs carpet)
is usually where the functional boundary is, and it is visible.

The method, per open-plan region:

1. over-segment the region into SLIC superpixels;
2. describe every superpixel by an **illumination-robust** appearance vector
   (chroma, relative gradient contrast, orientation coherence, and lightness at
   reduced weight);
3. cluster the superpixels (Ward linkage) into a handful of finishes;
4. keep only clusters that look like *floor*: a large connected patch that runs
   up to the walls, which rejects rugs, sofas and counter tops;
5. grow the surviving patches into the whole region with a flat watershed
   (geodesic Voronoi from extended seeds);
6. accept the split only if every zone is large enough, wall-supported, and the
   finishes really differ — otherwise leave the region alone.

Step 4 and step 6 are what make it usable: without them the method happily cuts
a bedroom into "bed" and "floor".
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from scipy.cluster import hierarchy
from skimage.segmentation import slic, watershed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from research.common import cut_ratio  # noqa: E402

BoolMask = NDArray[np.bool_]
LabelMap = NDArray[np.int32]
Image = NDArray[np.uint8]


@dataclass(frozen=True)
class FinishSplitConfig:
    """Parameters of the floor-finish split.

    Attributes
    ----------
    min_region_frac
        Only regions holding at least this share of the plan are treated as
        candidate open-plan areas. On the three samples every genuine open-plan
        region covers 31-38 % of the footprint while the largest single room
        stops at 19 %, so 0.22 sits between the two populations. This is a
        guard fitted on three images, not a law.
    superpixel_px
        Target area of one SLIC superpixel in pixels; sets ``n_segments``.
    compactness
        SLIC compactness. Low values follow colour edges too eagerly on
        photoreal renders, high values produce a grid.
    n_finishes
        Number of appearance clusters per region. Four is enough for
        "floor A, floor B, light furniture, dark furniture" on these renders.
    lightness_weight, chroma_weight, contrast_weight, coherence_weight
        Weights of the standardised features. Lightness is deliberately damped:
        a sunlit patch of one finish is further from its own shade, in L, than
        two different finishes are from each other.
    seed_min_frac
        A finish patch must cover this share of the region to become a seed.
    min_wall_contact
        A finish patch must run along the walls over this share of its own
        outline to count as floor rather than furniture.
    max_zones
        How many finishes a region may be cut into. Two, because the third
        cluster on these renders is reliably a lighting artefact (a sunlit
        strip of the same floor) rather than a third material.
    min_zone_frac
        Zones below this share of the region are merged back.
    min_finish_gap
        Minimum distance between two zone finishes in standardised feature
        units; below it the two zones are considered the same finish.
    max_cut_ratio
        Reject the split when the cut is longer than this multiple of
        ``sqrt(region area)``: a cut that long is going around an object, not
        across a space. See :func:`research.common.cut_ratio`.
    """

    min_region_frac: float = 0.22
    superpixel_px: int = 900
    compactness: float = 12.0
    n_finishes: int = 4
    lightness_weight: float = 0.3
    chroma_weight: float = 1.0
    contrast_weight: float = 1.0
    coherence_weight: float = 0.5
    seed_min_frac: float = 0.06
    min_wall_contact: float = 0.15
    max_zones: int = 2
    min_zone_frac: float = 0.18
    min_finish_gap: float = 1.0
    max_cut_ratio: float = 1.8


@dataclass(frozen=True)
class RegionReport:
    """What the method decided for one candidate region.

    Attributes
    ----------
    region_id
        Label in the baseline map.
    region_frac
        Share of the plan the region covers.
    n_superpixels
        Superpixels the region was over-segmented into.
    seeds
        One record per finish patch considered, with its area share, wall
        contact and whether it was accepted as a seed.
    decision
        ``"split"``, or the reason the region was left alone.
    zones
        Area share of each resulting zone within the region (empty when the
        region was not split).
    finish_gap
        Feature distance between the two most distant accepted finishes.
    cut_ratio
        Length of the proposed cut in units of ``sqrt(region area)``; 0 when
        no cut was proposed at all.
    """

    region_id: int
    region_frac: float
    n_superpixels: int
    seeds: list[dict[str, Any]]
    decision: str
    zones: list[float]
    finish_gap: float
    cut_ratio: float


def pixel_features(bgr: Image) -> dict[str, NDArray[np.float32]]:
    """Per-pixel appearance channels used to describe a floor finish.

    Returns
    -------
    dict
        ``lab`` (float32 CIELAB in OpenCV 8-bit encoding), ``gradient``
        (Sobel magnitude of L) and ``coherence`` (structure-tensor anisotropy
        over a 9x9 window, 0 for isotropic texture, 1 for a single dominant
        orientation such as floor planks).
    """
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB).astype(np.float32)
    lightness = lab[..., 0]
    gx = cv.Sobel(lightness, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(lightness, cv.CV_32F, 0, 1, ksize=3)
    window = (9, 9)
    jxx = cv.boxFilter(gx * gx, cv.CV_32F, window)
    jyy = cv.boxFilter(gy * gy, cv.CV_32F, window)
    jxy = cv.boxFilter(gx * gy, cv.CV_32F, window)
    spread = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy**2)
    coherence = spread / (jxx + jyy + 1e-6)
    return {"lab": lab, "gradient": np.hypot(gx, gy), "coherence": coherence}


def superpixels(bgr: Image, region: BoolMask, cfg: FinishSplitConfig) -> LabelMap:
    """Over-segment one region into SLIC superpixels (0 outside the region)."""
    n_segments = max(8, int(region.sum() // cfg.superpixel_px))
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    sp = slic(
        rgb,
        n_segments=n_segments,
        compactness=cfg.compactness,
        sigma=1.0,
        mask=region,
        start_label=1,
        enforce_connectivity=True,
    )
    return sp.astype(np.int32)


def superpixel_features(
    channels: dict[str, NDArray[np.float32]], sp: LabelMap
) -> tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.int64]]:
    """Summarise every superpixel by one appearance vector.

    The contrast feature is the mean gradient magnitude divided by the mean
    lightness: a multiplicative change in illumination cancels, so a sunlit and
    a shaded patch of the same floor get the same number.

    Returns
    -------
    ids, features, areas
        Superpixel ids, their ``(n, 5)`` feature matrix
        ``[L, a, b, relative contrast, coherence]`` and their pixel areas.
    """
    lab, gradient, coherence = channels["lab"], channels["gradient"], channels["coherence"]
    ids = np.unique(sp[sp > 0]).astype(np.int32)
    features = np.zeros((ids.size, 5), np.float32)
    areas = np.zeros(ids.size, np.int64)
    for i, s in enumerate(ids):
        m = sp == s
        lightness = float(np.median(lab[..., 0][m]))
        features[i] = (
            lightness,
            float(np.median(lab[..., 1][m])) - 128.0,
            float(np.median(lab[..., 2][m])) - 128.0,
            100.0 * float(np.mean(gradient[m])) / (lightness + 1e-3),
            float(np.mean(coherence[m])),
        )
        areas[i] = int(m.sum())
    return ids, features, areas


def _standardise(features: NDArray[np.float32], cfg: FinishSplitConfig) -> NDArray[np.float32]:
    """Z-score the features and apply the per-channel weights."""
    weights = np.array(
        [
            cfg.lightness_weight,
            cfg.chroma_weight,
            cfg.chroma_weight,
            cfg.contrast_weight,
            cfg.coherence_weight,
        ],
        np.float32,
    )
    std = features.std(axis=0) + 1e-6
    return (features - features.mean(axis=0)) / std * weights


def cluster_finishes(
    features: NDArray[np.float32], cfg: FinishSplitConfig
) -> NDArray[np.int32]:
    """Group superpixels into ``cfg.n_finishes`` appearance clusters."""
    if features.shape[0] <= cfg.n_finishes:
        return np.arange(1, features.shape[0] + 1, dtype=np.int32)
    linkage = hierarchy.linkage(_standardise(features, cfg), method="ward")
    return hierarchy.fcluster(linkage, t=cfg.n_finishes, criterion="maxclust").astype(
        np.int32
    )


def _wall_contact(patch: BoolMask, barrier: BoolMask) -> float:
    """Share of a patch's outline that runs along the barrier."""
    patch_u8 = patch.astype(np.uint8)
    outline = patch & ~cv.erode(patch_u8, np.ones((3, 3), np.uint8)).astype(bool)
    if not outline.any():
        return 0.0
    near_wall = cv.dilate(
        barrier.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2
    ).astype(bool)
    return float((outline & near_wall).sum() / outline.sum())


def finish_seeds(
    finish_map: LabelMap,
    region: BoolMask,
    barrier: BoolMask,
    cfg: FinishSplitConfig,
) -> tuple[LabelMap, list[dict[str, Any]]]:
    """Turn appearance clusters into floor seeds.

    A cluster contributes a seed when one of its connected patches is both
    large and wall-bound. Several patches of the same cluster share a seed
    label, so a hallway and a kitchen laid with the same plank floor stay one
    zone.

    Returns
    -------
    seeds, report
        Seed label map (0 = no seed) and one record per inspected patch.
    """
    area = float(region.sum())
    opening = np.ones((5, 5), np.uint8)
    seeds = np.zeros(finish_map.shape, np.int32)
    report: list[dict[str, Any]] = []

    for finish in [int(v) for v in np.unique(finish_map[finish_map > 0])]:
        cleaned = cv.morphologyEx(
            (finish_map == finish).astype(np.uint8), cv.MORPH_OPEN, opening
        ).astype(bool)
        components, n = ndi.label(cleaned, structure=np.ones((3, 3), int))
        for comp in range(1, n + 1):
            patch = components == comp
            frac = patch.sum() / area
            if frac < cfg.seed_min_frac / 2:  # skip specks, do not report them
                continue
            contact = _wall_contact(patch, barrier)
            accepted = frac >= cfg.seed_min_frac and contact >= cfg.min_wall_contact
            if accepted:
                seeds[patch] = finish
            report.append(
                {
                    "finish": finish,
                    "area_frac": round(float(frac), 3),
                    "wall_contact": round(contact, 3),
                    "accepted": bool(accepted),
                }
            )
    return seeds, report


def _keep_largest_finishes(
    seeds: LabelMap, report: list[dict[str, Any]], cfg: FinishSplitConfig
) -> LabelMap:
    """Keep the ``cfg.max_zones`` largest accepted finishes, drop the rest."""
    accepted: dict[int, float] = {}
    for record in report:
        if record["accepted"]:
            accepted[record["finish"]] = accepted.get(record["finish"], 0.0) + record["area_frac"]
    keep = sorted(accepted, key=lambda f: -accepted[f])[: cfg.max_zones]
    return np.where(np.isin(seeds, keep), seeds, 0).astype(np.int32)


def _merge_small_zones(zones: LabelMap, region: BoolMask, cfg: FinishSplitConfig) -> LabelMap:
    """Absorb zones under ``min_zone_frac`` into their largest neighbour."""
    area = float(region.sum())
    zones = zones.copy()
    while True:
        ids = [int(v) for v in np.unique(zones[zones > 0])]
        if len(ids) <= 1:
            return zones
        sizes = {i: int((zones == i).sum()) for i in ids}
        smallest = min(sizes, key=lambda i: sizes[i])
        if sizes[smallest] / area >= cfg.min_zone_frac:
            return zones
        others = [i for i in ids if i != smallest]
        neighbourhood = cv.dilate(
            (zones == smallest).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2
        ).astype(bool)
        contact = {i: int((neighbourhood & (zones == i)).sum()) for i in others}
        target = max(contact, key=lambda i: contact[i]) if any(contact.values()) else others[0]
        zones[zones == smallest] = target


def split_region(
    bgr: Image,
    region: BoolMask,
    barrier: BoolMask,
    cfg: FinishSplitConfig,
    channels: dict[str, NDArray[np.float32]] | None = None,
) -> tuple[LabelMap, RegionReport, dict[str, NDArray[np.int32]]]:
    """Attempt to split one region into floor-finish zones.

    Parameters
    ----------
    bgr
        The render.
    region
        Mask of the region to split.
    barrier
        Wall barrier mask from the production pipeline.
    cfg
        Method parameters.
    channels
        Pre-computed output of :func:`pixel_features`, to avoid recomputing it
        per region.

    Returns
    -------
    zones, report, debug
        ``zones`` is a label map over ``region`` numbered 1..k (all ones when
        the region is left alone), ``report`` records the decision and
        ``debug`` carries the superpixel, finish and seed maps for figures.
    """
    channels = channels or pixel_features(bgr)
    sp = superpixels(bgr, region, cfg)
    ids, features, areas = superpixel_features(channels, sp)

    finishes = cluster_finishes(features, cfg)
    lookup = np.zeros(int(sp.max()) + 1, np.int32)
    lookup[ids] = finishes
    finish_map = lookup[sp].astype(np.int32)

    seeds, seed_report = finish_seeds(finish_map, region, barrier, cfg)
    seeds = _keep_largest_finishes(seeds, seed_report, cfg)
    debug = {"superpixels": sp, "finishes": finish_map, "seeds": seeds}
    unsplit = region.astype(np.int32)

    def _report(decision: str, zones: list[float], gap: float, ratio: float) -> RegionReport:
        return RegionReport(
            region_id=0,
            region_frac=0.0,
            n_superpixels=int(ids.size),
            seeds=seed_report,
            decision=decision,
            zones=zones,
            finish_gap=round(gap, 2),
            cut_ratio=round(ratio, 2),
        )

    accepted = [int(v) for v in np.unique(seeds[seeds > 0])]
    if len(accepted) < 2:
        return unsplit, _report("declined: fewer than two wall-bound finishes", [], 0.0, 0.0), debug

    standardised = _standardise(features, cfg)
    centres = {
        f: np.average(standardised[finishes == f], axis=0, weights=areas[finishes == f])
        for f in accepted
    }
    gap = max(
        float(np.linalg.norm(centres[a] - centres[b]))
        for i, a in enumerate(accepted)
        for b in accepted[i + 1 :]
    )
    if gap < cfg.min_finish_gap:
        return unsplit, _report("declined: finishes too similar", [], gap, 0.0), debug

    # Flat terrain: the watershed degenerates into geodesic Voronoi cells, so
    # the boundary lands midway between the two finishes and never leaves the
    # region. A gradient-driven terrain was tried in the production pipeline
    # and flooded through furniture outlines.
    grown = watershed(
        np.zeros(region.shape, np.float32), markers=seeds, mask=region
    ).astype(np.int32)
    zones = _merge_small_zones(grown, region, cfg)

    ids_out = [int(v) for v in np.unique(zones[zones > 0])]
    if len(ids_out) < 2:
        return unsplit, _report("declined: one zone dominated", [], gap, 0.0), debug

    ratio = cut_ratio(zones, region)
    shares = [round(float((zones == z).sum() / region.sum()), 3) for z in ids_out]
    renumbered = np.zeros_like(zones)
    for new, old in enumerate(ids_out, start=1):
        renumbered[zones == old] = new
    debug["proposed_zones"] = renumbered

    if ratio > cfg.max_cut_ratio:
        return unsplit, _report("declined: cut wraps an object", shares, gap, ratio), debug
    return renumbered, _report("split", shares, gap, ratio), debug


def split_open_plan(
    bgr: Image,
    labels: LabelMap,
    plan: BoolMask,
    barrier: BoolMask,
    cfg: FinishSplitConfig | None = None,
) -> tuple[LabelMap, list[RegionReport], dict[int, dict[str, NDArray[np.int32]]]]:
    """Run the finish split over every open-plan region of a segmentation.

    Parameters
    ----------
    bgr
        The render.
    labels
        Baseline region labels.
    plan
        Plan mask (sets the "how big is a candidate region" scale).
    barrier
        Wall barrier mask.
    cfg
        Method parameters; defaults are used when omitted.

    Returns
    -------
    labels, reports, debug
        A relabelled map (1..N, regions that were split contribute several
        labels), one report per candidate region, and the per-region debug
        maps keyed by the original region id.
    """
    cfg = cfg or FinishSplitConfig()
    channels = pixel_features(bgr)
    plan_px = float(plan.sum())

    out = np.zeros_like(labels)
    reports: list[RegionReport] = []
    debug: dict[int, dict[str, NDArray[np.int32]]] = {}
    next_label = 1

    for region_id in [int(v) for v in np.unique(labels[labels > 0])]:
        region = labels == region_id
        frac = region.sum() / plan_px
        if frac < cfg.min_region_frac:
            out[region] = next_label
            next_label += 1
            continue

        zones, report, dbg = split_region(bgr, region, barrier, cfg, channels)
        reports.append(
            replace(report, region_id=region_id, region_frac=round(float(frac), 3))
        )
        debug[region_id] = dbg
        for zone in [int(v) for v in np.unique(zones[zones > 0])]:
            out[(zones == zone) & region] = next_label
            next_label += 1
    return out, reports, debug
