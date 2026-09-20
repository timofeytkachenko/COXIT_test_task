"""Turn wall-free space into candidate room regions.

A distance-transform watershed splits free space at its narrowest points,
which is where door openings are. Two corrections make this usable:

* markers come from an h-maxima transform rather than local peaks, because
  peak detection fires several times inside a single room;
* adjacent regions whose shared boundary runs through wide-open space are
  merged back, since only a narrow boundary corresponds to a real doorway.

What remains unsolved here is the open-plan case: a kitchen and a living room
with no constriction between them stay one region by construction. Splitting
those requires semantics, not geometry.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from skimage.filters import gaussian
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from .config import SeedConfig

BoolMask = NDArray[np.bool_]
LabelMap = NDArray[np.int32]

_SHIFTS = ((1, 0), (0, 1))


class _DisjointSet:
    """Minimal union-find over integer labels."""

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)


def _adjacency(
    labels: LabelMap, distance: NDArray[np.float64]
) -> dict[tuple[int, int], tuple[int, float]]:
    """Return ``{(a, b): (contact_px, widest_free_space_on_boundary)}``."""
    out: dict[tuple[int, int], tuple[int, float]] = {}
    for dy, dx in _SHIFTS:
        a = labels[: labels.shape[0] - dy, : labels.shape[1] - dx]
        b = labels[dy:, dx:]
        d = distance[dy:, dx:]
        touching = (a > 0) & (b > 0) & (a != b)
        if not touching.any():
            continue
        lo = np.minimum(a[touching], b[touching])
        hi = np.maximum(a[touching], b[touching])
        widths = d[touching]
        for pair in np.unique(np.stack([lo, hi], axis=1), axis=0):
            key = (int(pair[0]), int(pair[1]))
            sel = (lo == pair[0]) & (hi == pair[1])
            count, width = int(sel.sum()), float(widths[sel].max())
            prev = out.get(key)
            if prev is None:
                out[key] = (count, width)
            else:
                out[key] = (prev[0] + count, max(prev[1], width))
    return out


def _relabel_consecutive(labels: LabelMap) -> LabelMap:
    """Renumber labels to 1..N, keeping background at 0."""
    out = np.zeros_like(labels)
    for new, old in enumerate(np.unique(labels[labels > 0]), start=1):
        out[labels == old] = new
    return out


def _merge_open_boundaries(
    labels: LabelMap, distance: NDArray[np.float64], min_width: float
) -> LabelMap:
    """Merge neighbours separated by a wide boundary rather than a doorway."""
    ds = _DisjointSet()
    for lbl in np.unique(labels[labels > 0]):
        ds.find(int(lbl))
    for (a, b), (_, width) in _adjacency(labels, distance).items():
        if width > min_width:
            ds.union(a, b)

    out = np.zeros_like(labels)
    for lbl in np.unique(labels[labels > 0]):
        out[labels == lbl] = ds.find(int(lbl))
    return out


def _nearest_label(labels: LabelMap, islet: int, candidates: NDArray[np.int64]) -> int:
    """Label among ``candidates`` closest (Euclidean) to the islet's pixels."""
    targets = np.isin(labels, candidates)
    _, (iy, ix) = ndi.distance_transform_edt(~targets, return_indices=True)
    member = labels == islet
    nearest = labels[iy[member], ix[member]]
    return int(np.bincount(nearest).argmax())


def _absorb_small_regions(
    labels: LabelMap, distance: NDArray[np.float64], min_area: int
) -> LabelMap:
    """Absorb undersized regions into the neighbour they touch most.

    An islet that touches nothing (free space enclosed by barrier on every
    side) is attached to the nearest full-sized region instead of being
    discarded, so that the regions keep tiling the free space and no area is
    lost. Full-sized regions are preferred so that an islet cannot rescue
    another undersized scrap from absorption.
    """
    labels = labels.copy()
    while True:
        present, counts = np.unique(labels[labels > 0], return_counts=True)
        if present.size <= 1 or counts.min() >= min_area:
            return labels
        smallest = int(present[int(np.argmin(counts))])

        adjacency = _adjacency(labels, distance)
        neighbours = {
            (b if a == smallest else a): contact
            for (a, b), (contact, _) in adjacency.items()
            if smallest in (a, b)
        }
        if neighbours:
            target = max(neighbours, key=lambda k: neighbours[k])
        else:
            others = present[present != smallest]
            full_sized = others[counts[present != smallest] >= min_area]
            target = _nearest_label(labels, smallest, full_sized if full_sized.size else others)
        labels[labels == smallest] = target


def region_labels(
    plan: BoolMask, barrier: BoolMask, cfg: SeedConfig
) -> tuple[LabelMap, NDArray[np.float64]]:
    """Split free space into candidate rooms.

    Parameters
    ----------
    plan
        Boolean plan mask.
    barrier
        Boolean wall barrier mask.
    cfg
        Seeding parameters.

    Returns
    -------
    labels : numpy.ndarray
        ``(H, W)`` int32 label map numbered 1..N, 0 for background.
    distance
        Euclidean distance transform of the free space. Returned for
        inspection (debug figures, notebooks); the pipeline itself does not
        consume it.
    """
    free = plan & ~barrier
    distance = ndi.distance_transform_edt(free)
    smoothed = (
        gaussian(distance, sigma=cfg.smooth_sigma, preserve_range=True)
        if cfg.smooth_sigma > 0
        else distance
    )

    with warnings.catch_warnings():
        # scikit-image 0.26 still assigns ``arr.shape = ...`` in its
        # reconstruction code, which NumPy 2.5 deprecates; nothing to fix here.
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated",
            category=DeprecationWarning,
            module=r"skimage\.morphology\.grayreconstruct",
        )
        peaks = h_maxima(smoothed * free, cfg.h_maxima)
    markers, n_markers = ndi.label(peaks * free)
    if n_markers == 0:
        markers = free.astype(np.int32)

    labels = watershed(-smoothed, markers, mask=free).astype(np.int32)
    labels = _merge_open_boundaries(labels, distance, cfg.passage_merge_width)
    min_area = int(cfg.min_region_area_frac * float(plan.sum()))
    labels = _absorb_small_regions(labels, distance, min_area)
    return _relabel_consecutive(labels), distance
