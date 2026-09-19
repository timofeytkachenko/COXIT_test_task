"""Plan extraction and wall detection for 3D floor-plan renders.

The renders are near-orthographic top-down views in which walls are extruded,
so a wall shows up as a bright top face plus a shaded vertical face. Only the
top face is wall-coloured; the shaded face is recovered by dilation.
"""

from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from .config import PreprocessConfig

BoolMask = NDArray[np.bool_]


def load_bgr(path: str | Path) -> NDArray[np.uint8]:
    """Read an image as 3-channel BGR.

    Parameters
    ----------
    path
        Path to the image file.

    Returns
    -------
    numpy.ndarray
        ``(H, W, 3)`` uint8 array in BGR order.

    Raises
    ------
    FileNotFoundError
        If the file does not exist or cannot be decoded by OpenCV.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such image: {path}")
    bgr = cv.imread(str(path), cv.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"OpenCV could not decode: {path}")
    return bgr


def _fill_interior_holes(mask: BoolMask) -> BoolMask:
    """Fill background pockets that do not touch the image border."""
    inv = (~mask).astype(np.uint8)
    n, labels = cv.connectedComponents(inv, 4)
    border = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    outside = set(np.unique(border).tolist())
    filled = mask.copy()
    for lbl in range(1, n):
        if lbl not in outside:
            filled |= labels == lbl
    return filled


def plan_mask(bgr: NDArray[np.uint8], cfg: PreprocessConfig) -> BoolMask:
    """Segment the apartment render from the white page background.

    Flood-fills the background from the four corners, keeps the largest
    remaining component -- which discards the disclaimer text printed below
    the plan -- and fills interior pockets.

    Parameters
    ----------
    bgr
        Input image in BGR order.
    cfg
        Preprocessing parameters.

    Returns
    -------
    numpy.ndarray
        Boolean mask of the plan.
    """
    h, w = bgr.shape[:2]
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    scratch = gray.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    tol = cfg.bg_flood_tol
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        cv.floodFill(scratch, ff, seed, 0, loDiff=tol, upDiff=tol,
                     flags=4 | cv.FLOODFILL_FIXED_RANGE)

    fg = (~ff[1:-1, 1:-1].astype(bool)).astype(np.uint8)
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    n, labels, stats, _ = cv.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        raise ValueError("no foreground found; the image may be blank")
    largest = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
    return _fill_interior_holes(labels == largest)


def estimate_wall_lab(
    bgr: NDArray[np.uint8], plan: BoolMask, cfg: PreprocessConfig
) -> NDArray[np.float32]:
    """Estimate the wall colour from a ring just inside the plan outline.

    The outer boundary of a floor plan is an exterior wall, which makes this
    ring a reliable per-image colour prior. Estimating it per image is what
    makes the pipeline robust to renders with different palettes.

    Parameters
    ----------
    bgr
        Input image in BGR order.
    plan
        Boolean plan mask.
    cfg
        Preprocessing parameters.

    Returns
    -------
    numpy.ndarray
        Median CIELAB colour of the ring, shape ``(3,)``.

    Raises
    ------
    ValueError
        If the ring is empty, which means the plan mask is degenerate.
    """
    k = np.ones((3, 3), np.uint8)
    pm = plan.astype(np.uint8)
    inner = cv.erode(pm, k, iterations=cfg.wall_ring_inner).astype(bool)
    deeper = cv.erode(pm, k, iterations=cfg.wall_ring_outer).astype(bool)
    ring = inner & ~deeper
    if not ring.any():
        raise ValueError("empty wall ring; plan mask is too small")
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB).astype(np.float32)
    return np.median(lab[ring], axis=0).astype(np.float32)


def _drop_compact_blobs(mask: BoolMask, cfg: PreprocessConfig) -> BoolMask:
    """Keep large or elongated components; drop compact white objects."""
    n, labels, stats, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    areas = stats[1:, cv.CC_STAT_AREA]
    area_floor = cfg.wall_min_component_frac * float(areas.max())

    kept = np.zeros_like(mask)
    for lbl in range(1, n):
        if areas[lbl - 1] >= area_floor:
            kept |= labels == lbl
            continue
        pts = cv.findNonZero((labels == lbl).astype(np.uint8))
        (_, (rw, rh), _) = cv.minAreaRect(pts)
        short, long_ = sorted((rw, rh))
        if short > 0 and long_ / short >= cfg.wall_min_elongation:
            kept |= labels == lbl
    return kept


def wall_mask(
    bgr: NDArray[np.uint8], plan: BoolMask, cfg: PreprocessConfig
) -> tuple[BoolMask, BoolMask]:
    """Detect walls and the barrier mask that separates rooms.

    Parameters
    ----------
    bgr
        Input image in BGR order.
    plan
        Boolean plan mask.
    cfg
        Preprocessing parameters.

    Returns
    -------
    wall : numpy.ndarray
        Wall-coloured structure, after removal of compact white objects.
    barrier : numpy.ndarray
        ``wall`` dilated to cover the shaded vertical wall face. This is what
        room regions must not cross.
    """
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB).astype(np.float32)
    wall_lab = estimate_wall_lab(bgr, plan, cfg)
    delta = np.linalg.norm(lab - wall_lab, axis=2)

    raw = plan & (delta < cfg.wall_delta_e)
    raw = cv.morphologyEx(raw.astype(np.uint8), cv.MORPH_OPEN,
                          np.ones((3, 3), np.uint8)).astype(bool)
    wall = _drop_compact_blobs(raw, cfg)

    d = cfg.wall_dilate
    barrier = cv.dilate(wall.astype(np.uint8), np.ones((d, d), np.uint8)).astype(bool)
    return wall, barrier & plan
