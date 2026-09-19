"""Visualisation helpers."""

from __future__ import annotations

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

LabelMap = NDArray[np.int32]


def label_colours(n: int, seed: int = 0) -> NDArray[np.uint8]:
    """Return ``n`` visually distinct BGR colours."""
    rng = np.random.default_rng(seed)
    hues = (np.linspace(0, 179, num=max(n, 1), endpoint=False) +
            rng.integers(0, 20)) % 180
    hsv = np.stack([hues, np.full(max(n, 1), 200), np.full(max(n, 1), 255)], axis=1)
    hsv = hsv.astype(np.uint8).reshape(-1, 1, 3)
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR).reshape(-1, 3)


def _label_anchor(region: NDArray[np.bool_]) -> tuple[int, int]:
    """A point well inside the region, so the tag is not drawn outside it."""
    dist = cv.distanceTransform(region.astype(np.uint8), cv.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return int(x), int(y)


def overlay_labels(
    bgr: NDArray[np.uint8],
    labels: LabelMap,
    alpha: float = 0.45,
    names: dict[int, str] | None = None,
    font_scale: float = 0.8,
) -> NDArray[np.uint8]:
    """Blend a label map over the render and tag each region.

    Tags are drawn on an opaque plate at the most interior point of the
    region. Legibility matters beyond aesthetics: this overlay is what the
    vision-language model reads, and small thin digits get misread.

    Parameters
    ----------
    bgr
        Base image in BGR order.
    labels
        Region label map, 0 for background.
    alpha
        Opacity of the colour overlay.
    names
        Optional mapping from label to text; defaults to the label number.
    font_scale
        OpenCV font scale for the tags.

    Returns
    -------
    numpy.ndarray
        Annotated copy of the image.
    """
    ids = np.unique(labels[labels > 0])
    colours = label_colours(len(ids))
    tint = np.zeros_like(bgr)
    for colour, lbl in zip(colours, ids):
        tint[labels == lbl] = colour

    out = bgr.copy()
    mask = labels > 0
    out[mask] = cv.addWeighted(bgr, 1 - alpha, tint, alpha, 0)[mask]

    font, thickness = cv.FONT_HERSHEY_SIMPLEX, max(1, round(font_scale * 2))
    for lbl in ids:
        region = labels == lbl
        cx, cy = _label_anchor(region)
        text = names.get(int(lbl), str(int(lbl))) if names else str(int(lbl))
        (tw, th), base = cv.getTextSize(text, font, font_scale, thickness)
        x0, y0 = cx - tw // 2, cy - th // 2
        cv.rectangle(out, (x0 - 5, y0 - 5), (x0 + tw + 5, y0 + th + base + 2),
                     (255, 255, 255), cv.FILLED)
        cv.rectangle(out, (x0 - 5, y0 - 5), (x0 + tw + 5, y0 + th + base + 2),
                     (0, 0, 0), 1)
        cv.putText(out, text, (x0, y0 + th), font, font_scale, (0, 0, 0),
                   thickness, cv.LINE_AA)
    return out
