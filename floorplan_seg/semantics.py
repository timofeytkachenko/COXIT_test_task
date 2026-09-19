"""Vision-language labelling of geometric regions.

Uses set-of-mark prompting: the model receives the clean render together with
a numbered overlay of the candidate regions, names each number, and reports
the functional zones of any region that covers more than one (an open-plan
kitchen/dining/living area, typically). Those zones are then cut apart into
geodesic Voronoi cells around the reported anchors.

Driving that watershed with the image gradient was tried first, on the idea
that the cut would follow the change in floor finish. It collapses: furniture
outlines form closed ridges, one basin breaks out and floods the region. On
the sample render it gave zones of 730 and 92596 px where the Voronoi split
gave a balanced 27578 and 34959.
"""

from __future__ import annotations

import base64
import logging
import os

import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from skimage.segmentation import watershed

from .config import SemanticsConfig
from .viz import overlay_labels

logger = logging.getLogger(__name__)

LabelMap = NDArray[np.int32]

_ROOM_VOCABULARY = (
    "living room, dining room, kitchen, bedroom, bathroom, half bath, "
    "hallway, entry, closet, walk-in closet, pantry, laundry, mechanical, "
    "balcony, patio, storage, stairs"
)

_SYSTEM_PROMPT = (
    "You analyse 3D isometric apartment floor-plan renders. You are precise "
    "and you never invent rooms that are not visible in the image."
)

_USER_PROMPT = """The first image is a 3D floor-plan render. The second image is the same \
render with candidate regions tinted, each carrying a number on a white plate.

The regions are numbered {ids}. Report each of these ids exactly once, \
including small ones such as closets, balconies and mechanical cupboards.

For every region give a short room name. Prefer these names: {vocab}.

Work out each region's identity from the fixtures you can see inside its tint: \
a hob and a sink mean kitchen, a bed means bedroom, a toilet or a bathtub mean \
bathroom, hanging rails mean closet, an outdoor railing means balcony. Check \
the number plate carefully before you answer, because a region's tint can \
extend a long way from its plate.

Some regions are open plan and cover several functional zones with no wall \
between them, typically a kitchen, a dining area and a living room sharing one \
space. That is common for the single largest region. Whenever a region holds \
more than one zone, list every zone in `zones` with the pixel coordinates of a \
point clearly inside it, and keep `name` as a summary such as "open plan". Use \
the coordinate system of the images you were given: origin top-left, x to the \
right, y down, image size {width} by {height} pixels.

Leave `zones` empty only for a region that really is a single room."""


class SemanticsError(RuntimeError):
    """Raised when the labelling backend cannot produce a usable answer."""


class Zone(BaseModel):
    """A functional zone inside an open-plan region."""

    name: str = Field(description="Short room name for this zone.")
    x: int = Field(description="Pixel x of a point inside the zone.")
    y: int = Field(description="Pixel y of a point inside the zone.")


class RegionLabel(BaseModel):
    """A named region, optionally subdivided into zones."""

    id: int = Field(description="The number printed on the region.")
    name: str = Field(description="Short room name for the whole region.")
    zones: list[Zone] = Field(
        default_factory=list,
        description="Functional zones, only when the region holds more than one.",
    )


class PlanLabels(BaseModel):
    """Labelling of every numbered region in one plan."""

    regions: list[RegionLabel]


def _encode_png(bgr: NDArray[np.uint8], max_side: int) -> tuple[str, float]:
    """Return a base64 PNG and the scale factor applied to the image."""
    h, w = bgr.shape[:2]
    if h == 0 or w == 0:
        raise SemanticsError(f"image has invalid dimensions: {w}x{h}")
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        bgr = cv.resize(bgr, (round(w * scale), round(h * scale)),
                        interpolation=cv.INTER_AREA)
    ok, buf = cv.imencode(".png", bgr)
    if not ok:
        raise SemanticsError("failed to PNG-encode the image")
    return base64.b64encode(buf.tobytes()).decode("ascii"), scale


def request_labels(
    bgr: NDArray[np.uint8], labels: LabelMap, cfg: SemanticsConfig
) -> PlanLabels:
    """Ask the vision-language model to name each region.

    Parameters
    ----------
    bgr
        The render in BGR order.
    labels
        Region label map.
    cfg
        Semantics parameters.

    Returns
    -------
    PlanLabels
        Region names, with zone anchors rescaled to full-resolution pixels.

    Raises
    ------
    SemanticsError
        If the API key is absent, the request fails, or the model refuses.
    """
    from openai import OpenAI, OpenAIError

    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise SemanticsError(
            f"environment variable {cfg.api_key_env} is not set; "
            "set it or disable the semantics stage"
        )

    plain_b64, scale = _encode_png(bgr, cfg.max_image_side)
    marked_b64, _ = _encode_png(overlay_labels(bgr, labels), cfg.max_image_side)
    h, w = bgr.shape[:2]
    region_ids = [int(v) for v in np.unique(labels[labels > 0])]
    prompt = _USER_PROMPT.format(
        ids=", ".join(str(i) for i in region_ids),
        vocab=_ROOM_VOCABULARY,
        width=round(w * scale),
        height=round(h * scale),
    )

    client = OpenAI(api_key=api_key, timeout=cfg.request_timeout_s)
    try:
        completion = client.chat.completions.parse(
            model=cfg.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{plain_b64}"}},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{marked_b64}"}},
                    ],
                },
            ],
            response_format=PlanLabels,
            temperature=cfg.temperature,
        )
    except OpenAIError as exc:
        raise SemanticsError(f"vision-language request failed: {exc}") from exc

    message = completion.choices[0].message
    if message.refusal:
        raise SemanticsError(f"model refused the request: {message.refusal}")
    if message.parsed is None:
        raise SemanticsError("model returned no parsable content")

    parsed = message.parsed
    if scale < 1.0:
        for region in parsed.regions:
            for zone in region.zones:
                zone.x = round(zone.x / scale)
                zone.y = round(zone.y / scale)
    return parsed


def _snap_into_region(
    region: NDArray[np.bool_], x: int, y: int
) -> tuple[int, int] | None:
    """Move an anchor point to the nearest pixel inside ``region``."""
    h, w = region.shape
    if 0 <= y < h and 0 <= x < w and region[y, x]:
        return x, y
    ys, xs = np.nonzero(region)
    if ys.size == 0:
        return None
    idx = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
    return int(xs[idx]), int(ys[idx])


def split_region_by_zones(
    region: NDArray[np.bool_],
    zones: list[Zone],
) -> tuple[LabelMap, list[str]]:
    """Cut one region into geodesic Voronoi cells around the zone anchors.

    Parameters
    ----------
    region
        Boolean mask of the region to subdivide.
    zones
        Named anchor points, at least two.

    Returns
    -------
    sub : numpy.ndarray
        Label map over ``region``, numbered 1..k, 0 elsewhere.
    names : list of str
        Zone name for each label, in order.
    """
    markers = np.zeros(region.shape, np.int32)
    names: list[str] = []
    for zone in zones:
        snapped = _snap_into_region(region, zone.x, zone.y)
        if snapped is None:
            continue
        x, y = snapped
        if markers[y, x]:
            continue
        names.append(zone.name)
        markers[y, x] = len(names)

    if len(names) < 2:
        return np.where(region, 1, 0).astype(np.int32), names[:1] or ["room"]

    flat = np.zeros(region.shape, np.float32)
    sub = watershed(flat, markers, mask=region).astype(np.int32)
    return sub, names


def apply_labels(
    labels: LabelMap,
    plan_labels: PlanLabels,
    cfg: SemanticsConfig,
) -> tuple[LabelMap, dict[int, str]]:
    """Attach names to regions and split the open-plan ones.

    Parameters
    ----------
    labels
        Region label map from the geometric stage.
    plan_labels
        Model output.
    cfg
        Semantics parameters.

    Returns
    -------
    new_labels : numpy.ndarray
        Label map after any open-plan subdivision.
    names : dict
        Mapping from label to room name.
    """
    by_id = {region.id: region for region in plan_labels.regions}
    present = [int(v) for v in np.unique(labels[labels > 0])]
    missing = sorted(set(present) - set(by_id))
    if missing:
        logger.warning("model did not label regions %s", missing)

    new_labels = np.zeros_like(labels)
    names: dict[int, str] = {}
    next_id = 1

    for rid in present:
        region = labels == rid
        info = by_id.get(rid)
        zones = info.zones if (info and cfg.split_open_plan) else []

        if len(zones) >= 2:
            sub, zone_names = split_region_by_zones(region, zones)
            for local, name in enumerate(zone_names, start=1):
                piece = sub == local
                if not piece.any():
                    continue
                new_labels[piece] = next_id
                names[next_id] = name
                next_id += 1
            leftover = region & (new_labels == 0)
            if leftover.any() and next_id > 1:
                new_labels[leftover] = next_id - 1
        else:
            new_labels[region] = next_id
            names[next_id] = info.name if info else f"region {rid}"
            next_id += 1

    return new_labels, names
