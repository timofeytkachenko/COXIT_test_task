"""Polygon extraction, JSON serialisation and the annotated deliverable image."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from .pipeline import Segmentation
from .viz import label_colours, overlay_labels

Polygon = list[list[int]]

#: Douglas-Peucker tolerance as a fraction of the contour perimeter.
DEFAULT_SIMPLIFY = 0.004


def room_polygon(mask: NDArray[np.bool_], simplify: float = DEFAULT_SIMPLIFY) -> Polygon:
    """Approximate the outer boundary of a room mask with a polygon.

    Only the largest external contour is kept, so holes left by furniture or
    white fixtures inside the room are ignored on purpose: the deliverable is
    the room outline, not its free floor.

    Parameters
    ----------
    mask
        Boolean mask of one room.
    simplify
        Douglas-Peucker tolerance as a fraction of the contour perimeter.
        Zero keeps every contour vertex.

    Returns
    -------
    list of [x, y]
        Polygon vertices in pixel coordinates, or an empty list if the mask
        is empty.
    """
    contours, _ = cv.findContours(
        mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    contour = max(contours, key=cv.contourArea)
    if simplify > 0:
        eps = simplify * cv.arcLength(contour, closed=True)
        contour = cv.approxPolyDP(contour, eps, closed=True)
    return contour.reshape(-1, 2).astype(int).tolist()


def to_record(seg: Segmentation, simplify: float = DEFAULT_SIMPLIFY) -> dict[str, Any]:
    """Serialise a segmentation into a JSON-ready dictionary.

    Relative areas are pixel based: ``relative_area`` is the room's share of
    the summed room area, ``share_of_plan`` its share of the whole plan
    footprint including walls. Square-foot figures appear only when the file
    name carries the advertised total area.

    Parameters
    ----------
    seg
        Pipeline output.
    simplify
        Polygon simplification tolerance, see :func:`room_polygon`.

    Returns
    -------
    dict
        Record with image metadata and one entry per room.
    """
    h, w = seg.labels.shape
    plan_px = int(seg.plan.sum())
    rooms_px = sum(r.area_px for r in seg.rooms)

    rooms = []
    for room in sorted(seg.rooms, key=lambda r: -r.area_px):
        rooms.append(
            {
                "id": room.id,
                "name": room.name,
                "area_px": room.area_px,
                "relative_area": round(room.area_px / rooms_px, 4) if rooms_px else 0.0,
                "share_of_plan": round(room.area_px / plan_px, 4) if plan_px else 0.0,
                "area_sqft_estimate": room.area_sqft,
                "centroid": list(room.centroid),
                "polygon": room_polygon(seg.mask_of(room.id), simplify),
            }
        )

    return {
        "image": seg.path.name,
        "image_size": {"width": w, "height": h},
        "plan_area_px": plan_px,
        "rooms_area_px": rooms_px,
        "advertised_total_sqft": seg.total_sqft,
        "units": {
            "polygon": "pixel coordinates, origin top-left, x right, y down",
            "relative_area": "area_px / rooms_area_px",
            "share_of_plan": "area_px / plan_area_px",
        },
        "rooms": rooms,
    }


def write_json(record: dict[str, Any], path: str | Path) -> None:
    """Write a record to disk as indented UTF-8 JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)


def annotated_image(seg: Segmentation, record: dict[str, Any]) -> NDArray[np.uint8]:
    """Render the deliverable: tinted rooms, polygon outlines, name and share.

    Parameters
    ----------
    seg
        Pipeline output.
    record
        Output of :func:`to_record`, whose polygons are drawn.

    Returns
    -------
    numpy.ndarray
        Annotated BGR image.
    """
    tags = {
        r["id"]: f"{r['name'] or '#' + str(r['id'])} {100 * r['relative_area']:.0f}%"
        for r in record["rooms"]
    }
    out = overlay_labels(seg.image, seg.labels, alpha=0.35, names=tags, font_scale=0.55)

    ids = np.unique(seg.labels[seg.labels > 0])
    colours = {int(lbl): tuple(int(c) for c in col)
               for lbl, col in zip(ids, label_colours(len(ids)))}
    for r in record["rooms"]:
        if len(r["polygon"]) < 3:
            continue
        pts = np.asarray(r["polygon"], dtype=np.int32).reshape(-1, 1, 2)
        cv.polylines(out, [pts], isClosed=True, color=(255, 255, 255), thickness=3, lineType=cv.LINE_AA)
        cv.polylines(out, [pts], isClosed=True, color=colours[r["id"]], thickness=1, lineType=cv.LINE_AA)
    return out
