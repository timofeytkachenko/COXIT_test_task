"""End-to-end room segmentation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import PipelineConfig
from .preprocess import load_bgr, plan_mask, wall_mask
from .seeds import region_labels
from .semantics import apply_labels, request_labels

logger = logging.getLogger(__name__)

BoolMask = NDArray[np.bool_]
LabelMap = NDArray[np.int32]

_SQFT_RE = re.compile(r"(\d{2,5})\s*sq", re.IGNORECASE)


def parse_total_sqft(name: str) -> float | None:
    """Extract the advertised floor area from a file name.

    Parameters
    ----------
    name
        File name such as ``"limestone ranch_santa fe_625sq.webp"``.

    Returns
    -------
    float or None
        Area in square feet, or ``None`` when the name carries no area.

    Examples
    --------
    >>> parse_total_sqft("limestone ranch_santa fe_625sq.webp")
    625.0
    >>> parse_total_sqft("plan.webp") is None
    True
    """
    match = _SQFT_RE.search(name)
    return float(match.group(1)) if match else None


@dataclass(frozen=True)
class Room:
    """One segmented room.

    Attributes
    ----------
    id
        Label value in :attr:`Segmentation.labels`.
    name
        Room name, or ``None`` when the semantic stage was skipped.
    area_px
        Pixel count.
    area_sqft
        Area in square feet, present only when the total area is known.
    centroid
        ``(x, y)`` centroid in pixels.
    """

    id: int
    name: str | None
    area_px: int
    area_sqft: float | None
    centroid: tuple[int, int]


@dataclass
class Segmentation:
    """Result of segmenting one floor-plan render."""

    path: Path
    image: NDArray[np.uint8]
    plan: BoolMask
    wall: BoolMask
    barrier: BoolMask
    labels: LabelMap
    rooms: list[Room]
    total_sqft: float | None

    def mask_of(self, room_id: int) -> BoolMask:
        """Return the boolean mask of one room."""
        return self.labels == room_id

    @property
    def names(self) -> dict[int, str]:
        """Mapping from label to name, for annotated overlays."""
        return {r.id: r.name for r in self.rooms if r.name is not None}


def _build_rooms(
    labels: LabelMap, names: dict[int, str] | None, total_sqft: float | None
) -> list[Room]:
    """Assemble per-room records, converting pixel areas to square feet."""
    ids = [int(v) for v in np.unique(labels[labels > 0])]
    areas = {rid: int((labels == rid).sum()) for rid in ids}
    covered = sum(areas.values())
    # Rooms tile the interior, so their pixel sum maps onto the advertised area.
    sqft_per_px = (total_sqft / covered) if (total_sqft and covered) else None

    rooms: list[Room] = []
    for rid in ids:
        ys, xs = np.nonzero(labels == rid)
        rooms.append(
            Room(
                id=rid,
                name=names.get(rid) if names else None,
                area_px=areas[rid],
                area_sqft=round(areas[rid] * sqft_per_px, 1) if sqft_per_px else None,
                centroid=(int(xs.mean()), int(ys.mean())),
            )
        )
    return rooms


def segment_floorplan(
    path: str | Path, cfg: PipelineConfig | None = None
) -> Segmentation:
    """Segment the rooms of one 3D floor-plan render.

    Runs plan extraction, wall detection and a distance-transform watershed,
    then optionally names the regions and splits open-plan areas with a
    vision-language model.

    Parameters
    ----------
    path
        Path to the render.
    cfg
        Pipeline configuration; defaults are used when omitted.

    Returns
    -------
    Segmentation
        Masks, label map and per-room records.

    Raises
    ------
    FileNotFoundError
        If the image cannot be read.
    ValueError
        If no plan can be found in the image (blank page, degenerate plan
        mask) or a configuration value is out of range.
    SemanticsError
        If the semantic stage is enabled but fails.
    """
    cfg = cfg or PipelineConfig()
    path = Path(path)

    bgr = load_bgr(path)
    plan = plan_mask(bgr, cfg.preprocess)
    wall, barrier = wall_mask(bgr, plan, cfg.preprocess)
    labels, distance = region_labels(plan, barrier, cfg.seeds)
    logger.info("%s: %d geometric regions", path.name, labels.max())

    names: dict[int, str] | None = None
    if cfg.semantics.enabled:
        plan_labels = request_labels(bgr, labels, cfg.semantics)
        labels, names = apply_labels(labels, plan_labels, cfg.semantics)
        logger.info("%s: %d named rooms", path.name, labels.max())

    total_sqft = parse_total_sqft(path.name)
    return Segmentation(
        path=path,
        image=bgr,
        plan=plan,
        wall=wall,
        barrier=barrier,
        labels=labels,
        rooms=_build_rooms(labels, names, total_sqft),
        total_sqft=total_sqft,
    )
