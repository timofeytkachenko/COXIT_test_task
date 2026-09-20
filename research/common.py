"""Shared plumbing for the research approaches: baseline runs, figures, metrics.

Every approach in ``research/`` is compared against the same baseline — the
production pipeline of :mod:`floorplan_seg` — on the same images, with the same
figures and the same numbers. That comparison code lives here so that the
approach directories only hold the method itself.

Nothing in this module writes to ``output/`` or changes production behaviour;
it imports :mod:`floorplan_seg` read-only. A few private helpers of
:mod:`floorplan_seg.seeds` are reused on purpose (see
:func:`postprocess_like_baseline`): an approach that replaces one stage should
keep every other stage bit-identical, otherwise the comparison measures the
wrong thing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # allow `python research/<slug>/run.py`
    sys.path.insert(0, str(REPO_ROOT))

from floorplan_seg.pipeline import Segmentation, segment_floorplan  # noqa: E402
from floorplan_seg.seeds import (  # noqa: E402
    _absorb_small_regions,
    _merge_open_boundaries,
    _relabel_consecutive,
)
from floorplan_seg.viz import label_colours  # noqa: E402

BoolMask = NDArray[np.bool_]
LabelMap = NDArray[np.int32]
Image = NDArray[np.uint8]

DATA_DIR = REPO_ROOT / "data"

#: Asset encoding. WebP keeps a 1140x855 photoreal overlay near 150 kB, which
#: matters when every approach commits ~10 figures.
ASSET_QUALITY = 88

_FONT = cv.FONT_HERSHEY_SIMPLEX


def load_approach(slug: str) -> ModuleType:
    """Import ``research/<slug>/approach.py`` by path.

    The approach directories are named with hyphens, so they are not
    importable packages; tests and helper scripts load them through here.

    Parameters
    ----------
    slug
        Directory name under ``research/``, e.g. ``"medial-axis-doorways"``.

    Returns
    -------
    module
        The imported module.

    Raises
    ------
    FileNotFoundError
        If the approach has no ``approach.py``.
    """
    path = REPO_ROOT / "research" / slug / "approach.py"
    if not path.is_file():
        raise FileNotFoundError(f"no approach module at {path}")
    spec = importlib.util.spec_from_file_location(f"research_{slug.replace('-', '_')}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - import machinery
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    # `dataclass` looks the defining module up in sys.modules, so register it
    # before executing the module body.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample_images() -> list[Path]:
    """Return the three sample renders shipped in ``data/``, sorted by name."""
    return sorted(DATA_DIR.glob("*.webp"))


def short_name(path: Path) -> str:
    """Return a short, file-system friendly id for a render.

    Examples
    --------
    >>> short_name(Path("data/limestone ranch_santa fe_625sq.webp"))
    'limestone-ranch'
    """
    stem = path.stem.split("_")[0].split("-citadel")[0]
    return "-".join(stem.lower().split())


def baseline(path: Path) -> Segmentation:
    """Run the production pipeline (geometry only) on one render."""
    return segment_floorplan(path)


def free_space(seg: Segmentation) -> BoolMask:
    """Plan minus barrier: the pixels rooms are allowed to occupy."""
    return seg.plan & ~seg.barrier


def distance_map(free: BoolMask) -> NDArray[np.float64]:
    """Euclidean distance of every free pixel to the nearest barrier pixel."""
    return ndi.distance_transform_edt(free)


def postprocess_like_baseline(
    labels: LabelMap,
    distance: NDArray[np.float64],
    plan: BoolMask,
    merge_width: float = 16.0,
    min_area_frac: float = 0.004,
) -> LabelMap:
    """Apply the baseline's merge/absorb/renumber tail to a label map.

    Approaches that replace the *marker* stage run their output through this so
    that any difference against the baseline comes from the markers alone.

    Parameters
    ----------
    labels
        Raw region labels, 0 for background.
    distance
        Distance transform of the free space.
    plan
        Plan mask, used for the area threshold.
    merge_width
        ``SeedConfig.passage_merge_width`` equivalent.
    min_area_frac
        ``SeedConfig.min_region_area_frac`` equivalent.

    Returns
    -------
    numpy.ndarray
        Labels renumbered 1..N.
    """
    merged = _merge_open_boundaries(labels, distance, merge_width)
    absorbed = _absorb_small_regions(
        merged, distance, int(min_area_frac * float(plan.sum()))
    )
    return _relabel_consecutive(absorbed)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def _anchor(region: BoolMask) -> tuple[int, int]:
    """Most interior point of a region, so a tag is never drawn outside it."""
    dist = cv.distanceTransform(region.astype(np.uint8), cv.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return int(x), int(y)


def palette(labels: LabelMap) -> dict[int, tuple[int, int, int]]:
    """Assign one colour per label, in the production style."""
    ids = [int(v) for v in np.unique(labels[labels > 0])]
    colours = label_colours(len(ids))
    return {lbl: tuple(int(c) for c in col) for lbl, col in zip(ids, colours)}


def matched_palette(
    reference: LabelMap, labels: LabelMap
) -> dict[int, tuple[int, int, int]]:
    """Colour ``labels`` like the ``reference`` partition it is compared with.

    A region keeps the colour of the reference region it overlaps most, so the
    two overlays in a side-by-side read as "same room, same colour". When a
    reference region is split, the extra parts get a darkened variant of the
    same colour instead of an unrelated hue.
    """
    ref_colours = palette(reference)
    overlap = _overlap_matrix(reference, labels)
    used: dict[int, int] = {}
    out: dict[int, tuple[int, int, int]] = {}
    ids = [int(v) for v in np.unique(labels[labels > 0])]
    fallback = label_colours(max(len(ids), 1), seed=7)
    for i, lbl in enumerate(ids):
        column = overlap[:, i]
        if column.sum() == 0:
            out[lbl] = tuple(int(c) for c in fallback[i])
            continue
        ref = int(np.argmax(column))
        base = ref_colours.get(ref, tuple(int(c) for c in fallback[i]))
        rank = used.get(ref, 0)
        used[ref] = rank + 1
        scale = 1.0 if rank == 0 else max(0.45, 1.0 - 0.3 * rank)
        out[lbl] = tuple(int(min(255, c * scale)) for c in base)
    return out


def overlay(
    image: Image,
    labels: LabelMap,
    colours: dict[int, tuple[int, int, int]] | None = None,
    tags: dict[int, str] | None = None,
    alpha: float = 0.4,
    font_scale: float = 0.55,
) -> Image:
    """Blend a label map over a render and tag each region.

    Parameters
    ----------
    image
        Base render in BGR.
    labels
        Region label map.
    colours
        Optional label to BGR colour mapping; defaults to :func:`palette`.
    tags
        Optional label to text mapping; defaults to the relative area.
    alpha
        Opacity of the tint.
    font_scale
        OpenCV font scale for the tags.

    Returns
    -------
    numpy.ndarray
        Annotated copy of ``image``.
    """
    colours = colours or palette(labels)
    ids = [int(v) for v in np.unique(labels[labels > 0])]
    if tags is None:
        total = float((labels > 0).sum())
        tags = {
            lbl: f"#{lbl} {100 * (labels == lbl).sum() / total:.0f}%" for lbl in ids
        }

    tint = np.zeros_like(image)
    for lbl in ids:
        tint[labels == lbl] = colours[lbl]
    out = image.copy()
    mask = labels > 0
    out[mask] = cv.addWeighted(image, 1 - alpha, tint, alpha, 0)[mask]

    for lbl in ids:
        region = labels == lbl
        contours, _ = cv.findContours(
            region.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        cv.drawContours(out, contours, -1, (255, 255, 255), 2, cv.LINE_AA)
        cv.drawContours(out, contours, -1, colours[lbl], 1, cv.LINE_AA)

    thickness = max(1, round(font_scale * 2))
    for lbl in ids:
        text = tags.get(lbl, str(lbl))
        if not text:
            continue
        cx, cy = _anchor(labels == lbl)
        (tw, th), base = cv.getTextSize(text, _FONT, font_scale, thickness)
        x0, y0 = cx - tw // 2, cy - th // 2
        cv.rectangle(out, (x0 - 5, y0 - 5), (x0 + tw + 5, y0 + th + base + 2),
                     (255, 255, 255), cv.FILLED)
        cv.rectangle(out, (x0 - 5, y0 - 5), (x0 + tw + 5, y0 + th + base + 2),
                     (0, 0, 0), 1)
        cv.putText(out, text, (x0, y0 + th), _FONT, font_scale, (0, 0, 0),
                   thickness, cv.LINE_AA)
    return out


def boundaries(labels: LabelMap, both_sides: bool = True) -> BoolMask:
    """Pixels that sit on a border between two different labels.

    With ``both_sides`` the mask covers the pixel pair on either side of the
    border, which is what looks right when drawing. For *measuring* a cut,
    pass ``False``: the mask then holds one pixel per border crossing, so its
    size is a length rather than twice a length.
    """
    lab = labels.astype(np.int32)
    edge = np.zeros(lab.shape, bool)
    for dy, dx in ((1, 0), (0, 1)):
        a = lab[: lab.shape[0] - dy, : lab.shape[1] - dx]
        b = lab[dy:, dx:]
        diff = (a > 0) & (b > 0) & (a != b)
        edge[: lab.shape[0] - dy, : lab.shape[1] - dx] |= diff
        if both_sides:
            edge[dy:, dx:] |= diff
    return edge


def cut_ratio(labels: LabelMap, region: BoolMask) -> float:
    """Length of the cuts inside ``region``, in units of ``sqrt(area)``.

    A straight cut across a space scores about 1: its length is roughly the
    width of the space. A cut that wraps around a piece of furniture scores 2
    and up, because it has to travel around the object and back. This is the
    cheapest reliable way found here to tell "two floor finishes side by side"
    from "one floor and a bed on top of it".
    """
    edge = boundaries(labels, both_sides=False) & region
    return float(edge.sum()) / float(np.sqrt(max(region.sum(), 1)))


def changed_pixels(reference: LabelMap, labels: LabelMap) -> BoolMask:
    """Pixels that leave their reference region.

    Each reference region is matched to the new region holding most of it;
    everything else it used to own counts as changed. Read in this direction a
    split shows up as the smaller zone changing hands, which is what one wants
    to see, whereas matching in the other direction would call a refinement a
    perfect agreement.
    """
    overlap = _overlap_matrix(reference, labels)
    ids = [int(v) for v in np.unique(labels[labels > 0])]
    changed = (reference > 0) != (labels > 0)
    for ref in [int(v) for v in np.unique(reference[reference > 0])]:
        if not ids:
            break
        keeper = ids[int(np.argmax(overlap[ref, :]))]
        changed |= (reference == ref) & (labels != keeper)
    return changed


def difference_view(
    image: Image,
    reference: LabelMap,
    labels: LabelMap,
    ref_colour: tuple[int, int, int] = (210, 120, 20),
    new_colour: tuple[int, int, int] = (40, 40, 230),
) -> Image:
    """Show where two partitions disagree, over a faded render.

    Pixels that change hands (see :func:`changed_pixels`) are tinted red, the
    reference boundaries are drawn thick in blue and the new ones thin in red
    on top, so a boundary that both partitions agree on reads as a blue line
    with a red core.
    """
    out = cv.addWeighted(image, 0.35, np.full_like(image, 255), 0.65, 0)
    changed = changed_pixels(reference, labels)
    tint = np.zeros_like(out)
    tint[:] = (60, 60, 255)
    out[changed] = cv.addWeighted(out, 0.6, tint, 0.4, 0)[changed]
    out[_thicken(boundaries(reference), 5)] = ref_colour
    out[_thicken(boundaries(labels), 2)] = new_colour
    return out


def _thicken(mask: BoolMask, size: int = 3) -> BoolMask:
    return cv.dilate(mask.astype(np.uint8), np.ones((size, size), np.uint8)).astype(bool)


def caption(image: Image, text: str, height: int = 40) -> Image:
    """Add a white caption strip with ``text`` above the image."""
    w = image.shape[1]
    strip = np.full((height, w, 3), 255, np.uint8)
    scale = max(0.5, min(0.9, w / 1400))
    (tw, th), _ = cv.getTextSize(text, _FONT, scale, 2)
    cv.putText(strip, text, ((w - tw) // 2, (height + th) // 2), _FONT, scale,
               (20, 20, 20), 2, cv.LINE_AA)
    cv.line(strip, (0, height - 1), (w, height - 1), (210, 210, 210), 1)
    return np.vstack([strip, image])


def row(images: Sequence[Image], titles: Sequence[str] | None = None,
        gap: int = 14) -> Image:
    """Compose images side by side, padded to a common height."""
    if titles is not None:
        images = [caption(im, t) for im, t in zip(images, titles)]
    height = max(im.shape[0] for im in images)
    padded = []
    for im in images:
        pad = height - im.shape[0]
        padded.append(cv.copyMakeBorder(im, 0, pad, 0, 0, cv.BORDER_CONSTANT,
                                        value=(255, 255, 255)))
    spacer = np.full((height, gap, 3), 255, np.uint8)
    out: list[Image] = []
    for i, im in enumerate(padded):
        if i:
            out.append(spacer)
        out.append(im)
    return np.hstack(out)


def crop_to(mask: BoolMask, *images: Image, pad: int = 10) -> list[Image]:
    """Crop every image to the bounding box of ``mask`` plus a margin."""
    ys, xs = np.nonzero(mask)
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)
    return [im[y0:y1, x0:x1].copy() for im in images]


def save(image: Image, path: Path) -> Path:
    """Write an asset as WebP, creating the directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(path), image, [cv.IMWRITE_WEBP_QUALITY, ASSET_QUALITY]):
        raise OSError(f"could not write asset: {path}")
    return path


def write_json(payload: Any, path: Path) -> Path:
    """Write metrics as indented UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def _overlap_matrix(a: LabelMap, b: LabelMap) -> NDArray[np.int64]:
    """``(max(a)+1, n_b)`` pixel-overlap counts between two label maps."""
    ids_b = [int(v) for v in np.unique(b[b > 0])]
    out = np.zeros((int(a.max()) + 1, len(ids_b)), np.int64)
    for i, lbl in enumerate(ids_b):
        sel = b == lbl
        counts = np.bincount(a[sel].ravel(), minlength=out.shape[0])
        out[:, i] = counts
    return out


@dataclass(frozen=True)
class CutStats:
    """Geometry of the cuts a partition makes through free space.

    Attributes
    ----------
    length_px
        Number of pixels on inter-region boundaries.
    median_width_px
        Median free-space width (2x distance to the nearest barrier) along
        those boundaries. A doorway cut is narrow; a cut across an open room is
        wide.
    open_cut_frac
        Share of the boundary length whose free width exceeds
        ``doorway_width``. This is the fraction of the partition that is not
        supported by a constriction in the plan.
    """

    length_px: int
    median_width_px: float
    open_cut_frac: float


def cut_stats(
    labels: LabelMap, distance: NDArray[np.float64], doorway_width: float = 32.0
) -> CutStats:
    """Measure how well a partition's cuts are supported by the geometry."""
    edge = boundaries(labels, both_sides=False)
    if not edge.any():
        return CutStats(0, 0.0, 0.0)
    widths = 2.0 * distance[edge]
    return CutStats(
        length_px=int(edge.sum()),
        median_width_px=float(np.median(widths)),
        open_cut_frac=float((widths > doorway_width).mean()),
    )


def region_table(labels: LabelMap, plan: BoolMask) -> list[dict[str, Any]]:
    """Per-region id, pixel area, share of the room area and of the plan."""
    ids = [int(v) for v in np.unique(labels[labels > 0])]
    total = float((labels > 0).sum())
    plan_px = float(plan.sum())
    rows = [
        {
            "id": lbl,
            "area_px": int((labels == lbl).sum()),
            "relative_area": round((labels == lbl).sum() / total, 4),
            "share_of_plan": round((labels == lbl).sum() / plan_px, 4),
        }
        for lbl in ids
    ]
    return sorted(rows, key=lambda r: -r["area_px"])


def agreement(reference: LabelMap, labels: LabelMap) -> dict[str, float]:
    """How much of one partition survives in another.

    ``pixel_agreement`` is the share of reference pixels that stay together:
    for every reference region, the largest piece of it that ends up in a
    single new region. Splitting a region therefore costs agreement, which is
    the point — a plain "best match" comparison scores a refinement as a
    perfect match. ``mean_iou`` averages the best IoU per reference region,
    weighted by reference area.
    """
    overlap = _overlap_matrix(reference, labels)
    ids = [int(v) for v in np.unique(labels[labels > 0])]
    if not ids:
        return {"pixel_agreement": 0.0, "mean_iou": 0.0}

    ref_ids = [int(v) for v in np.unique(reference[reference > 0])]
    kept = sum(int(overlap[ref, :].max()) for ref in ref_ids)
    pixel_agreement = kept / float(sum(int((reference == ref).sum()) for ref in ref_ids))

    ious, weights = [], []
    for ref in ref_ids:
        ref_mask = reference == ref
        best = 0.0
        for i, lbl in enumerate(ids):
            inter = float(overlap[ref, i])
            if inter == 0:
                continue
            union = float(ref_mask.sum() + (labels == lbl).sum() - inter)
            best = max(best, inter / union)
        ious.append(best)
        weights.append(float(ref_mask.sum()))
    return {
        "pixel_agreement": round(pixel_agreement, 4),
        "mean_iou": round(float(np.average(ious, weights=weights)), 4),
    }


def markdown_table(rows: Iterable[dict[str, Any]], headers: Sequence[str]) -> str:
    """Render dictionaries as a GitHub markdown table."""
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)
