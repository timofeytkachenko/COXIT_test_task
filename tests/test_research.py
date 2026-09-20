"""Offline tests for the research approaches in ``research/``.

These are deliberately thin: the research code is exploratory, but it imports
private helpers of :mod:`floorplan_seg.seeds`, so a rename upstream should
fail here rather than silently rot.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from floorplan_seg.config import PreprocessConfig
from floorplan_seg.preprocess import load_bgr, plan_mask, wall_mask
from research import common

RIDGE = common.load_approach("medial-axis-doorways")
FINISH = common.load_approach("superpixel-finish-split")


@pytest.fixture(scope="module")
def two_room_masks(two_room_path: Path):
    """Plan, wall and barrier masks of the synthetic two-room render."""
    cfg = PreprocessConfig()
    bgr = load_bgr(two_room_path)
    plan = plan_mask(bgr, cfg)
    wall, barrier = wall_mask(bgr, plan, cfg)
    return bgr, plan, wall, barrier


def test_ridge_cut_separates_rooms_at_the_doorway(two_room_masks) -> None:
    """A threshold above the 15 px doorway radius cuts the two rooms apart."""
    _, plan, wall, barrier = two_room_masks
    labels, trace = RIDGE.region_labels(
        plan, barrier, wall, RIDGE.RidgeCutConfig(passage_half_width=18.0)
    )
    assert labels.max() == 2
    assert trace["passage_half_width_px"] == 18.0


def test_default_threshold_is_tied_to_wall_thickness(two_room_masks) -> None:
    """Characterises a known limitation, see the approach's ALGORITHM.md.

    The synthetic render has 12 px walls and a 30 px doorway, so half the wall
    thickness is well below the doorway radius and the opening is not treated
    as a passage: the two rooms stay merged.
    """
    _, plan, wall, barrier = two_room_masks
    labels, trace = RIDGE.region_labels(plan, barrier, wall)
    assert trace["passage_half_width_px"] == pytest.approx(6.0, abs=1.0)
    assert labels.max() == 1


def test_marker_variants_agree(two_room_masks) -> None:
    """The skeleton cut and the distance threshold are the same criterion."""
    _, plan, wall, barrier = two_room_masks
    config = RIDGE.RidgeCutConfig(passage_half_width=18.0)
    ridge, _ = RIDGE.region_labels(plan, barrier, wall, config)
    cores, _ = RIDGE.region_labels(
        plan, barrier, wall, replace(config, marker_source="cores")
    )
    assert ridge.max() == cores.max() == 2
    assert common.agreement(ridge, cores)["mean_iou"] > 0.95


def test_unknown_marker_source_is_rejected(two_room_masks) -> None:
    _, plan, wall, barrier = two_room_masks
    with pytest.raises(ValueError, match="marker_source"):
        RIDGE.region_labels(plan, barrier, wall, RIDGE.RidgeCutConfig(marker_source="magic"))


def test_passages_measure_the_doorway(two_room_masks) -> None:
    """The synthetic render has one 30 px doorway between its two rooms."""
    _, plan, wall, barrier = two_room_masks
    labels, trace = RIDGE.region_labels(
        plan, barrier, wall, RIDGE.RidgeCutConfig(passage_half_width=18.0)
    )
    found = RIDGE.passages(labels, trace["distance"])
    assert len(found) == 1
    assert 10.0 < found[0]["width_px"] < 40.0


def test_finish_split_declines_on_a_uniform_floor(two_room_masks) -> None:
    """Two rooms, one flat floor colour: there is no finish boundary to find."""
    bgr, plan, _, barrier = two_room_masks
    labels, reports, _ = FINISH.split_open_plan(bgr, plan.astype(np.int32), plan, barrier)
    assert labels.max() == 1
    assert all(r.decision.startswith("declined") for r in reports)


def test_cut_ratio_penalises_a_cut_that_wraps_an_object() -> None:
    """A straight cut across a square scores below a cut around a blob."""
    region = np.zeros((100, 100), bool)
    region[10:90, 10:90] = True

    straight = np.zeros(region.shape, np.int32)
    straight[region] = 1
    straight[50:90, 10:90] = 2

    wrapping = np.zeros(region.shape, np.int32)
    wrapping[region] = 1
    wrapping[30:70, 30:70] = 2

    assert common.cut_ratio(straight, region) < common.cut_ratio(wrapping, region)


def test_postprocess_absorbs_a_sliver() -> None:
    """The reused production tail must still merge undersized regions."""
    plan = np.zeros((60, 60), bool)
    plan[10:50, 10:50] = True
    labels = np.zeros(plan.shape, np.int32)
    labels[plan] = 1
    labels[10:12, 10:12] = 2
    distance = common.distance_map(plan)
    out = common.postprocess_like_baseline(labels, distance, plan)
    assert out.max() == 1
