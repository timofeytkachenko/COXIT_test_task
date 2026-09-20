"""Tests for the offline half of the semantic stage (no API calls)."""

from __future__ import annotations

import numpy as np

from floorplan_seg.config import SemanticsConfig
from floorplan_seg.semantics import PlanLabels, RegionLabel, Zone, apply_labels


def _two_regions() -> np.ndarray:
    labels = np.zeros((50, 100), np.int32)
    labels[5:45, 5:45] = 1
    labels[5:45, 55:95] = 2
    return labels


def test_apply_labels_names_regions() -> None:
    plan_labels = PlanLabels(
        regions=[RegionLabel(id=1, name="bedroom"), RegionLabel(id=2, name="bathroom")]
    )
    new_labels, names = apply_labels(_two_regions(), plan_labels, SemanticsConfig())
    assert names == {1: "bedroom", 2: "bathroom"}
    assert np.array_equal(new_labels, _two_regions())


def test_apply_labels_splits_open_plan_into_voronoi_cells() -> None:
    labels = _two_regions()
    plan_labels = PlanLabels(
        regions=[
            RegionLabel(
                id=1,
                name="open plan",
                zones=[Zone(name="kitchen", x=12, y=25), Zone(name="living", x=37, y=25)],
            ),
            RegionLabel(id=2, name="bedroom"),
        ]
    )
    new_labels, names = apply_labels(labels, plan_labels, SemanticsConfig())
    assert names == {1: "kitchen", 2: "living", 3: "bedroom"}
    # The split covers the whole original region and nothing else.
    assert np.array_equal(new_labels > 0, labels > 0)
    kitchen, living = (new_labels == 1).sum(), (new_labels == 2).sum()
    assert kitchen == living  # anchors mirror each other about the region centre
    assert (new_labels == 3).sum() == (labels == 2).sum()


def test_apply_labels_falls_back_when_model_skips_a_region() -> None:
    plan_labels = PlanLabels(regions=[RegionLabel(id=2, name="bathroom")])
    _, names = apply_labels(_two_regions(), plan_labels, SemanticsConfig())
    assert names == {1: "region 1", 2: "bathroom"}


def test_split_can_be_disabled() -> None:
    plan_labels = PlanLabels(
        regions=[
            RegionLabel(
                id=1,
                name="open plan",
                zones=[Zone(name="kitchen", x=12, y=25), Zone(name="living", x=37, y=25)],
            ),
            RegionLabel(id=2, name="bedroom"),
        ]
    )
    _, names = apply_labels(
        _two_regions(), plan_labels, SemanticsConfig(split_open_plan=False)
    )
    assert names == {1: "open plan", 2: "bedroom"}
