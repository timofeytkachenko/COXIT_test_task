"""Regression tests for the geometric pipeline and the CLI boundary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from floorplan_seg import PipelineConfig, parse_total_sqft, segment_floorplan
from floorplan_seg.cli import main
from floorplan_seg.config import PreprocessConfig
from floorplan_seg.preprocess import (
    _fill_interior_holes,
    load_bgr,
    plan_mask,
    wall_mask,
)


def test_semantic_stage_is_off_by_default() -> None:
    # README: "Default: classical CV only, no ML, no network".
    assert PipelineConfig().semantics.enabled is False


def test_default_config_segments_without_network(two_room_path: Path) -> None:
    seg = segment_floorplan(two_room_path)
    assert len(seg.rooms) == 2
    assert all(room.name is None for room in seg.rooms)
    assert seg.total_sqft == 400.0
    # The two rooms are symmetric, so areas must be close and sum to the estimate.
    areas = sorted(room.area_px for room in seg.rooms)
    assert areas[0] / areas[1] > 0.9
    assert sum(room.area_sqft for room in seg.rooms) == pytest.approx(400.0, abs=0.2)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("limestone ranch_santa fe_625sq.webp", 625.0),
        ("heritage towers_a1_1 Bed 1 Bath 593 Sq. Ft..webp", 593.0),
        ("plan.webp", None),
    ],
)
def test_parse_total_sqft(name: str, expected: float | None) -> None:
    assert parse_total_sqft(name) == expected


def test_fill_interior_holes_uses_4_connectivity() -> None:
    # A pocket linked to the border only through a diagonal chain is a hole
    # under 4-connectivity; with the OpenCV default (8) it would leak out.
    mask = np.ones((7, 7), bool)
    for i in range(4):
        mask[i, i] = False
    filled = _fill_interior_holes(mask)
    assert filled[3, 3] and filled[2, 2] and filled[1, 1]
    assert not filled[0, 0]


def test_wall_dilate_zero_means_no_dilation(two_room_path: Path) -> None:
    bgr = load_bgr(two_room_path)
    plan = plan_mask(bgr, PreprocessConfig())
    wall0, barrier0 = wall_mask(bgr, plan, PreprocessConfig(wall_dilate=0))
    _, barrier3 = wall_mask(bgr, plan, PreprocessConfig(wall_dilate=3))
    assert np.array_equal(barrier0, wall0 & plan)
    assert barrier3.sum() > barrier0.sum()


def test_wall_dilate_negative_raises(two_room_path: Path) -> None:
    bgr = load_bgr(two_room_path)
    plan = plan_mask(bgr, PreprocessConfig())
    with pytest.raises(ValueError, match="wall_dilate"):
        wall_mask(bgr, plan, PreprocessConfig(wall_dilate=-1))


def test_cli_writes_json_and_png(two_room_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert main([str(two_room_path), "-o", str(out), "--debug"]) == 0

    record = json.loads((out / "two rooms 400sq.json").read_text(encoding="utf-8"))
    assert (out / "two rooms 400sq.png").is_file()
    assert (out / "two rooms 400sq_debug" / "regions.png").is_file()
    assert len(record["rooms"]) == 2
    assert sum(r["relative_area"] for r in record["rooms"]) == pytest.approx(1.0, abs=1e-3)
    assert all(len(r["polygon"]) >= 4 for r in record["rooms"])
    assert record["advertised_total_sqft"] == 400.0


def test_cli_blank_image_exits_cleanly(blank_path: Path, tmp_path: Path) -> None:
    assert main([str(blank_path), "-o", str(tmp_path)]) == 1


def test_cli_missing_file_exits_cleanly(tmp_path: Path) -> None:
    assert main([str(tmp_path / "nope.png"), "-o", str(tmp_path)]) == 1


def test_cli_rejects_out_of_range_tuning(two_room_path: Path, tmp_path: Path) -> None:
    assert main([str(two_room_path), "-o", str(tmp_path), "--wall-dilate", "-1"]) == 1
    assert main([str(two_room_path), "-o", str(tmp_path), "--h-maxima", "0"]) == 1
