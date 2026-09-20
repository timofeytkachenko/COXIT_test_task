"""Shared fixtures: a synthetic two-room render, no network access needed."""

from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

WALL_BGR = (215, 215, 215)
FLOOR_BGR = (90, 120, 150)


def synthetic_two_room_render() -> np.ndarray:
    """White page, wall-coloured outline, one divider wall with a doorway.

    Geometry (x right, y down): plan spans x 100..500, y 100..300; walls are
    12 px thick; the divider at x 294..306 leaves a 30 px doorway at the top.
    """
    img = np.full((400, 600, 3), 255, np.uint8)
    img[100:300, 100:500] = WALL_BGR
    img[112:288, 112:488] = FLOOR_BGR
    img[112:288, 294:306] = WALL_BGR
    img[112:142, 294:306] = FLOOR_BGR  # doorway
    return img


@pytest.fixture(scope="session")
def two_room_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "two rooms 400sq.png"
    assert cv.imwrite(str(path), synthetic_two_room_render())
    return path


@pytest.fixture(scope="session")
def blank_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "blank.png"
    assert cv.imwrite(str(path), np.full((200, 300, 3), 255, np.uint8))
    return path
