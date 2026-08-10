import numpy as np
import pytest

from seamop._planner import find_forward_seams, find_seams


def test_tracks_source_coordinates_across_removals():
    image = np.zeros((2, 10, 3), dtype=np.uint8)
    original = image.copy()
    widths = []

    def column_energy(current):
        widths.append(current.shape[1])
        columns = np.arange(current.shape[1], dtype=np.float32)
        return np.broadcast_to(columns, current.shape[:2]).copy()

    mask = find_seams(image, 3, column_energy)

    assert widths == [10, 9, 8]
    assert np.array_equal(np.flatnonzero(mask[0]), np.arange(3))
    assert np.all(mask.sum(axis=1) == 3)
    assert np.array_equal(image, original)


def test_stops_without_progress():
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="no progress"):
        find_seams(
            image,
            1,
            lambda current: np.full(
                current.shape[:2],
                np.inf,
                dtype=np.float32,
            ),
        )


def test_forward_tracks_source_coordinates_and_preserves_input():
    image = np.random.default_rng(2).integers(
        0,
        256,
        (4, 8, 3),
        dtype=np.uint8,
    )
    original = image.copy()

    mask = find_forward_seams(image, 3)

    assert mask.shape == image.shape[:2]
    assert mask.dtype == np.bool_
    assert np.all(mask.sum(axis=1) == 3)
    assert np.array_equal(image, original)
