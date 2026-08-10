from itertools import product

import numpy as np
import pytest

from seamop._search import (
    SeamNotFoundError,
    cumulative_costs,
    find_forward_seam,
    find_seam,
    forward_cumulative_costs,
)


def minimum_seam_cost(energy):
    """Return the cheapest connected top-to-bottom path."""
    height, width = energy.shape
    paths = product(range(width), repeat=height)
    return min(
        sum(energy[row, column] for row, column in enumerate(path))
        for path in paths
        if all(abs(left - right) <= 1 for left, right in zip(path, path[1:]))
    )


def forward_path_cost(image, path):
    total = 0.0
    for row in range(1, len(path)):
        column = path[row]
        current = image[row].astype(np.float64)
        previous = image[row - 1].astype(np.float64)
        left = current[max(0, column - 1)]
        right = current[min(image.shape[1] - 1, column + 1)]
        upward = np.abs(right - left).sum()
        if path[row - 1] == column - 1:
            total += upward + np.abs(previous[column] - left).sum()
        elif path[row - 1] == column:
            total += upward
        else:
            total += upward + np.abs(previous[column] - right).sum()
    return total


def minimum_forward_seam_cost(image):
    height, width = image.shape[:2]
    return min(
        forward_path_cost(image, path)
        for path in product(range(width), repeat=height)
        if all(abs(left - right) <= 1 for left, right in zip(path, path[1:]))
    )


@pytest.mark.parametrize(
    "energy",
    [
        np.array([[4, 1, 3]], dtype=np.float32),
        np.array([[3, 1], [1, 3]], dtype=np.float32),
        np.array(
            [
                [5, 1, 4, 3],
                [2, 6, 1, 7],
                [4, 2, 3, 1],
            ],
            dtype=np.float32,
        ),
    ],
    ids=["single-row", "two-rows", "three-rows"],
)
def test_finds_minimum_connected_seam_without_mutation(energy):
    original = energy.copy()

    mask = find_seam(energy)
    columns = np.argmax(mask, axis=1)

    assert mask.shape == energy.shape
    assert mask.dtype == np.bool_
    assert np.all(mask.sum(axis=1) == 1)
    assert np.all(np.abs(np.diff(columns)) <= 1)
    assert energy[mask].sum() == minimum_seam_cost(energy)
    assert np.array_equal(energy, original)


@pytest.mark.parametrize("sign", [-1, 1], ids=["negative", "positive"])
def test_cumulative_costs_remain_finite(sign):
    energy = np.full(
        (3, 3),
        sign * np.finfo(np.float32).max,
        dtype=np.float32,
    )

    costs = cumulative_costs(energy)

    assert costs.dtype == np.float64
    assert np.isfinite(costs).all()


def test_rejects_exhausted_energy():
    energy = np.full((3, 3), np.inf, dtype=np.float32)

    with pytest.raises(SeamNotFoundError):
        find_seam(energy)


def test_finds_minimum_forward_energy_seam_without_mutation():
    image = np.array(
        [
            [[10, 20, 30], [30, 20, 10], [50, 40, 30], [70, 60, 50]],
            [[20, 30, 40], [40, 30, 20], [60, 50, 40], [80, 70, 60]],
            [[30, 40, 50], [50, 40, 30], [70, 60, 50], [90, 80, 70]],
        ],
        dtype=np.uint8,
    )
    original = image.copy()

    mask = find_forward_seam(image)
    path = tuple(np.argmax(mask, axis=1))
    costs = forward_cumulative_costs(image)

    assert mask.shape == image.shape[:2]
    assert mask.dtype == np.bool_
    assert np.all(mask.sum(axis=1) == 1)
    assert np.all(np.abs(np.diff(path)) <= 1)
    assert forward_path_cost(image, path) == pytest.approx(
        minimum_forward_seam_cost(image)
    )
    assert costs[-1, path[-1]] == pytest.approx(minimum_forward_seam_cost(image))
    assert np.array_equal(image, original)
