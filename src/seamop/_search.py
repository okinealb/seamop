"""Internal single-seam dynamic-programming search."""

import numpy as np
import numpy.typing as npt


class SeamNotFoundError(Exception):
    """Raised when no finite seam remains in an energy map."""


def find_seam(
    energy: npt.NDArray[np.float32],
) -> npt.NDArray[np.bool_]:
    """Return one minimum-cost seam without modifying the energy map."""
    costs = cumulative_costs(energy)
    seam = np.zeros(energy.shape, dtype=bool)
    column = int(np.argmin(costs[-1]))

    if costs[-1, column] == np.inf:
        raise SeamNotFoundError("No valid starting point found.")

    seam[-1, column] = True
    height, width = energy.shape

    for row in range(height - 2, -1, -1):
        left = max(0, column - 1)
        right = min(width, column + 2)
        column = int(np.argmin(costs[row, left:right])) + left

        if costs[row, column] == np.inf:
            raise SeamNotFoundError("No valid seam found.")

        seam[row, column] = True

    return seam


def find_forward_seam(
    image: npt.NDArray[np.uint8],
) -> npt.NDArray[np.bool_]:
    """Return one minimum-cost forward-energy seam."""
    costs = forward_cumulative_costs(image)
    seam = np.zeros(image.shape[:2], dtype=bool)
    column = int(np.argmin(costs[-1]))

    if costs[-1, column] == np.inf:
        raise SeamNotFoundError("No valid starting point found.")

    seam[-1, column] = True
    height, width = image.shape[:2]

    for row in range(height - 1, 0, -1):
        left_cost, up_cost, right_cost = _forward_transition_costs(image, row)
        start = max(0, column - 1)
        stop = min(width, column + 2)
        candidates = costs[row - 1, start:stop].copy()

        candidates[column - start] += up_cost[column]
        if column > 0:
            candidates[0] += left_cost[column]
        if column + 1 < width:
            candidates[column + 1 - start] += right_cost[column]

        column = int(np.argmin(candidates)) + start
        if candidates[column - start] == np.inf:
            raise SeamNotFoundError("No valid seam found.")
        seam[row - 1, column] = True

    return seam


def cumulative_costs(
    energy: npt.NDArray[np.float32],
) -> npt.NDArray[np.float64]:
    """Return cumulative minimum seam costs for an energy map."""
    costs = energy.astype(np.float64, copy=True)

    for row in range(1, energy.shape[0]):
        previous = costs[row - 1]
        current = costs[row]
        current[1:-1] += np.minimum(
            np.minimum(previous[:-2], previous[1:-1]),
            previous[2:],
        )
        current[0] += min(previous[0], previous[1])
        current[-1] += min(previous[-1], previous[-2])

    return costs


def forward_cumulative_costs(
    image: npt.NDArray[np.uint8],
) -> npt.NDArray[np.float64]:
    """Return cumulative costs for forward-energy seam search."""
    height, width = image.shape[:2]
    costs = np.zeros((height, width), dtype=np.float64)

    for row in range(1, height):
        left_cost, up_cost, right_cost = _forward_transition_costs(image, row)
        previous = costs[row - 1]
        current = costs[row]
        current[:] = previous + up_cost
        current[1:] = np.minimum(
            current[1:],
            previous[:-1] + left_cost[1:],
        )
        current[:-1] = np.minimum(
            current[:-1],
            previous[1:] + right_cost[:-1],
        )

    return costs


def _forward_transition_costs(
    image: npt.NDArray[np.uint8],
    row: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Return left, up, and right transition costs for one image row."""
    current = image[row].astype(np.float64, copy=False)
    previous = image[row - 1].astype(np.float64, copy=False)
    left = np.empty_like(current)
    right = np.empty_like(current)
    left[0] = current[0]
    left[1:] = current[:-1]
    right[-1] = current[-1]
    right[:-1] = current[1:]

    up = np.abs(right - left).sum(axis=1)
    left_cost = up + np.abs(previous - left).sum(axis=1)
    right_cost = up + np.abs(previous - right).sum(axis=1)
    return left_cost, up, right_cost
