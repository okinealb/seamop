import numpy as np
import pytest

import seamop
from seamop import CarvingStrategy, _native


@pytest.mark.parametrize(
    ("native_planner", "strategy"),
    [
        (_native.plan, CarvingStrategy.BACKWARD),
        (_native.plan_forward, CarvingStrategy.FORWARD),
    ],
    ids=["backward", "forward"],
)
@pytest.mark.parametrize(
    ("target_height", "target_width"),
    [(4, 5), (3, 5), (4, 3), (3, 3), (2, 2)],
)
def test_native_plan_matches_python_reference(
    native_planner,
    strategy,
    target_height,
    target_width,
):
    image = np.random.default_rng(7).integers(
        0,
        256,
        (4, 5, 3),
        dtype=np.uint8,
    )
    reference = seamop.plan(
        image,
        height=target_height,
        width=target_width,
        strategy=strategy,
    )

    result, removed = native_planner(image, target_height, target_width)

    assert result.dtype == np.uint8
    assert removed.dtype == np.bool_
    assert np.array_equal(result, reference.result())
    assert np.array_equal(removed, reference._removed)


@pytest.mark.parametrize("native_planner", [_native.plan, _native.plan_forward])
def test_native_plan_does_not_mutate_input(native_planner):
    image = np.random.default_rng(8).integers(
        0,
        256,
        (4, 5, 3),
        dtype=np.uint8,
    )
    original = image.copy()

    native_planner(image, 3, 4)

    assert np.array_equal(image, original)


def test_native_plan_rejects_wrong_channel_count():
    image = np.zeros((3, 4, 1), dtype=np.uint8)

    with pytest.raises(ValueError, match="exactly 3 RGB channels"):
        _native.plan(image, 2, 3)


def test_native_plan_rejects_noncontiguous_input():
    image = np.zeros((3, 6, 3), dtype=np.uint8)[:, ::2]

    with pytest.raises(ValueError, match="C-contiguous"):
        _native.plan(image, 2, 2)


def test_native_plan_rejects_invalid_target():
    image = np.zeros((3, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="target width"):
        _native.plan(image, 3, 0)
