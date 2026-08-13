import numpy as np
import pytest

from seamop import CarvingStrategy, _engine
from seamop._image import normalize_image
from seamop._plan import build_plan
from seamop._planner import find_forward_seams
from seamop.calculator import SeamCalculator
from seamop.methods import GradientEnergy


@pytest.mark.parametrize(
    ("engine_planner", "strategy"),
    [
        (_engine.plan, CarvingStrategy.BACKWARD),
        (_engine.plan_forward, CarvingStrategy.FORWARD),
    ],
    ids=["backward", "forward"],
)
@pytest.mark.parametrize(
    ("target_height", "target_width"),
    [(4, 5), (3, 5), (4, 3), (3, 3), (2, 2)],
)
def test_engine_plan_matches_python_reference(
    engine_planner,
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
    normalized = normalize_image(image)
    if strategy is CarvingStrategy.FORWARD:
        seam_finder = find_forward_seams
    else:
        seam_finder = SeamCalculator(GradientEnergy())
    reference = build_plan(
        normalized,
        height=target_height,
        width=target_width,
        seam_finder=seam_finder,
    )

    result, removed = engine_planner(image, target_height, target_width)

    assert result.dtype == np.uint8
    assert removed.dtype == np.bool_
    assert np.array_equal(result, reference.result())
    assert np.array_equal(removed, reference._removed)


@pytest.mark.parametrize("engine_planner", [_engine.plan, _engine.plan_forward])
def test_engine_plan_does_not_mutate_input(engine_planner):
    image = np.random.default_rng(8).integers(
        0,
        256,
        (4, 5, 3),
        dtype=np.uint8,
    )
    original = image.copy()

    engine_planner(image, 3, 4)

    assert np.array_equal(image, original)


def test_engine_plan_rejects_wrong_channel_count():
    image = np.zeros((3, 4, 1), dtype=np.uint8)

    with pytest.raises(ValueError, match="exactly 3 RGB channels"):
        _engine.plan(image, 2, 3)


def test_engine_plan_rejects_noncontiguous_input():
    image = np.zeros((3, 6, 3), dtype=np.uint8)[:, ::2]

    with pytest.raises(ValueError, match="C-contiguous"):
        _engine.plan(image, 2, 2)


def test_engine_plan_rejects_invalid_target():
    image = np.zeros((3, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="target width"):
        _engine.plan(image, 3, 0)
