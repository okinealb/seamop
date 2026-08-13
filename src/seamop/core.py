"""Public operations for content-aware image resizing."""

from enum import Enum
from typing import SupportsIndex

import numpy as np
import numpy.typing as npt

from ._engine import plan as engine_plan
from ._engine import plan_forward as engine_plan_forward
from ._image import ImageInput, normalize_image
from ._plan import ResizePlan, build_plan
from ._validation import validate_resize_target
from .calculator import SeamCalculator
from .methods import GradientEnergy
from .methods.interface import EnergyCallable


class CarvingStrategy(Enum):
    """Seam-carving policy used to select seams."""

    BACKWARD = "backward"
    FORWARD = "forward"


def plan(
    image: ImageInput,
    *,
    height: SupportsIndex,
    width: SupportsIndex,
    energy: EnergyCallable | None = None,
    strategy: CarvingStrategy = CarvingStrategy.BACKWARD,
) -> ResizePlan:
    """Plan a width-first resize without mutating the source.

    Args:
        image: Filesystem path, Pillow image, RGB uint8 NumPy array, or nested
            RGB integer list.
        height: Positive target height no larger than the source height.
        width: Positive target width no larger than the source width.
        energy: Energy callable used by the backward strategy. Defaults to
            :class:`GradientEnergy`. The forward strategy does not accept an
            energy callable.
        strategy: Seam-carving policy. Defaults to
            :attr:`CarvingStrategy.BACKWARD`.

    Returns:
        A plan that produces a carved result and source-sized preview from the
        same seam decisions.

    Raises:
        TypeError: An input has an unsupported type.
        ValueError: The image or target dimensions are invalid.
    """
    if not isinstance(strategy, CarvingStrategy):
        raise TypeError("strategy must be a CarvingStrategy.")

    normalized = normalize_image(image)
    height = validate_resize_target("height", height, normalized.shape[0])
    width = validate_resize_target("width", width, normalized.shape[1])
    if strategy is CarvingStrategy.FORWARD:
        if energy is not None:
            raise ValueError("The forward strategy does not accept an energy callable.")
        result, removed = engine_plan_forward(normalized, height, width)
        return ResizePlan(normalized, result, removed)
    if energy is None or type(energy) is GradientEnergy:
        result, removed = engine_plan(normalized, height, width)
        return ResizePlan(normalized, result, removed)
    return build_plan(
        normalized,
        height=height,
        width=width,
        seam_finder=SeamCalculator(energy),
    )


def resize(
    image: ImageInput,
    *,
    height: SupportsIndex,
    width: SupportsIndex,
    energy: EnergyCallable | None = None,
    strategy: CarvingStrategy = CarvingStrategy.BACKWARD,
) -> npt.NDArray[np.uint8]:
    """Return a width-first resized RGB uint8 image.

    The source input is not mutated. Target dimensions must be positive and no
    larger than the source because seam addition is not implemented.
    """
    return plan(
        image,
        height=height,
        width=width,
        energy=energy,
        strategy=strategy,
    ).result()
