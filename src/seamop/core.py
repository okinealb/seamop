"""Public operations for content-aware image resizing."""

from enum import Enum
from typing import SupportsIndex
from warnings import warn

import numpy as np
import numpy.typing as npt

from ._image import ImageInput, normalize_image
from ._plan import ResizePlan, build_plan
from ._planner import find_forward_seams
from ._validation import validate_resize_target
from .calculator import SeamCalculator
from .methods import GradientEnergy
from .methods.interface import EnergyCallable


class CarvingStrategy(Enum):
    """Seam-carving policy used to select seams."""

    BACKWARD = "backward"
    FORWARD = "forward"


def _resolve_energy(
    energy: EnergyCallable | None,
    method: EnergyCallable | None,
) -> EnergyCallable | None:
    """Resolve the canonical energy name and its deprecated alias."""
    if energy is not None and method is not None:
        raise TypeError("Pass either energy or method, not both.")
    if method is not None:
        warn(
            "method= is deprecated; use energy= instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    return energy if energy is not None else method


def plan(
    image: ImageInput,
    *,
    height: SupportsIndex,
    width: SupportsIndex,
    energy: EnergyCallable | None = None,
    strategy: CarvingStrategy = CarvingStrategy.BACKWARD,
    method: EnergyCallable | None = None,
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
        method: Deprecated alias for ``energy``.

    Returns:
        A plan that produces a carved result and source-sized preview from the
        same seam decisions.

    Raises:
        TypeError: An input has an unsupported type.
        ValueError: The image or target dimensions are invalid.
    """
    if not isinstance(strategy, CarvingStrategy):
        raise TypeError("strategy must be a CarvingStrategy.")
    energy = _resolve_energy(energy, method)

    normalized = normalize_image(image)
    height = validate_resize_target("height", height, normalized.shape[0])
    width = validate_resize_target("width", width, normalized.shape[1])
    if strategy is CarvingStrategy.FORWARD:
        if energy is not None:
            raise ValueError("The forward strategy does not accept an energy callable.")
        seam_finder = find_forward_seams
    else:
        seam_finder = SeamCalculator(GradientEnergy() if energy is None else energy)
    return build_plan(
        normalized,
        height=height,
        width=width,
        seam_finder=seam_finder,
    )


def resize(
    image: ImageInput,
    *,
    height: SupportsIndex,
    width: SupportsIndex,
    energy: EnergyCallable | None = None,
    strategy: CarvingStrategy = CarvingStrategy.BACKWARD,
    method: EnergyCallable | None = None,
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
        method=method,
    ).result()
