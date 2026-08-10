"""Internal resize-plan result and construction."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import SupportsIndex

import numpy as np
import numpy.typing as npt

from ._validation import validate_color

DEFAULT_HIGHLIGHT_COLOR = (255, 0, 0)
SeamFinder = Callable[
    [npt.NDArray[np.uint8], int],
    npt.NDArray[np.bool_],
]


@dataclass(eq=False, frozen=True, repr=False, slots=True)
class ResizePlan:
    """A completed resize and its source-pixel removals.

    Create plans with :func:`seamop.plan` rather than constructing this
    class directly.
    """

    _source: npt.NDArray[np.uint8]
    _result: npt.NDArray[np.uint8]
    _removed: npt.NDArray[np.bool_]

    def __post_init__(self) -> None:
        self._source.flags.writeable = False
        self._result.flags.writeable = False
        self._removed.flags.writeable = False

    def __repr__(self) -> str:
        return (
            f"ResizePlan(source_shape={self.source_shape}, "
            f"target_shape={self.target_shape})"
        )

    @property
    def source_shape(self) -> tuple[int, int, int]:
        return (
            self._source.shape[0],
            self._source.shape[1],
            self._source.shape[2],
        )

    @property
    def target_shape(self) -> tuple[int, int, int]:
        return (
            self._result.shape[0],
            self._result.shape[1],
            self._result.shape[2],
        )

    def result(self) -> npt.NDArray[np.uint8]:
        """Return a writable copy of the resized image."""
        return self._result.copy()

    def preview(
        self,
        color: Sequence[SupportsIndex] = DEFAULT_HIGHLIGHT_COLOR,
    ) -> npt.NDArray[np.uint8]:
        """Return a source-sized copy with planned removals colored.

        Args:
            color: Three RGB integer values from 0 through 255.
        """
        color = validate_color(color)
        preview = self._source.copy()
        preview[self._removed] = color
        return preview


def build_plan(
    image: npt.NDArray[np.uint8],
    *,
    height: int,
    width: int,
    seam_finder: SeamFinder,
) -> ResizePlan:
    """Build a width-first shrinking plan from validated inputs."""
    source = image.copy()
    working = image.copy()
    source_height, source_width = image.shape[:2]
    source_indices: npt.NDArray[np.signedinteger] = np.arange(
        source_height * source_width
    ).reshape(source_height, source_width)
    removed = np.zeros(source_height * source_width, dtype=bool)

    if width < source_width:
        working, source_indices, removed_indices = _remove(
            working,
            source_indices,
            source_width - width,
            seam_finder,
        )
        removed[removed_indices] = True

    if height < source_height:
        oriented_image = np.transpose(working, (1, 0, 2))
        oriented_indices = source_indices.T
        oriented_image, oriented_indices, removed_indices = _remove(
            oriented_image,
            oriented_indices,
            source_height - height,
            seam_finder,
        )
        removed[removed_indices] = True
        working = np.ascontiguousarray(np.transpose(oriented_image, (1, 0, 2)))

    return ResizePlan(
        source,
        working,
        removed.reshape(source_height, source_width),
    )


def _remove(
    image: npt.NDArray[np.uint8],
    source_indices: npt.NDArray[np.signedinteger],
    num_seams: int,
    seam_finder: SeamFinder,
) -> tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.signedinteger],
    npt.NDArray[np.signedinteger],
]:
    """Remove planned seams from an oriented image and its source map."""
    mask = seam_finder(image, num_seams)
    height = image.shape[0]
    flat_mask = mask.ravel()
    flat_image = image.reshape(-1, 3)
    flat_indices = source_indices.ravel()
    return (
        np.compress(~flat_mask, flat_image, axis=0).reshape(height, -1, 3),
        np.compress(~flat_mask, flat_indices).reshape(height, -1),
        np.compress(flat_mask, flat_indices),
    )
