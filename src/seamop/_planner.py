"""Internal multi-seam planning."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from ._search import SeamNotFoundError, find_forward_seam, find_seam

EnergyComputer = Callable[
    [npt.NDArray[np.uint8]],
    npt.NDArray[np.float32],
]
SeamFinder = Callable[
    [npt.NDArray[np.uint8]],
    npt.NDArray[np.bool_],
]


def find_seams(
    image: npt.NDArray[np.uint8],
    num_seams: int,
    compute_energy: EnergyComputer,
) -> npt.NDArray[np.bool_]:
    """Return backward-energy seams in the source image's coordinates."""
    return _remove_seams(
        image,
        num_seams,
        lambda current: find_seam(compute_energy(current)),
    )


def find_forward_seams(
    image: npt.NDArray[np.uint8],
    num_seams: int,
) -> npt.NDArray[np.bool_]:
    """Return forward-energy seams in the source image's coordinates."""
    return _remove_seams(image, num_seams, find_forward_seam)


def _remove_seams(
    image: npt.NDArray[np.uint8],
    num_seams: int,
    seam_finder: SeamFinder,
) -> npt.NDArray[np.bool_]:
    """Remove seams one at a time while tracking source coordinates."""
    height, width = image.shape[:2]
    image = image.copy()
    kept: npt.NDArray[np.signedinteger] = np.arange(height * width)

    while num_seams > 0:
        try:
            seams = seam_finder(image)
        except SeamNotFoundError as error:
            raise RuntimeError("Seam extraction made no progress.") from error

        num_seams -= 1
        flat_seams = seams.ravel()
        image = np.compress(
            ~flat_seams,
            image.reshape(-1, 3),
            axis=0,
        ).reshape(height, -1, 3)
        kept = np.compress(~flat_seams, kept)

    mask = np.ones(height * width, dtype=bool)
    mask[kept] = False
    return mask.reshape(height, width)
