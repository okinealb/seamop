import numpy as np
import numpy.typing as npt

def plan(
    image: npt.NDArray[np.uint8],
    target_height: int,
    target_width: int,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool_]]: ...
def plan_forward(
    image: npt.NDArray[np.uint8],
    target_height: int,
    target_width: int,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool_]]: ...
