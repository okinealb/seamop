"""Content-aware image resizing by seam removal.

Use :func:`resize` for a transformed image or :func:`plan` when a preview and
result must share the same seam decisions.

>>> import numpy as np
>>> import seamop
>>> image = np.zeros((4, 5, 3), dtype=np.uint8)
>>> seamop.resize(image, height=3, width=4).shape
(3, 4, 3)
"""

from importlib.metadata import version as _distribution_version

from ._plan import ResizePlan
from .core import CarvingStrategy, plan, resize
from .methods import GradientEnergy, LaplacianEnergy, SobelEnergy

__version__ = _distribution_version("seamop")

__all__ = [
    "ResizePlan",
    "resize",
    "plan",
    "CarvingStrategy",
    "GradientEnergy",
    "LaplacianEnergy",
    "SobelEnergy",
    "__version__",
]
