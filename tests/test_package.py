from importlib.metadata import version

import seamop
from seamop.calculator import SeamCalculator
from seamop.methods import EnergyMethod


def test_version_matches_distribution():
    assert seamop.__version__ == version("seamop")


def test_top_level_exports_are_intentional():
    assert set(seamop.__all__) == {
        "ResizePlan",
        "resize",
        "plan",
        "CarvingStrategy",
        "GradientEnergy",
        "LaplacianEnergy",
        "SobelEnergy",
        "__version__",
    }
    assert not hasattr(seamop, "SeamCalculator")
    assert not hasattr(seamop, "EnergyMethod")


def test_advanced_interfaces_remain_in_submodules():
    assert SeamCalculator.__module__ == "seamop.calculator"
    assert EnergyMethod.__module__ == "seamop.methods.interface"
    assert not hasattr(SeamCalculator, "mask_to_index")
