import numpy.testing as npt

from febtools import load_model
from febtools.febio import uses_must_points
from febtools.test.fixtures import DIR_FIXTURES, DIR_OUT


def test_uses_must_points():
    """Test if detection of must point use is accurate

    Only tested for FEBio XML 3.0.  The must point check is mostly for automated
    analysis, which should use the most recent FEBio XML format.

    """
    pth = DIR_FIXTURES / "test_check.uses_must_points.feb"
    model = load_model(pth)
    actual = uses_must_points(model)
    expected = [True, False]
    assert actual == expected
