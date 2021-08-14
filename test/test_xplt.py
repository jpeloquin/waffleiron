from shutil import copyfile

import numpy as np
from numpy import array, float32
import numpy.testing as npt

import febtools as feb
from febtools.test.fixtures import DIR_FIXTURES, DIR_OUT


def test_FEBio_RigidBodyForceObjectData():
    """Test read of rigid body reaction forces as object data from FEBio plotfile"""
    febio_cmd = "febio3.4"
    srcpath = DIR_FIXTURES / "bar_twist_stretch_rb_grip.feb"
    runpath = DIR_OUT / f"test_xplt.RigidBodyReactionForce.{febio_cmd}.feb"
    copyfile(srcpath, runpath)
    feb.febio.run_febio_checked(runpath, cmd=febio_cmd)
    with open(runpath.with_suffix(".xplt"), "rb") as f:
        xplt = feb.xplt.XpltData(f.read())
    actual = xplt.values("Force", 0)["Force"]
    expected = array(
        [
            [0.0000000e00, 0.0000000e00, 0.0000000e00],
            [8.6046467e00, -5.0073800e-07, 5.3780335e-01],
            [1.7282633e01, -3.4109435e-06, 1.3892564e00],
            [2.5925314e01, -2.4284923e-06, 2.4894822e00],
            [3.4445309e01, -1.3585981e-04, 3.7830777e00],
            [4.3499420e01, -8.3914092e-05, 5.3527007e00],
            [5.3066723e01, -8.4563995e-05, 7.2032232e00],
            [6.3163391e01, -1.2433829e-04, 9.3459635e00],
            [7.3779984e01, -1.3212441e-04, 1.1786593e01],
            [8.4912582e01, -4.2944087e-04, 1.4532055e01],
            [9.6728363e01, -3.5772764e-04, 1.7634827e01],
            [1.0923435e02, -2.9566133e-04, 2.1112232e01],
            [1.2243171e02, -4.1294866e-04, 2.4979637e01],
            [1.2406560e02, 1.2126654e-06, 2.5471798e01],
        ],
        dtype=float32,
    )
    npt.assert_almost_equal(actual, expected, decimal=6)
