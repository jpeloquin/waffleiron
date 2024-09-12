from shutil import copyfile

import numpy as np
import numpy.testing as npt

import waffleiron as wfl
from waffleiron.test.fixtures import DIR_FIXTURES, DIR_OUT


def test_FEBio_RigidBodyObjectData():
    """Test read object data for rigid body force with 1 rigid body"""
    febio_cmd = "febio3.4"  # Need FEBio version ≥ 3.4 to produce object data
    srcpath = DIR_FIXTURES / "bar_explicit_rb_grip_twist_stretch.feb"
    runpath = DIR_OUT / f"test_xplt.RigidBodyObjectData.{febio_cmd}.feb"
    copyfile(srcpath, runpath)
    wfl.febio.run_febio_checked(runpath, cmd=febio_cmd, threads=1)
    with open(runpath.with_suffix(".xplt"), "rb") as f:
        xplt = wfl.xplt.XpltData(f.read())
    actual = xplt.values("Force", 0)["Force"]
    expected = np.array(
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
        dtype=np.float32,
    )
    npt.assert_almost_equal(actual, expected, decimal=6)


def test_FEBio_RigidBodyObjectsData():
    """Test read object data for rigid body force with 2 rigid bodies"""
    febio_cmd = "febio3.4"  # Need FEBio version ≥ 3.4 to produce object data
    srcpath = DIR_FIXTURES / "bar_explicit_rb_grips_twist_stretch.feb"
    runpath = DIR_OUT / f"test_xplt.RigidBodyObjectsData.{febio_cmd}.feb"
    copyfile(srcpath, runpath)
    wfl.febio.run_febio_checked(runpath, cmd=febio_cmd, threads=1)
    with open(runpath.with_suffix(".xplt"), "rb") as f:
        xplt = wfl.xplt.XpltData(f.read())

    # Right grip should be first object (FEBio ID1 = 2).  It has variable
    # displacement and rotation via prescribed BCs.
    expected = np.array(
        [
            [0.0000000e00, 0.0000000e00, 0.0000000e00],
            [8.60464677727, -5.0073800e-07, 5.3780335e-01],
            [17.2826320708, -3.4109435e-06, 1.3892564e00],
            [25.9253135362, -2.4284923e-06, 2.4894822e00],
            [34.4453097791, -1.3585981e-04, 3.7830777e00],
            [43.4994194053, -8.3914092e-05, 5.3527007e00],
            [53.0667235758, -8.4563995e-05, 7.2032232e00],
            [63.1633898109, -1.2433829e-04, 9.3459635e00],
            [73.7799867408, -1.3212441e-04, 1.1786593e01],
            [84.9125825896, -4.2944087e-04, 1.4532055e01],
            [96.7283631274, -3.5772764e-04, 1.7634827e01],
            [109.234349559, -2.9566133e-04, 2.1112232e01],
            [122.43170897, -4.1294866e-04, 2.4979637e01],
            [124.065599025, 1.2126653e-06, 2.5471798e01],
        ],
        dtype=np.float32,
    )
    actual = np.array(xplt.values("Force", 0)["Force"])
    npt.assert_almost_equal(actual, expected, decimal=6)

    # Left grip should be second object (FEBio ID1 = 3).  It has constant
    # displacement and rotation via fixed BC.  Usually FEBio does not store the
    # forces or moments in the xplt file for fixed BCs, although somehow it manages
    # to calculate them for the text data file.
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [-8.60465, 5.56916177e-07, -0.53780365],
            [-17.282639, 3.48201047e-06, -1.3892581],
            [-25.925314, 2.42881652e-06, -2.4894822],
            [-34.44935, 6.90777015e-05, -3.7833438],
            [-43.497646, 4.95319473e-05, -5.351818],
            [-53.07068, 5.26015501e-05, -7.203857],
            [-63.166084, 1.11007052e-04, -9.3464775],
            [-73.78107, 1.67513179e-04, -11.7869005],
            [-84.91095, 1.00016696e-04, -14.531519],
            [-96.72665, -1.93339583e-05, -17.634287],
            [-109.232574, -1.18282471e-04, -21.11171],
            [-122.4299, -3.62322171e-05, -24.97913],
            [-124.06552, -1.93067035e-06, -25.471775],
        ],
        dtype=np.float32,
    )
    actual = np.array(xplt.values("Force", 1)["Force"])
    npt.assert_almost_equal(actual, expected, decimal=6)
