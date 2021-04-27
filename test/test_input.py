# Run these tests with pytest
import unittest
import os
from pathlib import Path
import subprocess
import numpy.testing as npt
import numpy as np
import pytest

import febtools as feb
from febtools.febio import run_febio_checked
from febtools.test.fixtures import febio_cmd


class MeshSolutionTest(unittest.TestCase):
    """Tests `MeshSolution.f`

    This test also depends on `textdata_list` and `XpltReader`
    functioning correctly.  Results from an FEBio simulation are read
    from the text log and the binary file.  F tensors are computed for
    each element based on the binary file data and compared to those
    recorded in the text log.

    """

    def setUp(self):
        self.soln = feb.input.XpltReader(
            os.path.join("test", "fixtures", "complex_loading.xplt")
        )
        reader = feb.input.FebReader(
            os.path.join("test", "fixtures", "complex_loading.feb")
        )
        self.model = reader.model()
        self.model.apply_solution(self.soln)
        self.elemdata = feb.input.textdata_list(
            os.path.join("test", "fixtures", "complex_loading_elem_data.txt"), delim=","
        )
        self.nodedata = feb.input.textdata_list(
            os.path.join("test", "fixtures", "complex_loading_node_data.txt"), delim=","
        )

    def cmp_f(self, row, col, key):
        """Helper function for comparing f tensors.

        Check the F tensor read from the xplt file against the text
        data in the logfile.

        """
        for i, e in enumerate(self.model.mesh.elements):
            # Check if rigid body. FEBio gives rigid bodies F_ii =
            # +/-1
            if not np.isnan(self.elemdata[-1]["s1"][i]):
                f = e.f((0, 0, 0))
                npt.assert_approx_equal(
                    f[row, col], self.elemdata[-1][key][i], significant=5
                )

    def test_fx(self):
        self.cmp_f(0, 0, "Fxx")

    def test_fy(self):
        self.cmp_f(1, 1, "Fyy")

    def test_fz(self):
        self.cmp_f(2, 2, "Fzz")

    def test_fxy(self):
        self.cmp_f(0, 1, "Fxy")

    def test_fxz(self):
        self.cmp_f(0, 2, "Fxz")

    def test_fyx(self):
        self.cmp_f(1, 0, "Fyx")

    def test_fyz(self):
        self.cmp_f(1, 2, "Fyz")

    def test_fzx(self):
        self.cmp_f(2, 0, "Fzx")

    def test_fzy(self):
        self.cmp_f(2, 1, "Fzy")


@pytest.fixture(scope="module")
def FixedNodeBC_Solid_Model(febio_cmd):
    """Solve solid model with fixed nodal boundary conditions"""
    pth = Path("test") / "fixtures" / "cube_hex8_n=1_solid_all_BCs_fixed.feb"
    run_febio_checked(pth, cmd=febio_cmd)
    yield pth
    # Delete FEBio-generated output
    pth.with_suffix(".log").unlink()
    pth.with_suffix(".xplt").unlink()


def test_FEBio_FixedNodeBC_Solid(FixedNodeBC_Solid_Model):
    """Test read of FEBio XML and XPLT files with fixed boundary conditions."""
    pth = FixedNodeBC_Solid_Model
    # Test 1: Read XML?
    model = feb.load_model(pth)
    # Test 2: Read XPLT?
    pth_xplt = pth.with_suffix(".xplt")
    with open(pth_xplt, "rb") as f:
        xplt = feb.xplt.XpltData(f.read())
    # Test 3: Read mesh from XPLT?
    mesh = xplt.mesh()
    # Check node values
    assert len(mesh.nodes) == 8
    assert np.array(mesh.nodes).shape[1] == 3
    npt.assert_almost_equal(mesh.nodes[3], [-0.5, 0.5, 0.0])
    # Check element values
    assert len(mesh.elements) == 1
    npt.assert_equal(mesh.elements[0].nodes, mesh.nodes)


@pytest.fixture(scope="module")
def FixedNodeBC_Biphasic_Model(febio_cmd):
    """Solve biphasic model with fixed nodal boundary conditions"""
    pth = Path("test") / "fixtures" / "cube_hex8_n=1_biphasic_all_BCs_fixed.feb"
    run_febio_checked(pth, cmd=febio_cmd)
    yield pth
    # Delete FEBio-generated output
    pth.with_suffix(".log").unlink()
    pth.with_suffix(".xplt").unlink()


def test_FEBio_Fixed_NodeBC_Biphasic(FixedNodeBC_Biphasic_Model):
    pth = FixedNodeBC_Biphasic_Model
    # Test 1: Read XML?
    model = feb.load_model(pth)
    # Test 2: Read XPLT?
    pth_xplt = pth.with_suffix(".xplt")
    with open(str(pth_xplt), "rb") as f:
        xplt = feb.xplt.XpltData(f.read())
    # Test 3: Read mesh from XPLT?
    mesh = xplt.mesh()
    # Check node values
    assert len(mesh.nodes) == 8
    assert np.array(mesh.nodes).shape[1] == 3
    npt.assert_almost_equal(mesh.nodes[3], [-0.5, 0.5, 0.0])
    # Check element values
    assert len(mesh.elements) == 1
    npt.assert_equal(mesh.elements[0].nodes, mesh.nodes)


class EnvironmentConstants(unittest.TestCase):
    """Test read of environmental constants."""

    path = "test/fixtures/isotropic_elastic.feb"

    def test_temperature_constant(self):
        model = feb.load_model(self.path)
        assert model.environment["temperature"] == 300


class UniversalConstants(unittest.TestCase):
    """Test read of universal constants."""

    path = "test/fixtures/isotropic_elastic.feb"

    def test_idal_gas_constant(self):
        model = feb.load_model(self.path)
        assert model.constants["R"] == 8.314e-6

    def test_Faraday_constant(self):
        model = feb.load_model(self.path)
        assert model.constants["F"] == 96485e-9
