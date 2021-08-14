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
from febtools.test.fixtures import DIR_FIXTURES, febio_cmd


@pytest.fixture(scope="module")
def FixedNodeBC_Solid_Model(febio_cmd):
    """Solve solid model with fixed nodal boundary conditions"""
    pth = DIR_FIXTURES / "cube_hex8_n=1_solid_all_BCs_fixed.feb"
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
