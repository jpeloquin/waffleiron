# Run these tests with pytest
import shutil
import unittest
from pathlib import Path

import numpy.testing as npt
import numpy as np
import pytest

import waffleiron as wfl
from waffleiron import load_model
from waffleiron.febio import run_febio_checked
from waffleiron.input import read_febio_xml
from waffleiron.output import write_feb
from waffleiron.test.fixtures import DIR_FIXTURES, DIR_OUT, febio_cmd


@pytest.fixture(scope="module")
def FixedNodeBC_Solid_Model(febio_cmd):
    """Solve solid model with fixed nodal boundary conditions"""
    pth_in = DIR_FIXTURES / "cube_hex8_n=1_solid_all_BCs_fixed.feb"
    pth_out = DIR_OUT / "cube_hex8_n=1_solid_all_BCs_fixed.feb"
    shutil.copy(pth_in, pth_out)
    run_febio_checked(pth_out, threads=1)
    yield pth_out
    # Delete FEBio-generated output
    pth_out.with_suffix(".log").unlink()
    pth_out.with_suffix(".xplt").unlink()


def test_Unit_Read_FebioXMLVerbatimMaterial():
    """Test read of unrecognized material in FEBio XML"""
    p_in = DIR_FIXTURES / "cube_hex8_n=1_unknown_material_febioxml3.0.feb"
    with open(p_in, "rb") as f:
        xml_in = read_febio_xml(f)
    model = load_model(p_in)
    p_out = DIR_OUT / "cube_hex8_n=1_unknown_material_febioxml3.0.feb"
    with open(p_out, "wb") as f:
        write_feb(model, f, version="3.0")
    with open(p_out, "rb") as f:
        xml_out = read_febio_xml(f)
    e = xml_out.find("Material/material[@name='Material1']")
    assert e.tag == "material"
    assert e.attrib["name"] == "Material1"
    assert e.attrib["type"] == "special snowflake"
    assert len(e) == 2
    assert e.find("param_a").text == "1"
    assert e.find("param_b").text == "2"


def test_read_FEBio_XML_element_IDs_and_sets():
    """Test reading an FEBio XML file with element IDs and element sets"""
    # Test 1: Read XML?
    model = wfl.load_model(DIR_FIXTURES / "element_IDs_element_sets.feb")
    # Test 2: Does the element set have the correct element IDs?
    assert set(lbl for lbl, _ in model.named["elements"].pairs()) == set([2, 3, 4, 5])
    # Test 3: Does the node set referred to in the BC via an element set name refer
    # to the correct element nodes?
    assert model.fixed["node"][("x1", "displacement")] == {10, 1, 2, 14, 11, 5, 6, 19}
    # TODO: test write–sim–read round trip


def test_FEBio_FixedNodeBC_Solid(FixedNodeBC_Solid_Model):
    """Test read of FEBio XML and XPLT files with fixed boundary conditions."""
    pth = FixedNodeBC_Solid_Model
    # Test 1: Read XML?
    model = wfl.load_model(pth)
    # Test 2: Read XPLT?
    pth_xplt = pth.with_suffix(".xplt")
    with open(pth_xplt, "rb") as f:
        xplt = wfl.xplt.XpltData(f.read())
    # Test 3: Read mesh from XPLT?
    mesh, _ = xplt.mesh()
    # Check node values
    assert len(mesh.nodes) == 8
    assert np.array(mesh.nodes).shape[1] == 3
    npt.assert_almost_equal(mesh.nodes[3], [-0.5, 0.5, 0.0])
    # Check element values
    assert len(mesh.elements) == 1
    npt.assert_equal(mesh.elements[0].nodes, mesh.nodes)


@pytest.fixture(scope="module")
def FixedNodeBC_Biphasic_Model():
    """Solve biphasic model with fixed nodal boundary conditions"""
    pth_in = DIR_FIXTURES / "cube_hex8_n=1_biphasic_all_BCs_fixed.feb"
    pth_out = DIR_OUT / "cube_hex8_n=1_biphasic_all_BCs_fixed.feb"
    shutil.copy(pth_in, pth_out)
    run_febio_checked(pth_out, threads=1)
    yield pth_out
    # Delete FEBio-generated output
    pth_out.with_suffix(".log").unlink()
    pth_out.with_suffix(".xplt").unlink()


def test_FEBio_Fixed_NodeBC_Biphasic(FixedNodeBC_Biphasic_Model):
    pth = FixedNodeBC_Biphasic_Model
    # Test 1: Read XML?
    model = wfl.load_model(pth)
    # Test 2: Read XPLT?
    pth_xplt = pth.with_suffix(".xplt")
    with open(str(pth_xplt), "rb") as f:
        xplt = wfl.xplt.XpltData(f.read())
    # Test 3: Read mesh from XPLT?
    mesh, _ = xplt.mesh()
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
        model = wfl.load_model(self.path)
        assert model.environment["temperature"] == 300


class UniversalConstants(unittest.TestCase):
    """Test read of universal constants."""

    path = "test/fixtures/isotropic_elastic.feb"

    def test_idal_gas_constant(self):
        model = wfl.load_model(self.path)
        assert model.constants["R"] == 8.314e-6

    def test_Faraday_constant(self):
        model = wfl.load_model(self.path)
        assert model.constants["F"] == 96485e-9


#####################################################
# Test reading boundary conditions of various kinds #
#####################################################


def test_roundtrip_variable_rigid_bc_force():
    """Test reading a time-varying force BC on an (implicit) rigid body"""
    pth_original = DIR_FIXTURES / "ccomp_elastic_implicit_rb_force.xml4.feb"
    model = wfl.load_model(pth_original)
    febio_cmd = "febio4"

    # Test 1: Read—verify that rigid body force BC in step 1 showed up
    assert len(model.steps[0].step.bc["body"]) == 1
    bc = list(model.steps[0].step.bc["body"].values())[0]["x1"]
    assert bc["variable"] == "force"
    assert len(bc["sequence"].points) == 2
    assert bc["sequence"].points[-1] == (1, 1)

    # Test 2: Write
    pth_write = DIR_OUT / (
        f"{Path(__file__).with_suffix('').name}."
        + f"{pth_original.name.removesuffix(".xml4.feb")}.{febio_cmd}.feb"
    )
    with open(pth_write, "wb") as f:
        wfl.output.write_feb(model, f)

    # Test 3: Run
    wfl.febio.run_febio_checked(pth_write, cmd=febio_cmd, threads=1)

    # Test 4: verify output
    solved = wfl.load_model(pth_write)
    force = solved.solution.values("reaction forces", 0)["reaction forces"]
    npt.assert_allclose(force[-1, :], np.array([2, 0, 0]), rtol=1e-7, atol=5e-7)
