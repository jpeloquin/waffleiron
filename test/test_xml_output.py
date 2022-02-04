# Run these tests with pytest
import hashlib
from pathlib import Path

import waffleiron as wfl
from waffleiron import febioxml_2_5 as febioxml
from waffleiron.test.fixtures import DIR_FIXTURES, DIR_OUT, xml_version


# Basic XML output.


def test_write_feb(xml_version):
    mat = wfl.material.NeoHookean(E=1.1, ν=0.3)
    mesh = wfl.mesh.rectangular_prism((1, 2), (1, 2), (1, 2), material=mat)
    model = wfl.Model(mesh)
    basename = Path(__file__).with_suffix("").stem
    pth = DIR_OUT / f"{basename}_xmlver={xml_version}.feb"
    with open(pth, "wb") as f:
        wfl.output.write_feb(model, f, version=xml_version)
    pth.unlink()


def test_write_feb2p0():
    # This function is special because XML 2.0 is tested, which is not
    # the case for other test functions.
    test_write_feb("2.0")


# Environmental constants


def test_write_temperature_constant(xml_version):
    mat = wfl.material.HolmesMow(1.5, 0.3, 2.0)
    mesh = wfl.mesh.rectangular_prism((1, 2), (1, 2), (1, 2), material=mat)
    model = wfl.Model(mesh)
    model.environment["temperature"] = 341
    xml = wfl.output.xml(model, version=xml_version)
    assert xml.find("Globals/Constants/T").text == "341"


# Universal constants


def test_write_ideal_gas_constant(xml_version):
    mat = wfl.material.HolmesMow(1.5, 0.3, 2.0)
    mesh = wfl.mesh.rectangular_prism((1, 2), (1, 2), (1, 2), material=mat)
    model = wfl.Model(mesh)
    model.constants["R"] = 8.314  # J/mol·K
    xml = wfl.output.xml(model, version=xml_version)
    assert xml.find("Globals/Constants/R").text == "8.314"


def test_write_Faraday_constant(xml_version):
    mat = wfl.material.HolmesMow(1.5, 0.3, 2.0)
    mesh = wfl.mesh.rectangular_prism((1, 2), (1, 2), (1, 2), material=mat)
    model = wfl.Model(mesh)
    model.constants["F"] = 26.801  # A·h/mol
    xml = wfl.output.xml(model, version=xml_version)
    assert xml.find("Globals/Constants/Fc").text == "26.801"


# Test for mutation


def test_repeated_write_gives_same_output(xml_version):
    """Test repeated write of same model file"""
    pth_in = DIR_FIXTURES / "uniaxial_tension_implicit_rb_grips_biphasic.feb"
    model = wfl.load_model(pth_in)
    # First write
    pth_out1 = DIR_OUT / (
        f"{Path(__file__).with_suffix('').name}."
        f"repeated_write_gives_same_output_-_write=1.xml{xml_version}.feb"
    )
    # Need to remove time stamp or we won't have same output
    xml = wfl.output.xml(model, version=xml_version)
    tstamp = xml.xpath("//comment()")[0]
    xml.getroot().remove(tstamp)
    with open(pth_out1, "wb") as f:
        wfl.output.write_xml(xml, f)
    with open(pth_out1, "rb") as f:
        sha1 = hashlib.sha1()
        sha1.update(f.read())
        hash1 = sha1.hexdigest()
    # Second write
    pth_out2 = DIR_OUT / (
        f"{Path(__file__).with_suffix('').name}."
        f"repeated_write_gives_same_output_-_write=2.xml{xml_version}.feb"
    )
    # Need to remove time stamp or we won't have same output
    xml = wfl.output.xml(model, version=xml_version)
    tstamp = xml.xpath("//comment()")[0]
    xml.getroot().remove(tstamp)
    with open(pth_out2, "wb") as f:
        wfl.output.write_xml(xml, f)
    with open(pth_out2, "rb") as f:
        sha1 = hashlib.sha1()
        sha1.update(f.read())
        hash2 = sha1.hexdigest()
    # Does write 1 == write 2?
    assert hash1 == hash2
