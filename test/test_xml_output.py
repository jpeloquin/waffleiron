# Run these tests with nose
import hashlib
from pathlib import Path
from nose.tools import with_setup
import febtools as feb
from febtools import febioxml_2_5 as febioxml
from febtools.test.fixtures import DIR_FIXTURES, DIR_OUT, xml_version


# Environmental constants


def test_write_temperature_constant(xml_version):
    mat = feb.material.HolmesMow(1.5, 0.3, 2.0)
    mesh = feb.mesh.rectangular_prism((1, 2), (1, 2), (1, 2), material=mat)
    model = feb.Model(mesh)
    model.environment["temperature"] = 341
    xml = feb.output.xml(model, version=xml_version)
    assert xml.find("Globals/Constants/T").text == "341"


# Universal constants


def test_write_ideal_gas_constant(xml_version):
    mat = feb.material.HolmesMow(1.5, 0.3, 2.0)
    mesh = feb.mesh.rectangular_prism((1, 2), (1, 2), (1, 2), material=mat)
    model = feb.Model(mesh)
    model.constants["R"] = 8.314  # J/mol·K
    xml = feb.output.xml(model, version=xml_version)
    assert xml.find("Globals/Constants/R").text == "8.314"


def test_write_Faraday_constant(xml_version):
    mat = feb.material.HolmesMow(1.5, 0.3, 2.0)
    mesh = feb.mesh.rectangular_prism((1, 2), (1, 2), (1, 2), material=mat)
    model = feb.Model(mesh)
    model.constants["F"] = 26.801  # A·h/mol
    xml = feb.output.xml(model, version=xml_version)
    assert xml.find("Globals/Constants/Fc").text == "26.801"


# Test for mutation


def test_repeated_write_gives_same_output(xml_version):
    """Test repeated write of same model file"""
    pth_in = DIR_FIXTURES / "uniaxial_testion_implicit_rb_grips_biphasic.feb"
    model = feb.load_model(pth_in)
    # First write
    print("Write 1")
    pth_out1 = DIR_OUT / (
        f"{Path(__file__).with_suffix('').name}."
        "repeated_write_gives_same_output_-_write=1.xml{xml_version}.feb"
    )
    with open(pth_out1, "wb") as f:
        feb.output.write_feb(model, f, version=xml_version)
    with open(pth_out1, "rb") as f:
        sha1 = hashlib.sha1()
        sha1.update(f.read())
        hash1 = sha1.hexdigest()
    # Second write
    print("Write 2")
    pth_out2 = DIR_OUT / (
        f"{Path(__file__).with_suffix('').name}."
        "repeated_write_gives_same_output_-_write=2.xml{xml_version}.feb"
    )
    with open(pth_out2, "wb") as f:
        feb.output.write_feb(model, f, version=xml_version)
    with open(pth_out2, "rb") as f:
        sha1 = hashlib.sha1()
        sha1.update(f.read())
        hash2 = sha1.hexdigest()
    # Does write 1 == write 2?
    assert hash1 == hash2
