# Run these tests with nose
import hashlib
from pathlib import Path
import unittest
from nose.tools import with_setup
import febtools as feb
from febtools import febioxml_2_5 as febioxml
from febtools.test.fixtures import DIR_FIXTURES, DIR_OUT


class EnvironmentConstants(unittest.TestCase):
    """Test write of environmental constants."""

    def test_temperature_constant(self):
        mesh = feb.mesh.rectangular_prism((1, 2), (1, 2), (1, 2))
        model = feb.Model(mesh)
        model.environment["temperature"] = 341
        xml = feb.output.xml(model)
        assert xml.find("Globals/Constants/T").text == "341"


class UniversalConstants(unittest.TestCase):
    """Test write of universal constants."""

    def test_ideal_gas_constant(self):
        mesh = feb.mesh.rectangular_prism((1, 2), (1, 2), (1, 2))
        model = feb.Model(mesh)
        model.constants["R"] = 8.314  # J/mol·K
        xml = feb.output.xml(model)
        assert xml.find("Globals/Constants/R").text == "8.314"

    def test_Faraday_constant(self):
        mesh = feb.mesh.rectangular_prism((1, 2), (1, 2), (1, 2))
        model = feb.Model(mesh)
        model.constants["F"] = 26.801  # A·h/mol
        xml = feb.output.xml(model)
        assert xml.find("Globals/Constants/Fc").text == "26.801"


def test_repeated_write_gives_same_output():
    """Test repeated write of same model file"""
    pth_in = DIR_FIXTURES / "uniaxial_testion_implicit_rb_grips_biphasic.feb"
    model = feb.load_model(pth_in)
    # First write
    print("Write 1")
    pth_out1 = DIR_OUT / (
        f"{Path(__file__).with_suffix('').name}."
        "repeated_write_gives_same_output_-_write=1.feb"
    )
    with open(pth_out1, "wb") as f:
        feb.output.write_feb(model, f)
    with open(pth_out1, "rb") as f:
        sha1 = hashlib.sha1()
        sha1.update(f.read())
        hash1 = sha1.hexdigest()
    # Second write
    print("Write 2")
    pth_out2 = DIR_OUT / (
        f"{Path(__file__).with_suffix('').name}."
        "repeated_write_gives_same_output_-_write=2.feb"
    )
    with open(pth_out2, "wb") as f:
        feb.output.write_feb(model, f)
    with open(pth_out2, "rb") as f:
        sha1 = hashlib.sha1()
        sha1.update(f.read())
        hash2 = sha1.hexdigest()
    # Does write 1 == write 2?
    assert hash1 == hash2
