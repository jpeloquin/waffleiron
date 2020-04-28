# Run these tests with nose
import unittest
import os
from nose.tools import with_setup
import febtools as feb
from febtools import febioxml_2_5 as febioxml

class EnvironmentConstants(unittest.TestCase):
    """Test write of environmental constants."""

    def test_temperature_constant(self):
        mesh = feb.mesh.rectangular_prism(1, 1, 1, 0.5)
        model = feb.Model(mesh)
        model.environment["temperature"] = 341
        xml = feb.output.xml(model)
        assert(xml.find("Globals/Constants/T").text == "341")


class UniversalConstants(unittest.TestCase):
    """Test write of universal constants."""

    def test_ideal_gas_constant(self):
        mesh = feb.mesh.rectangular_prism(1, 1, 1, 0.5)
        model = feb.Model(mesh)
        model.constants["R"] = 8.314  # J/mol·K
        xml = feb.output.xml(model)
        assert(xml.find("Globals/Constants/R").text == "8.314")

    def test_Faraday_constant(self):
        mesh = feb.mesh.rectangular_prism(1, 1, 1, 0.5)
        model = feb.Model(mesh)
        model.constants["F"] = 26.801  # A·h/mol
        xml = feb.output.xml(model)
        assert(xml.find("Globals/Constants/Fc").text == "26.801")
