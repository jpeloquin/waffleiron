# Run these tests with nose
import unittest
import os
from nose.tools import with_setup
import febtools as feb
from febtools import febioxml_2_5 as febioxml

class EnvironmentConstants(unittest.TestCase):
    """Test write of environmental constants."""

    def test_temperature_constant(self):
        mesh = feb.mesh.hexa.rectangular_prism(1, 1, 1, 0.5)
        model = feb.Model(mesh)
        model.environment["temperature"] = 341
        xml = feb.output.xml(model)
        assert(xml.find("Globals/Constants/T").text == "341")
        mesh = feb.mesh.hexa.rectangular_prism(1, 1, 1, 0.5)
        model = feb.Model(mesh)
        mat = feb.material.IsotropicElastic({"E": 1, "v": .3})
        model.environment["temperature"] = 341
        xml = feb.output.xml(model)
        assert(xml.find("Globals/Constants/T").text == "341")
