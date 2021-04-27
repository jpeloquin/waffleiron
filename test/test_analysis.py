# Run these tests with pytest
from unittest import TestCase
import os
import itertools, math
from math import pi, radians, cos, sin
from typing import Generator

import numpy.testing as npt
import numpy as np
import pytest

import febtools as feb
from febtools.control import Step, auto_ticker
from febtools.input import FebReader
from febtools.material import from_Lamé, to_Lamé
from febtools.analysis import *
from febtools import material
from febtools.test.fixtures import (
    DIR_OUT,
    RTOL_F,
    ATOL_F,
    febio_cmd_xml,
    gen_model_center_crack_Hex8,
)


class CenterCrackHex8(TestCase):
    """Center cracked isotropic elastic plate in 3d."""

    def setUp(self):
        self.model, attrib = gen_model_center_crack_Hex8()
        for k in attrib:
            setattr(self, k, attrib[k])

    def _griffith(self):
        """Calculate Griffith strain energy."""
        a = 1.0e-3  # m
        minima = np.min(self.model.mesh.nodes, axis=0)
        maxima = np.max(self.model.mesh.nodes, axis=0)
        width = maxima[0] - minima[0]
        elems_up_down = [
            e
            for e in self.model.mesh.elements
            if (
                np.any(e.nodes[:, 1] == minima[1]) or np.any(e.nodes[:, 1] == maxima[1])
            )
        ]
        pavg = np.mean(
            [e.material.tstress(e.f((0, 0, 0))) for e in elems_up_down], axis=0
        )

        # Define G for plane stress
        K_I = pavg[1][1] * (pi * a * 1.0 / cos(pi * a / width)) ** 0.5
        G = K_I ** 2.0 / self.E
        return G

    def test_right_tip(self):
        # Calculate J

        zslice = feb.select.element_slice(
            self.model.mesh.elements, v=0e-3, axis=(0, 0, 1)
        )
        nodes = [n for e in zslice for n in e.nodes]
        maxima = np.max(nodes, axis=0)
        minima = np.min(nodes, axis=0)
        deltaL = maxima[2] - minima[2]

        # define integration domain
        domain = [e for e in zslice if self.tip_line_r.intersection(e.ids)]
        domain = feb.select.e_grow(domain, zslice, n=2)
        domain = feb.analysis.apply_q_3d(
            domain, self.crack_faces, self.tip_line_r, q=[1, 0, 0]
        )
        # assert len(domain) == 4**2.0 * 4**2.0 * 2

        jbdl = feb.analysis.jintegral(domain)
        jbar = jbdl / (0.5 * deltaL)
        # 0.5 * deltaL is standing in for ∫q(η)dη; this is ok for a
        # tringular q(η)

        # Test if approximately equal to G
        G = self._griffith()
        npt.assert_allclose(jbar, G, rtol=0.07)
        # Test for consistency with value calculated when code
        # initially verified
        npt.assert_allclose(jbar, 73.33, rtol=1e-4)

    def test_left_tip(self):
        """Test if J is valid for left crack tip."""
        zslice = feb.select.element_slice(
            self.model.mesh.elements, v=0e-3, axis=(0, 0, 1)
        )
        nodes = [n for e in zslice for n in e.nodes]
        maxima = np.max(nodes, axis=0)
        minima = np.min(nodes, axis=0)
        deltaL = maxima[2] - minima[2]

        # define integration domain
        domain = [e for e in zslice if self.tip_line_l.intersection(e.ids)]
        domain = feb.select.e_grow(domain, zslice, n=2)
        domain = feb.analysis.apply_q_3d(
            domain, self.crack_faces, self.tip_line_l, q=[-1, 0, 0]
        )
        assert len(domain) == 6 * 6 * 2

        jbdl = feb.analysis.jintegral(domain)
        jbar = jbdl / (0.5 * deltaL)
        # 0.5 * deltaL is standing in for ∫q(η)dη; this is ok for a
        # tringular q(η)

        # Test if approximately equal to G
        G = self._griffith()
        npt.assert_allclose(jbar, G, rtol=0.07)
        # Test for consistency with value calculated when code
        # initially verified
        npt.assert_allclose(jbar, 73.33, rtol=1e-4)

    def test_rotated_right_tip(self):
        """Test if J is the same after a coordinate shift."""
        mesh = self.model.mesh
        G = self._griffith()

        # Create object transform (rotation matrix)
        angle = radians(30)
        R = np.array(
            [[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]]
        )
        # Transform nodes
        mesh.nodes = [np.dot(R, n) for n in mesh.nodes]
        mesh.update_elements()
        # Transform displacements
        for e in mesh.elements:
            e.properties["displacement"] = np.dot(R, e.properties["displacement"].T).T

        # Calculate J
        zslice = feb.select.element_slice(
            self.model.mesh.elements, v=0e-3, axis=(0, 0, 1)
        )
        nodes = [n for e in zslice for n in e.nodes]
        deltaL = np.max(nodes, axis=0)[2] - np.min(nodes, axis=0)[2]

        # Right tip
        domain = [e for e in zslice if self.tip_line_r.intersection(e.ids)]
        domain = feb.select.e_grow(domain, zslice, n=2)
        domain = feb.analysis.apply_q_3d(
            domain, self.crack_faces, self.tip_line_r, q=[cos(angle), sin(angle), 0]
        )
        assert len(domain) == 6 * 6 * 2

        jbar_r = feb.analysis.jintegral(domain) / (0.5 * deltaL)
        npt.assert_allclose(jbar_r, G, rtol=0.07)
        # Test for consistency with value calculated when code
        # initially verified
        npt.assert_allclose(jbar_r, 76.20, rtol=1e-4)

    def test_rotated_left_tip(self):
        """Test if J is the same after a coordinate shift."""
        mesh = self.model.mesh
        G = self._griffith()
        # Geometry (easier before rotation)

        # Create object transform (rotation matrix)
        angle = radians(30)
        R = np.array(
            [[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]]
        )
        # Transform nodes
        mesh.nodes = [np.dot(R, n) for n in mesh.nodes]
        mesh.update_elements()
        # Transform displacements
        for e in mesh.elements:
            e.properties["displacement"] = np.dot(R, e.properties["displacement"].T).T

        # Calculate J
        zslice = feb.select.element_slice(
            self.model.mesh.elements, v=0e-3, axis=(0, 0, 1)
        )

        nodes = [n for e in zslice for n in e.nodes]
        deltaL = np.max(nodes, axis=0)[2] - np.min(nodes, axis=0)[2]

        # define integration domain
        # Right tip
        domain = [e for e in zslice if self.tip_line_l.intersection(e.ids)]
        domain = feb.select.e_grow(domain, zslice, n=2)
        domain = feb.analysis.apply_q_3d(
            domain, self.crack_faces, self.tip_line_l, q=[-cos(angle), -sin(angle), 0]
        )
        assert len(domain) == 6 * 6 * 2

        jbar_l = feb.analysis.jintegral(domain) / (0.5 * deltaL)
        npt.assert_allclose(jbar_l, G, rtol=0.07)
        # Test for consistency with value calculated when code
        # initially verified
        npt.assert_allclose(jbar_l, 76.20, rtol=1e-4)


class CenterCrackQuad4(TestCase):
    def setUp(self):
        reader = feb.input.FebReader(
            os.path.join(
                "test", "fixtures", "center_crack_uniax_isotropic_elastic_quad4.feb"
            )
        )
        self.model = reader.model()
        with open(
            os.path.join(
                "test", "fixtures", "center_crack_uniax_isotropic_elastic_quad4.xplt"
            ),
            "rb",
        ) as f:
            self.soln = feb.xplt.XpltData(f.read())
        self.model.apply_solution(self.soln)

        material = self.model.mesh.elements[0].material
        y = material.y
        mu = material.mu
        E, nu = from_Lamé(y, mu)
        self.E = E
        self.nu = nu

    def test_jintegral(self):
        """Test j integral for Quad4 mesh, isotropic elastic material, small strain."""
        a = 1.0e-3  # m
        W = 10.0e-3  # m
        minima = np.min(self.model.mesh.nodes, axis=0)
        maxima = np.max(self.model.mesh.nodes, axis=0)
        ymin = minima[1]
        ymax = maxima[1]

        def pk1(element_ids):
            """Convert Cauchy stress in each element to 1st P-K."""
            data = self.soln.step_data(-1)
            for i in element_ids:
                t = data["domain variables"]["stress"][i]
                e = self.model.mesh.elements[i]
                f = e.f((0, 0))
                fdet = np.linalg.det(f)
                finv = np.linalg.inv(f)
                P = fdet * np.dot(finv, np.dot(t, finv.T))
                yield P

        e_top = [
            (i, e)
            for i, e in enumerate(self.model.mesh.elements)
            if np.any(np.isclose(e.nodes[:, 1], ymin))
        ]
        e_bot = [
            (i, e)
            for i, e in enumerate(self.model.mesh.elements)
            if np.any(np.isclose(e.nodes[:, 1], ymax))
        ]
        P = list(pk1([i for i, e in e_top + e_bot]))
        Pavg = sum(P) / len(P)
        stress = Pavg[1][1]

        # calculate stress intensity
        K_I = stress * (math.pi * a * 1.0 / math.cos(math.pi * a / W)) ** 0.5
        # Felderson; accurate to 0.3% for a/W ≤ 0.35
        G = K_I ** 2.0 / self.E

        id_crack_tip = [self.model.mesh.find_nearest_nodes(*(1e-3, 0.0, 0.0))[0]]
        elements = apply_q_2d(self.model.mesh, id_crack_tip, n=2, q=[1, 0, 0])
        J = jintegral(elements)
        npt.assert_allclose(J, 72.75, atol=0.01)


@pytest.fixture(scope="module")
def complex_strain_hex8_model(febio_cmd_xml) -> Generator:
    """Return path to cube undergoing complex deformation

    Intended for checking strain gauge functions.

    """
    febio_cmd, xml_version = febio_cmd_xml
    path = (
        DIR_OUT
        / f"test_analysis.complex_strain_hex8_model.{febio_cmd}.xml{xml_version}.feb"
    )
    mat = feb.material.HolmesMow(10, 0.3, 4)
    model = feb.Model(feb.mesh.rectangular_prism((2, 2), (2, 2), (2, 2), material=mat))
    seq = feb.Sequence(((0, 0), (1, 1)), interp="linear", extrap="constant")
    step = Step("solid", ticker=auto_ticker(seq, 10))
    model.add_step(step)
    F = np.array([[1.5, 0.5, 0], [0, 1, 0], [0, 0, 1]])
    left = model.named["node sets"].obj("−x1 face")
    right = model.named["node sets"].obj("+x1 face")
    feb.load.prescribe_deformation(model, step, left, np.eye(3), seq)
    feb.load.prescribe_deformation(model, step, right, F, seq)
    with open(path, "wb") as f:
        feb.output.write_feb(model, f, version=xml_version)
    feb.febio.run_febio_checked(path, cmd=febio_cmd)
    solved = feb.load_model(path)
    yield solved

    # Cleanup
    path.unlink()


def test_strain_gauge_nodesets(complex_strain_hex8_model):
    model = complex_strain_hex8_model
    left = model.named["node sets"].obj("−x1 face")
    right = model.named["node sets"].obj("+x1 face")
    λ = feb.analysis.strain_gauge(model, left, right)
    expected = np.array(
        [
            1.0,
            1.025,
            1.05,
            1.07500001,
            1.1,
            1.125,
            1.15000002,
            1.175,
            1.2,
            1.22499998,
            1.25,
        ]
    )
    npt.assert_allclose(λ, expected, rtol=RTOL_F, atol=ATOL_F)
