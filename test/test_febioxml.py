# Python built-ins
from math import radians
from pathlib import Path
from typing import Generator
from unittest import TestCase

# Public modules
import numpy as np
import numpy.testing as npt
import pytest

# waffleiron' local modules
import waffleiron as wfl
from waffleiron import Step
from waffleiron.load import prescribe_deformation
from waffleiron.control import auto_ticker
from waffleiron.febioxml import basis_mat_axis_local
from waffleiron.element import Hex8
from waffleiron.model import Model, Mesh
from waffleiron.material import IsotropicElastic, OrthotropicLinearElastic
from waffleiron.output import write_feb, write_xml
from waffleiron.test.fixtures import (
    DIR_OUT,
    febio_cmd,
    febio_cmd_xml,
    gen_model_single_spiky_Hex8,
)


DIR_THIS = Path(__file__).parent
DIR_FIXTURES = Path(__file__).parent / "fixtures"


def test_FEBio_normalizeXML_bare_Boundary(febio_cmd):
    """End-to-end test of Boundary/prescribe normalization."""
    # This is an FEBio XML 2.5 file, so normalization of FEBio XML 3.0
    # is not tested
    pth_in = DIR_FIXTURES / (
        f"{Path(__file__).with_suffix('').name}." + "normalizeXML_bare_Boundary.feb"
    )
    tree = wfl.input.read_febio_xml(pth_in)
    normalized_root = wfl.febioxml.normalize_xml(tree.getroot())
    normalized_tree = normalized_root.getroottree()
    pth_out = DIR_OUT / (
        f"{Path(__file__).with_suffix('').name}."
        + f"normalizeXML_bare_Boundary.{febio_cmd}.feb"
    )
    with open(pth_out, "wb") as f:
        wfl.output.write_xml(normalized_tree, f)
    # Test 1: Can FEBio still read the normalized file?
    wfl.febio.run_febio_checked(pth_out, cmd=febio_cmd, threads=1)
    # Test 2: Did the right displacements get applied?
    model = wfl.load_model(pth_out)
    ## Test 2.1: Did node 0 stay fixed?
    expected_Δx_node0 = (0, 0, 0)
    Δx_node0 = model.solution.value("displacement", step=1, entity_id=0)
    npt.assert_array_almost_equal_nulp(Δx_node0, expected_Δx_node0, nulp=1)
    ## Test 2.2: Did node 6 move the applied displacement?
    expected_Δx_node6 = (0.5, 0, 0)
    Δx_node6 = model.solution.value("displacement", step=1, entity_id=6)
    npt.assert_allclose(Δx_node6, expected_Δx_node6, atol=1e-15)


class Unit_MatAxisLocal_Hex8(TestCase):
    """Test <mat_axis type="local"> → basis for a Hex8 element"""

    def setUp(self):
        # Create an irregularly shaped element in which no two edges are
        # parallel.
        x1 = np.array([0, 0, 0])
        x2 = [
            np.cos(radians(17)) * np.cos(radians(6)),
            np.sin(radians(17)) * np.cos(radians(6)),
            np.sin(radians(6)),
        ]
        x3 = np.array([0.8348, 0.9758, 0.3460])
        x4 = np.array([0.0794, 0.9076, 0.1564])
        x5 = x1 + np.array(
            [
                0.638 * np.cos(radians(26)) * np.sin(radians(1)),
                0.638 * np.sin(radians(26)) * np.sin(radians(1)),
                np.cos(radians(1)),
            ]
        )
        x6 = x5 + np.array(
            [
                0.71 * np.cos(radians(-24)) * np.cos(radians(-7)),
                0.71 * np.sin(radians(-24)) * np.cos(radians(-7)),
                np.sin(radians(-7)),
            ]
        )
        x7 = [1, 1, 1]
        x8 = x5 + [
            np.sin(radians(9)) * np.cos(radians(-11)),
            np.cos(radians(9)) * np.cos(radians(-11)),
            np.sin(radians(-11)),
        ]
        self.nodes = np.vstack([x1, x2, x3, x4, x5, x6, x7, x8])
        self.element = Hex8.from_ids([i for i in range(8)], self.nodes)

    def test_local_default(self):
        """Test default local basis 1,2,4."""
        basis = basis_mat_axis_local(self.element)
        e1 = (self.nodes[1] - self.nodes[0]) / np.linalg.norm(
            self.nodes[1] - self.nodes[0]
        )
        # Check basis vector directions.  The first vector can be
        # checked exactly; the others can only be checked roughly—an
        # exact check would just be a re-implementation of the function
        # under test.
        npt.assert_allclose(basis[:, 0], e1)
        assert basis[:, 1] @ (self.nodes[3] - self.nodes[0]) > 0.71
        assert basis[:, 2] @ (self.nodes[5] - self.nodes[0]) > 0.71
        # Demonstrate that the basis is orthonormal
        npt.assert_allclose(
            np.array(basis) @ np.array(basis).T,
            np.eye(3),
            atol=np.finfo(np.array(basis).dtype).resolution,
        )

    def test_local_000(self):
        """Test FEBio special case of local basis 0,0,0 → 1,2,4."""
        basis_000 = np.array(basis_mat_axis_local(self.element, (0, 0, 0)))
        basis_124 = np.array(basis_mat_axis_local(self.element, (1, 2, 4)))
        npt.assert_equal(basis_000, basis_124)

    def test_local_125(self):
        """Test local basis 1,2,5."""
        basis = basis_mat_axis_local(self.element, (1, 2, 5))
        e1 = (self.nodes[1] - self.nodes[0]) / np.linalg.norm(
            self.nodes[1] - self.nodes[0]
        )
        # Check basis vector directions.  The first vector can
        # be checked exactly.  The other two can only be checked
        # roughly, as an exact check would just be a re-implementation
        # of the function under test.
        npt.assert_allclose(basis[:, 0], e1)
        assert basis[:, 1] @ (self.nodes[4] - self.nodes[0]) > 0.71
        assert basis[:, 2] @ (self.nodes[0] - self.nodes[3]) > 0.71
        # Demonstrate that the basis is orthonormal
        npt.assert_allclose(
            np.array(basis) @ np.array(basis).T,
            np.eye(3),
            atol=np.finfo(np.array(basis).dtype).resolution,
        )

    def test_local_762(self):
        """Test local basis 7,6,2."""
        basis = basis_mat_axis_local(self.element, (7, 6, 2))
        e1 = (self.nodes[5] - self.nodes[6]) / np.linalg.norm(
            self.nodes[5] - self.nodes[6]
        )
        # Check basis vector directions.  The first vector can be
        # checked exactly.  The other two can only be checked roughly,
        # as an exact check would just be a re-implementation of the
        # function under test.
        npt.assert_allclose(basis[:, 0], e1)
        assert basis[:, 1] @ [0, 0, -1] > 0.71
        assert basis[:, 2] @ [1, 0, 0] > 0.71
        # Demonstrate that the basis is orthonormal
        npt.assert_allclose(
            np.array(basis) @ np.array(basis).T,
            np.eye(3),
            atol=np.finfo(np.array(basis).dtype).resolution,
        )


@pytest.fixture(scope="module")
def mataxis_local_global_hex8_models(febio_cmd_xml) -> Generator:
    """Create models with equivalent local and global element bases

    Create two models using the same irregularly shaped element, each
    with an orthotropic material.

    To model 1: Assign a local element basis (<mat axis type="local">).
    Apply a deformation consistent with a chosen deformation gradient
    tensor F.

    To model 2: Assign a global basis that matches the local element
    basis in model 1.  Apply the same deformation gradient tensor F.

    Solve both models with FEBio and return the model paths.  This
    fixture is intended to support testing the consistency of <mat_axis
    type="local"> and global material axes.

    """
    febio_cmd, xml_version = febio_cmd_xml
    material = OrthotropicLinearElastic(
        {
            "E1": 23,
            "E2": 81,
            "E3": 50,
            "G12": 15,
            "G23": 82,
            "G31": 5,
            "ν12": 0.26,
            "ν23": -0.36,
            "ν31": 0.2,
        }
    )
    F = np.array([[0.734, -0.10, -0.02], [-0.01, 0.953, 0.09], [-0.23, 0.10, 1.24]])
    basis_code = (4, 7, 3)
    #
    # Model 1: Local basis; material axes given by <mat_axis type="local">
    localb_model = gen_model_single_spiky_Hex8(material=material)
    sequence = wfl.Sequence(((0, 0), (1, 1)), extrap="linear", interp="linear")
    step = Step(physics="solid", dynamics="static", ticker=auto_ticker(sequence))
    localb_model.add_step(step)
    node_set = [i for i in range(len(localb_model.mesh.nodes))]
    prescribe_deformation(localb_model, step, node_set, F, sequence)
    pth_localb_model = (
        DIR_OUT
        / f"mataxis_local_global_hex8_models.localb.{febio_cmd}.xml{xml_version}.feb"
    )
    tree = wfl.output.xml(localb_model, version=xml_version)
    # add <mat_axis type="local">
    e_mat = tree.find("Material/material")
    e_mat_axis = e_mat.makeelement("mat_axis")
    e_mat.append(e_mat_axis)
    e_mat_axis.attrib["type"] = "local"
    e_mat_axis.text = ", ".join([str(a) for a in basis_code])
    with open(pth_localb_model, "wb") as f:
        write_xml(tree, f)
    wfl.febio.run_febio_checked(pth_localb_model, cmd=febio_cmd, threads=1)
    #
    # Model 2: Global basis; material axes are x1, x2, x3
    basis = np.array(basis_mat_axis_local(localb_model.mesh.elements[0], basis_code))
    # ^ dim 0 over basis vectors, dim 1 over X
    globalb_model = gen_model_single_spiky_Hex8(material=material)
    seq = wfl.Sequence(((0, 0), (1, 1)), extrap="linear", interp="linear")
    step = Step("solid", dynamics="static", ticker=auto_ticker(seq))
    globalb_model.add_step(step)
    node_set = [i for i in range(len(globalb_model.mesh.nodes))]
    prescribe_deformation(globalb_model, step, node_set, F, sequence)
    pth_globalb_model = (
        DIR_OUT
        / f"mataxis_local_global_hex8_models.globalb.{febio_cmd}.xml{xml_version}..feb"
    )
    tree = wfl.output.xml(localb_model, version=xml_version)
    # add <mat_axis type="vector">
    e_mat = tree.find("Material/material")
    e_mat_axis = e_mat.makeelement("mat_axis")
    e_mat_axis.attrib["type"] = "vector"
    e_mat.append(e_mat_axis)
    e_a = e_mat_axis.makeelement("a")
    e_mat_axis.append(e_a)
    e_a.text = ", ".join([str(a) for a in basis[:, 0]])
    e_d = e_mat_axis.makeelement("d")
    e_mat_axis.append(e_d)
    e_d = tree.find("Material/material/mat_axis/d")
    e_d.text = ", ".join([str(a) for a in basis[:, 1]])
    with open(pth_globalb_model, "wb") as f:
        write_xml(tree, f)
    wfl.febio.run_febio_checked(pth_globalb_model, cmd=febio_cmd, threads=1)

    yield (pth_localb_model, pth_globalb_model)

    # Cleanup
    for ext in (".feb", ".log", ".xplt"):
        pth_localb_model.with_suffix(ext).unlink()
        pth_globalb_model.with_suffix(ext).unlink()


def test_mataxis_stress(mataxis_local_global_hex8_models) -> None:
    """Test consistency of local and global mat_axis with FEBio.

    Compare stress between two models, one with mat_axis set locally and
    one with mat_axis set globally.  If the local element basis in model
    1 was interpreted correctly, the stresses in both models should
    match.

    """
    pth_localb_model, pth_globalb_model = mataxis_local_global_hex8_models
    with open(pth_localb_model.with_suffix(".xplt"), "rb") as f:
        localb_result = wfl.xplt.XpltData(f.read())
    localb_stress = localb_result.step_data(-1)[("stress", "domain")][0]
    with open(pth_globalb_model.with_suffix(".xplt"), "rb") as f:
        globalb_result = wfl.xplt.XpltData(f.read())
    globalb_stress = globalb_result.step_data(-1)[("stress", "domain")][0]
    npt.assert_array_almost_equal_nulp(globalb_stress, localb_stress)
