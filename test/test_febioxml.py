# Python built-ins
from math import radians
from pathlib import Path
from unittest import TestCase
# Public modules
import numpy as np
import numpy.testing as npt
# febtools' local modules
import febtools as feb
from febtools.load import prescribe_deformation
from febtools.control import auto_control_section
from febtools.febioxml import basis_mat_axis_local
from febtools.element import Hex8
from febtools.model import Model, Mesh
from febtools.material import IsotropicElastic, OrthotropicElastic
from febtools.output import write_feb, write_xml
from febtools.test.fixtures import gen_model_single_spiky_Hex8


DIR_THIS = Path(__file__).parent
DIR_OUTPUT = DIR_THIS / "output"
DIR_FIXTURES = Path(__file__).parent / "fixtures"


def test_FEBio_normalizeXML_bare_Boundary():
    """End-to-end test of Boundary/prescribe normalization."""
    pth_in = DIR_FIXTURES / \
        (f"{Path(__file__).with_suffix('').name}." +\
         "normalizeXML_bare_Boundary.feb")
    tree = feb.input.read_febio_xml(pth_in)
    normalized_root = feb.febioxml.normalize_xml(tree.getroot())
    normalized_tree = normalized_root.getroottree()
    pth_out = DIR_OUTPUT / \
        (f"{Path(__file__).with_suffix('').name}." +\
         "normalizeXML_bare_Boundary.feb")
    with open(pth_out, "wb") as f:
        feb.output.write_xml(normalized_tree, f)
    # Test 1: Can FEBio still read the normalized file?
    feb.febio.run_febio(pth_out)
    # Test 2: Did the right displacements get applied?
    model = feb.load_model(pth_out)
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
        x2 = [np.cos(radians(17))*np.cos(radians(6)),
              np.sin(radians(17))*np.cos(radians(6)),
              np.sin(radians(6))]
        x3 = np.array([0.8348, 0.9758, 0.3460])
        x4 = np.array([0.0794, 0.9076, 0.1564])
        x5 = x1 + np.array([0.638*np.cos(radians(26))*np.sin(radians(1)),
                            0.638*np.sin(radians(26))*np.sin(radians(1)),
                            np.cos(radians(1))])
        x6 = x5 + np.array([0.71*np.cos(radians(-24))*np.cos(radians(-7)),
                            0.71*np.sin(radians(-24))*np.cos(radians(-7)),
                            np.sin(radians(-7))])
        x7 = [1, 1, 1]
        x8 = x5 + [np.sin(radians(9))*np.cos(radians(-11)),
                   np.cos(radians(9))*np.cos(radians(-11)),
                   np.sin(radians(-11))]
        self.nodes = np.vstack([x1, x2, x3, x4, x5, x6, x7, x8])
        self.element = Hex8.from_ids([i for i in range(8)], self.nodes)

    def test_local_default(self):
        """Test default local basis 1,2,4."""
        basis = basis_mat_axis_local(self.element)
        e1 = (self.nodes[1] - self.nodes[0]) / np.linalg.norm(self.nodes[1] - self.nodes[0])
        # Check basis vector directions.  The first vector can be
        # checked exactly; the others can only be checked roughly—an
        # exact check would just be a re-implementation of the function
        # under test.
        npt.assert_allclose(basis[:,0], e1)
        assert basis[:,1] @ (self.nodes[3] - self.nodes[0]) > 0.71
        assert basis[:,2] @ (self.nodes[5] - self.nodes[0]) > 0.71
        # Demonstrate that the basis is orthonormal
        npt.assert_allclose(np.array(basis) @ np.array(basis).T, np.eye(3),
                            atol=np.finfo(np.array(basis).dtype).resolution)

    def test_local_000(self):
        """Test FEBio special case of local basis 0,0,0 → 1,2,4."""
        basis_000 = np.array(basis_mat_axis_local(self.element, (0, 0, 0)))
        basis_124 = np.array(basis_mat_axis_local(self.element, (1, 2, 4)))
        npt.assert_equal(basis_000, basis_124)

    def test_local_125(self):
        """Test local basis 1,2,5."""
        basis = basis_mat_axis_local(self.element, (1, 2, 5))
        e1 = (self.nodes[1] - self.nodes[0]) / np.linalg.norm(self.nodes[1] - self.nodes[0])
        # Check basis vector directions.  The first vector can
        # be checked exactly.  The other two can only be checked
        # roughly, as an exact check would just be a re-implementation
        # of the function under test.
        npt.assert_allclose(basis[:,0], e1)
        assert basis[:,1] @ (self.nodes[4] - self.nodes[0]) > 0.71
        assert basis[:,2] @ (self.nodes[0] - self.nodes[3]) > 0.71
        # Demonstrate that the basis is orthonormal
        npt.assert_allclose(np.array(basis) @ np.array(basis).T, np.eye(3),
                            atol=np.finfo(np.array(basis).dtype).resolution)

    def test_local_762(self):
        """Test local basis 7,6,2."""
        basis = basis_mat_axis_local(self.element, (7, 6, 2))
        e1 = (self.nodes[5] - self.nodes[6]) / np.linalg.norm(self.nodes[5] - self.nodes[6])
        # Check basis vector directions.  The first vector can be
        # checked exactly.  The other two can only be checked roughly,
        # as an exact check would just be a re-implementation of the
        # function under test.
        npt.assert_allclose(basis[:,0], e1)
        assert basis[:,1] @ [0, 0, -1] > 0.71
        assert basis[:,2] @ [1, 0, 0] > 0.71
        # Demonstrate that the basis is orthonormal
        npt.assert_allclose(np.array(basis) @ np.array(basis).T, np.eye(3),
                            atol=np.finfo(np.array(basis).dtype).resolution)


class FEBio_MatAxisLocal_Hex8(TestCase):
    """Test consistency of <mat_axis type="local"> interpretation with FEBio.

    Create two models using the same irregularly shaped element, each
    with an orthotropic material.

    To model 1: Assign a local element basis (<mat axis type="local">).
    Apply a deformation consistent with a chosen deformation gradient
    tensor F.

    To model 2: Assign a global basis that matches the local element
    basis in model 1.  Apply the same deformation gradient tensor F.

    Solve both models with FEBio and read the output stress.  If the
    local element basis in model 1 was interpreted correctly, the
    stresses in both models should match.

    """
    def setUp(self):
        material = OrthotropicElastic(\
            {"E1": 23,
             "E2": 81,
             "E3": 50,
             "G12": 15,
             "G23": 82,
             "G31": 5,
             "ν12": 0.26,
             "ν23": -0.36,
             "ν31": 0.2})
        F = np.array([[0.734, -0.10, -0.02],
                      [-0.01,  0.953, 0.09],
                      [-0.23,  0.10,  1.24]])
        basis_code = (4, 7, 3)
        #
        # Model 1: Local basis; material axes given by <mat_axis type="local">
        localb_model = gen_model_single_spiky_Hex8(material=material)
        sequence = feb.Sequence(((0, 0), (1, 1)),
                                extend="extrapolate", typ="linear")
        localb_model.add_step(control=auto_control_section(sequence, pts_per_segment=1))
        node_set = [i for i in range(len(localb_model.mesh.nodes))]
        prescribe_deformation(localb_model, node_set, F, sequence)
        self.pth_localb_model = DIR_THIS / "output" /\
            "FEBio_MatAxisLocal_Hex8_localb.feb"
        tree = feb.output.xml(localb_model)
        # add <mat_axis type="local">
        e_mat = tree.find("Material/material")
        e_mat_axis = e_mat.makeelement("mat_axis")
        e_mat.append(e_mat_axis)
        e_mat_axis.attrib["type"] = "local"
        e_mat_axis.text = ", ".join([str(a) for a in basis_code])
        with open(self.pth_localb_model, "wb") as f:
            write_xml(tree, f)
        feb.febio.run_febio(self.pth_localb_model)
        #
        # Model 2: Global basis; material axes are x1, x2, x3
        basis = np.array(basis_mat_axis_local(localb_model.mesh.elements[0], basis_code))
        # ^ dim 0 over basis vectors, dim 1 over X
        globalb_model = gen_model_single_spiky_Hex8(material=material)
        sequence = feb.Sequence(((0, 0), (1, 1)),
                                extend="extrapolate", typ="linear")
        globalb_model.add_step(control=auto_control_section(sequence, pts_per_segment=1))
        node_set = [i for i in range(len(globalb_model.mesh.nodes))]
        prescribe_deformation(globalb_model, node_set, F, sequence)
        self.pth_globalb_model = DIR_THIS / "output" /\
            "FEBio_MatAxisLocal_Hex8_globalb.feb"
        tree = feb.output.xml(localb_model)
        # add <mat_axis type="vector">
        e_mat = tree.find("Material/material")
        e_mat_axis = e_mat.makeelement("mat_axis")
        e_mat_axis.attrib["type"] = "vector"
        e_mat.append(e_mat_axis)
        e_a = e_mat_axis.makeelement("a")
        e_mat_axis.append(e_a)
        e_a.text = ", ".join([str(a) for a in basis[:,0]])
        e_d = e_mat_axis.makeelement("d")
        e_mat_axis.append(e_d)
        e_d = tree.find("Material/material/mat_axis/d")
        e_d.text = ", ".join([str(a) for a in basis[:,1]])
        with open(self.pth_globalb_model, "wb") as f:
            write_xml(tree, f)
        feb.febio.run_febio(self.pth_globalb_model)


    def test_compare_stress(self):
        with open(self.pth_localb_model.with_suffix(".xplt"), "rb") as f:
            localb_result = feb.xplt.XpltData(f.read())
        localb_stress = localb_result.step_data(-1)["domain variables"]["stress"][0]
        with open(self.pth_globalb_model.with_suffix(".xplt"), "rb") as f:
            globalb_result = feb.xplt.XpltData(f.read())
        globalb_stress = globalb_result.step_data(-1)["domain variables"]["stress"][0]
        npt.assert_array_almost_equal_nulp(globalb_stress, localb_stress)


    def tearDown(self):
        for ext in (".feb", ".log", ".xplt"):
            self.pth_localb_model.with_suffix(ext).unlink()
            self.pth_globalb_model.with_suffix(ext).unlink()
