# Python built-ins
from math import radians
from unittest import TestCase
# Public modules
import numpy as np
import numpy.testing as npt
# febtools' local modules
from febtools.febioxml import basis_mat_axis_local
from febtools.element import Hex8
from febtools.model import Model, Mesh
from febtools.material import IsotropicElastic
from febtools.output import write_feb


class MatAxisLocalHex8(TestCase):
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
        npt.assert_allclose(basis[0], e1)
        assert basis[1] @ (self.nodes[3] - self.nodes[0]) > 0.71
        assert basis[2] @ (self.nodes[5] - self.nodes[0]) > 0.71
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
        npt.assert_allclose(basis[0], e1)
        assert basis[1] @ (self.nodes[4] - self.nodes[0]) > 0.71
        assert basis[2] @ (self.nodes[0] - self.nodes[3]) > 0.71
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
        npt.assert_allclose(basis[0], e1)
        assert basis[1] @ [0, 0, -1] > 0.71
        assert basis[2] @ [1, 0, 0] > 0.71
        # Demonstrate that the basis is orthonormal
        npt.assert_allclose(np.array(basis) @ np.array(basis).T, np.eye(3),
                            atol=np.finfo(np.array(basis).dtype).resolution)
