# Run these tests with pytest
import unittest
from pathlib import Path

import numpy as np

import waffleiron as wfl
from copy import deepcopy

from waffleiron import Model, load_model
from waffleiron.mesh import rectangular_prism_hex27
from waffleiron.output import write_feb
from waffleiron.test.fixtures import DIR_OUT


class MergeTestTri2d(unittest.TestCase):
    """Test merging meshes"""

    def setUp(self):
        """Create two meshes sharing a common boundary."""
        nodes1 = [
            (0.0, 0.0),
            (0.3, 0.0),
            (0.6, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ]
        nodes2 = [(x, -1.0 * y) for (x, y) in nodes1]
        elements = [(0, 1, 5), (1, 4, 5), (1, 2, 4), (2, 3, 4)]
        self.mesh1 = wfl.Mesh.from_ids(nodes1, elements, wfl.element.Tri3)
        self.mesh2 = wfl.Mesh.from_ids(nodes2, elements, wfl.element.Tri3)

    def test_mesh_merge_all(self):
        """Test merging two meshes, merging all common nodes."""
        mesh1 = deepcopy(self.mesh1)
        mesh2 = deepcopy(self.mesh2)
        mesh1.merge(mesh2)
        assert len(mesh1.elements) == 8
        assert len(mesh1.nodes) == 8

    def test_mesh_merge_sel(self):
        """Test merging two meshes, merging only selected nodes."""
        pass


def test_rectangular_prism_hex27() -> None:
    # Does a single-element mesh work?
    mesh = rectangular_prism_hex27((1, 1, 1))
    # Does a 2×2×2 element mesh work?
    mesh = rectangular_prism_hex27((2, 2, 2))
    # Are the axes and dimensions correct?
    mesh = rectangular_prism_hex27((2, 2, 2), ((0, 1), (0, 2), (0, 3)))
    assert np.allclose(np.max(mesh.nodes, axis=0), (1, 2, 3))
    # Is node ordering correct?
    model = Model(
        rectangular_prism_hex27((1, 1, 1), [(-0.5, 0.5), (-0.5, 0.5), (0, 1)])
    )
    good_nodes = np.array(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [-0.5, -0.5, 1.0],
            [0.5, -0.5, 1.0],
            [0.5, 0.5, 1.0],
            [-0.5, 0.5, 1.0],
            [0.0, -0.5, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [-0.5, 0.0, 0.0],
            [0.0, -0.5, 1.0],
            [0.5, 0.0, 1.0],
            [0.0, 0.5, 1.0],
            [-0.5, 0.0, 1.0],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [0.0, -0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [-0.5, 0.0, 0.5],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.5],
        ]
    )
    test_nodes = np.array([model.mesh.nodes[i] for i in model.mesh.elements[0].ids])
    assert np.allclose(good_nodes, test_nodes)
