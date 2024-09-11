import os
from math import inf
import numpy as np
from numpy import dot
from math import degrees, radians, cos, sin
from unittest import TestCase

import pytest

import waffleiron as wfl
from waffleiron.select import find_closest_timestep, adj_faces, surface_faces


class FindClosestTimestep(TestCase):
    def setUp(self):
        self.times = [0.0, 0.5, 1.0, 1.5, 2.0]
        self.steps = [0, 1, 2, 3, 4]

    def test_in_middle(self):
        assert find_closest_timestep(1.0, self.times, self.steps) == 2

    def test_in_middle_atol_ok(self):
        assert (
            find_closest_timestep(1.2, self.times, self.steps, atol=0.2, rtol=inf) == 2
        )

    def test_in_middle_rtol_ok(self):
        assert (
            find_closest_timestep(0.75, self.times, self.steps, atol=inf, rtol=0.51)
            == 1
        )

    def test_in_middle_atol_bad(self):
        with self.assertRaisesRegex(ValueError, "absolute error > atol"):
            assert find_closest_timestep(0.52, self.times, self.steps, rtol=inf) == 1

    def test_in_middle_rtol_bad(self):
        with self.assertRaisesRegex(ValueError, "relative error > rtol"):
            assert find_closest_timestep(0.52, self.times, self.steps, atol=inf) == 1

    def test_before_start_ok(self):
        assert find_closest_timestep(-0.005, self.times, self.steps, atol=0.005) == 0

    def test_before_start_bad(self):
        with self.assertRaisesRegex(ValueError, "absolute error > atol"):
            assert find_closest_timestep(-0.005, self.times, self.steps) == 0

    def test_at_start(self):
        assert find_closest_timestep(0.0, self.times, self.steps) == 0

    def test_at_end(self):
        assert find_closest_timestep(2.0, self.times, self.steps) == 4

    def test_past_end_ok(self):
        assert (
            find_closest_timestep(2.5, self.times, self.steps, atol=0.51, rtol=1.01)
            == 4
        )

    def test_past_end_bad(self):
        with self.assertRaisesRegex(ValueError, "absolute error > atol"):
            assert find_closest_timestep(2.5, self.times, self.steps) == 4

    def test_nonmatching_values(self):
        times = [0.0, 0.5, 1.0, 1.5, 2.0]
        steps = [0, 1, 2, 3]
        with self.assertRaisesRegex(ValueError, "len\(steps\) ≠ len\(times\)"):
            find_closest_timestep(0.5, times, steps)


class QuadMesh(TestCase):
    """Test selection of corners in rotated mesh."""

    def setUp(self):
        reader = wfl.input.FebReader(
            os.path.join(
                "test", "fixtures", "center_crack_uniax_isotropic_elastic_quad4.feb"
            )
        )
        self.model = reader.model()
        # rotate mesh to make the corner identification a little more
        # difficult
        a = radians(15)
        R = np.array([[cos(a), -sin(a), 0], [sin(a), cos(a), 0], [0, 0, 1]])
        nodes_new = [dot(R, node) for node in self.model.mesh.nodes]
        self.model.mesh.nodes = nodes_new
        self.model.mesh.update_elements()

    def test_select_corners(self):
        """Test for selection of four exterior corner nodes."""
        corner_nodes = wfl.select.corner_nodes(self.model.mesh)
        assert not set(corner_nodes) - set([0, 100, 5554, 5454])


class SelectionHex8Consolidated(TestCase):
    """Test selections for hex8 mesh with center crack."""

    # gradually move SelectionHex8 tests to here

    def setUp(self):
        reader = wfl.input.FebReader(
            os.path.join(
                "test", "fixtures", "center_crack_uniax_isotropic_elastic_hex8.feb"
            )
        )
        self.mesh = reader.mesh()

    def test_bisect_oblique_vector(self):
        """Test bisect with an angled plane."""
        # p1 and p2 define the corners of a triangle on the upper
        # right of the mesh.  The mesh corner makes the third point of
        # the triangle.
        p1 = np.array([0.005, 0.00734127, 0.0])
        p2 = np.array([0.0028879, 0.01, 0.0])
        l = p2 - p1
        n = wfl.geometry.cross(l, (0, 0, 1))
        # Bisect off the elements in the afforementioned triangle
        elset = wfl.select.bisect(self.mesh.elements, p=p1, v=n)
        assert len(elset) == 6 * 4

    def test_element_slice(self):
        # select the two layers in the middle
        eset = wfl.select.element_slice(self.mesh.elements, v=0, axis=(0, 0, 1))
        assert len(eset) == len(self.mesh.elements) / 2


@pytest.mark.skip(reason="too slow")
class SelectionHex8(TestCase):
    """Test selections for a hex8 mesh with a hole."""

    def setUp(self):
        reader = wfl.input.FebReader(
            os.path.join("test", "fixtures", "center_crack_uniax_isotropic_elastic.feb")
        )
        self.mesh = reader.mesh()
        # get a specific face
        faces = self.mesh.faces_with_node(0)
        self.face = (0, 55, 56, 1)

    ### Adjacent faces

    def test_all_adjacency(self):
        faces = adj_faces(self.face, self.mesh, mode="all")
        assert len(faces) == 22
        # make sure the input face is not returned
        assert not any(b == self.face for b in faces)

    def test_edge_adjacency(self):
        """Test edge adjacency."""
        faces = adj_faces(self.face, self.mesh, mode="edge")
        assert len(faces) == 10
        # make sure the input face is not returned
        assert not any(b == self.face for b in faces)

    def test_full_adjacency(self):
        """Test full adjacency (superimposition).

        Since the selected face is on the surface, there should be no
        superimposed faces.

        """
        faces = adj_faces(self.face, self.mesh, mode="face")
        assert len(faces) == 0
        # make sure the input face is not returned
        assert self.face not in faces

    ### Surface faces

    def test_surface_faces(self):
        """Test selection of surface faces."""
        faces = surface_faces(self.mesh)
        assert len(faces) == 6756
