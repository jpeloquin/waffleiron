import febtools as feb
import unittest, os
import numpy as np
from numpy import dot
from math import degrees, radians, cos, sin

class QuadMesh(unittest.TestCase):
    """Quadrilateral mesh (2D case) with a hole.

    """
    
    def setUp(self):
        reader = feb.input.FebReader(os.path.join('test', 'fixtures', 'uniax-2d-center-crack-1mm.feb'))
        self.model = reader.model()
        # rotate mesh to make the corner identification a little more
        # difficult
        a = radians(15)
        R = np.array([[cos(a), -sin(a), 0],
                      [sin(a), cos(a), 0],
                      [0, 0, 1]])
        nodes_new = [dot(R, node) for node in self.model.mesh.nodes]
        self.model.mesh.nodes = nodes_new
        self.model.mesh.update_elements()
            

    def test_select_corners(self):
        """Test for selection of four exterior corner nodes.

        """
        corner_nodes = feb.selection.corner_nodes(self.model.mesh)
        assert not set(corner_nodes) - set([0, 100, 5554, 5454])


from febtools.selection import adj_faces, surface_faces

class SelectionHex8(unittest.TestCase):
    """Test for correct face connectivity.

    """
    def setUp(self):
        reader = feb.input.FebReader(os.path.join('test', 'fixtures', 'center_crack_uniax_isotropic_elastic.feb'))
        self.mesh = reader.mesh()
        # get a specific face
        faces = self.mesh.faces_with_node[0]
        nids = set((0, 1, 55, 56))
        self.face = next(f for f in faces
                         if not set(f.ids) - nids)

    ### Adjacent faces

    def test_all_adjacency(self):
        faces = adj_faces(self.mesh, self.face, mode='all')
        assert len(faces) == 10
        # make sure the input face is not returned
        assert self.face not in faces

    def test_edge_adjacency(self):
        """Test edge adjacency.

        """
        faces = adj_faces(self.mesh, self.face, mode='edge')
        assert len(faces) == 10
        # make sure the input face is not returned
        assert self.face not in faces

    def test_full_adjacency(self):
        """Test full adjacency (superimposition).

        Since the selected face is on the surface, there should be no
        superimposed faces.

        """
        faces = adj_faces(self.mesh, self.face, mode='face')
        assert len(faces) == 0
        # make sure the input face is not returned
        assert self.face not in faces

    ### Surface faces

    def test_surface_faces(self):
        """Test selection of surface faces.

        """
        faces = surface_faces(self.mesh)
        assert len(faces) == 6756
