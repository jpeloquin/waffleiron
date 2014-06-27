# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import fsolve

def f(r, X, u):
    """Calculate F tensor from nodal values and shape functions.

    r = target coordinates in natural basis (tuple)
    X = nodal coordinates in reference configuration (n x 3)
    u = nodal displacements (n x 3)

    """
    dN = elem_type.dN(*r)
    J = np.dot(X.T, dN)
    Jinv = np.linalg.inv(J)
    Jdet = np.linalg.det(J)
    du = np.dot(u.T, dN)
    # Push from natural basis to reference configuration
    f = np.dot(Jinv, du) + np.eye(3)
    return f

def _cross(u, v):
    """Cross product for two vectors in R3.

    """
    w = np.array([u[1]*v[2] - u[2]*v[1],
                  u[2]*v[0] - u[0]*v[2],
                  u[0]*v[1] - u[1]*v[0]])
    return w

def elem_obj(element, nodes, eid=None):
    """Returns an Element object from node and element tuples.

    element := the indices of this element's nodes
    nodes := the global list of node coordinates

    """
    n = len(element)
    if n == 3:
        etype = Tri3
    elif n == 4 \
       and (len(nodes[0]) == 2 or all([x[2] == 0.0 for x in nodes])):
        etype = Quad4
    elif n == 8:
        etype = Hex8
    else:
        s = "{} node element not recognized".format(n)
        raise Exception(s)
    return etype(element, nodes, elem_id=eid)


class Element:
    """Data and metadata for an element.

    Attributes
    ----------
    eid := element id

    Notes
    -----
    When using 2d elements, it is highly recommended that they
    coincide with the xy plane.  The current implementation of the
    methods cannot do anything useful with the z dimension.

    """
    matl_id = None # material integer code (FEBio codes are 1-indexed)
    matl = None # material definition class
    inode = [] # list of node indices
    xnode_mesh = [] # list of node coordinates for whole mesh
    # material := material class

    def __init__(self, inode, xnode_mesh,
                 elem_id=None, matl_id=None):
        self.eid = elem_id
        self.inode = inode
        self.xnode_mesh = xnode_mesh
        self.matl_id = matl_id

    @property
    def xnode(self):
        """List of node coordinate tuples.

        """
        return [self.xnode_mesh[i] for i in self.inode]

    @property
    def centroid(self):
        """Centroid of element.

        """
        return self.interp((0,0,0), self.xnode_mesh)

    def face_normals(self):
        """List of face normals

        """
        points = np.array(self.xnode)
        normals = []
        # Iterate over faces
        for f in self.face_nodes:
            # Define vectors for two face edges, using the first face
            # node as the origin.  For quadrilateral faces, one node
            # is left unused.
            v1 = points[f[1]] - points[f[0]]
            v2 = points[f[-1]] - points[f[0]]
            # compute the face normal
            normals.append(_cross(v1, v2))
        return normals

    def faces_with_node(self, node_id):
        """Indices of faces that include node id

        The node id here is local to the element.

        """
        return [i for i, f in enumerate(self.face_nodes)
                if node_id in f]

    def f(self, r, u):
        """Calculate F tensor.

        r := coordinate vector in element's natural basis
        u := list of displacements for all the nodes in the mesh.

        """
        dudx = self.dinterp(r, u)
        F = dudx + np.eye(3)
        return F

    def j(self, r):
        """Jacobian matrix (∂x_i/∂r_j) evaluated at r

        """
        ddr = self.dN(*r)
        ddr = np.vstack(self.dN(*r))
        J = np.dot(np.array(self.xnode).T, ddr)
        return J

    def integrate(self, f, *args):
        """Integrate a function over the element.

        f := The function to integrate.  Must be callable as `f(e,r)`,
            with `e` being this element object instance and `r` being
            a 2d or 3d coordinate vector (a 2- or 3-element
            array-like).

        """
        def jdet(r):
            """Calculate determinant of the jacobian, handling R3 → R2
            transformations correctly.

            """
            j = self.j(r)
            if j.shape[1] == 2 and j.shape[0] == 3:
                # jacobian of transform from R3 to R2
                n = np.cross(j[:,0], j[:,1])
                jd = n[2]
            elif j.shape[0] == j.shape[1]:
                # is square; working in R2 or R3
                jd = np.linalg.det(j)
            else:
                s = "Transformations from R{} to R{} space are not handled.".format(j.shape[0], j.shape[1])
                raise Exception(s)
            return jd

        return sum((f(self, r, *args) * jdet(r) * w
                    for r, w in zip(self.gloc, self.gwt)))

    def interp(self, r, values):
        """Interpolate node-valued data at r.

        values := A list with a 1:1 mapping to the list of nodes in
        the mesh.  The list elements can be scalar or vector valued
        (but must be consistent).

        For example, to obtain the centroid of a 2d element:

            element.interp((0,0), element.xnode_mesh)

        """
        v_node = np.array([values[i] for i in self.inode])
        return np.dot(self.N(*r), v_node)

    def dinterp(self, r, values):
        """Evalute d/dx of node-valued data at r

        The node-valued data may be scalar or vector.

        Note: If you are using a 2d element, do not use 3d vector
        values.

        """
        v_node = np.array([values[i] for i in self.inode])
        j = self.j(r)
        jinv = np.linalg.pinv(j).T
        ddr = np.vstack(self.dN(*r))
        dvdr = np.dot(v_node.T, ddr)
        dvdx = np.dot(jinv, dvdr.T).T
        return dvdx

class Element3D(Element):
    """Class for 3D elements.

    """
    is_planar = False

class Element2D(Element):
    """Class for 2D elements.

    All 2D elements should inherit from this class.

    """
    is_planar = True

    def edges_with_node(self, node_id):
        """Indices of edges that include node id.

        """
        return [i for i, l in enumerate(self.edge_nodes)
                if node_id in l]

    def edge_normals(self):
        """Return list of edge normals.

        The edge normals are constrained to lie in the same plane as
        the element.  The normals point outward.

        """
        points = np.array(self.xnode)
        normals = []
        # Iterate over edges
        for l in self.edge_nodes:
            v = points[l[1]] - points[l[0]]
            face_normal = self.face_normals()[0]
            normals.append(_cross(v, face_normal))
        return normals

    def to_natural(self, p):
        """Return natural coordinates for p = (x, y, z)

        """
        p = np.array(p)
        x0 = np.dot(self.N(*[0]*self.r_n), self.xnode)
        v = p - x0
        j = self.j([0]*self.r_n)
        jinv = np.linalg.pinv(j)
        nat_coords = np.dot(jinv, v)
        return nat_coords


class Tri3(Element2D):
    """Functions for tri3 elements.

    """
    n = 3
    r_n = 2 # number of natural basis parameters
    node_connectivity = [[1, 2],
                         [0, 2],
                         [1, 0]]

    # oriented so positive normal follows node ordering convention
    face_nodes = [[0, 1, 2]]

    @property
    def centroid(self):
        """Centroid of element.

        """
        return self.interp((1.0/3.0, 1.0/3.0), self.xnode_mesh)

    @staticmethod
    def N(r, s, t=None):
        """Shape functions.

        """
        n = [0.0] * 3
        n[0] = 1.0 - r - s
        n[1] = r
        n[2] = s
        return n

    @staticmethod
    def dN(r, s, t=None):
        """Shape function 1st derivatives.

        """
        dn = [np.zeros(2) for i in xrange(3)]
        # d/dr
        dn[0][0] = -1.0
        dn[1][0] = 1.0
        dn[2][0] = 0.0
        # d/ds
        dn[0][1] = -1.0
        dn[1][1] = 0.0
        dn[2][1] = 1.0

        return dn


class Hex8(Element3D):
    """Functions for hex8 trilinear elements.

    """
    # gwt
    # gloc

    n = 8 # number of vertices
    r_n = 3 # number of natural basis parameters

    node_connectivity = [[1, 3, 4], # 0
                         [0, 2, 3], # 1
                         [1, 3, 6], # 2
                         [0, 2, 7], # 3
                         [0, 5, 7], # 4
                         [1, 4, 6], # 5
                         [2, 5, 7], # 6
                         [3, 4, 6]] # 7

    # Oriented positive = out
    face_nodes = [[0, 1, 5, 4],
                  [1, 2, 6, 5],
                  [2, 3, 7, 6],
                  [3, 0, 4, 7],
                  [4, 5, 6, 7],
                  [0, 3, 2, 1]]

    @staticmethod
    def N(r, s, t):
        """Shape functions.

        8-element vector

        """
        n = [0.0] * 8
        n[0] = 1. / 8. * (1 - r) * (1 - s) * (1 - t)
        n[1] = 1. / 8. * (1 + r) * (1 - s) * (1 - t)
        n[2] = 1. / 8. * (1 + r) * (1 - s) * (1 - t)
        n[3] = 1. / 8. * (1 - r) * (1 - s) * (1 - t)
        n[4] = 1. / 8. * (1 - r) * (1 - s) * (1 - t)
        n[5] = 1. / 8. * (1 + r) * (1 - s) * (1 - t)
        n[6] = 1. / 8. * (1 + r) * (1 - s) * (1 - t)
        n[7] = 1. / 8. * (1 - r) * (1 - s) * (1 - t)
        return n

    @staticmethod
    def dN(r, s, t):
        """Shape function 1st derivatives.

        """
        dn = [np.zeros(3) for i in xrange(8)]
        # d/dr
        dn[0][0] = -1. / 8. * (1 - s) * (1 - t)
        dn[1][0] =  1. / 8. * (1 - s) * (1 - t)
        dn[2][0] =  1. / 8. * (1 - s) * (1 - t)
        dn[3][0] = -1. / 8. * (1 - s) * (1 - t)
        dn[4][0] = -1. / 8. * (1 - s) * (1 - t)
        dn[5][0] =  1. / 8. * (1 - s) * (1 - t)
        dn[6][0] =  1. / 8. * (1 - s) * (1 - t)
        dn[7][0] = -1. / 8. * (1 - s) * (1 - t)
        # d/ds
        dn[0][1] = -1. / 8. * (1 - r) * (1 - t)
        dn[1][1] = -1. / 8. * (1 - r) * (1 - t)
        dn[2][1] =  1. / 8. * (1 - r) * (1 - t)
        dn[3][1] =  1. / 8. * (1 - r) * (1 - t)
        dn[4][1] = -1. / 8. * (1 - r) * (1 - t)
        dn[5][1] = -1. / 8. * (1 - r) * (1 - t)
        dn[6][1] =  1. / 8. * (1 - r) * (1 - t)
        dn[7][1] =  1. / 8. * (1 - r) * (1 - t)
        # d/dt
        dn[0][2] = -1. / 8. * (1 - r) * (1 - s)
        dn[1][2] = -1. / 8. * (1 - r) * (1 - s)
        dn[2][2] = -1. / 8. * (1 - r) * (1 - s)
        dn[3][2] = -1. / 8. * (1 - r) * (1 - s)
        dn[4][2] =  1. / 8. * (1 - r) * (1 - s)
        dn[5][2] =  1. / 8. * (1 - r) * (1 - s)
        dn[6][2] =  1. / 8. * (1 - r) * (1 - s)
        dn[7][2] =  1. / 8. * (1 - r) * (1 - s)

        return dn

    @staticmethod
    def ddN(r, s, t):
        """"Shape function 2nd derivatives.

        """
        pass

class Quad4(Element2D):
    """Shape functions for quad4 bilinear shell element.

    This definition uses Guass point integration.  FEBio also has a
    nodal integration-type quad4 element defined; I'm not exactly sure
    which is used in the solver.

    """
    n = 4
    r_n = 2 # number of natural basis parameters
    node_connectivity = [[1, 3],
                         [0, 2],
                         [1, 3],
                         [2, 0]]

    face_nodes = [[0, 1, 2, 3]]

    edge_nodes = [[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 0]]

    a = 1.0 / 3.0**0.5
    gloc = ((-a, -a),           # Guass point locations
          ( a, -a),
          ( a, a),
          (-a, a))
    gwt = (1, 1, 1, 1)          # Guass weights

    @staticmethod
    def N(r, s):
        """Shape functions.

        """
        n = [0.0] * 4
        n[0] = 0.25 * (1 - r) * (1 - s)
        n[1] = 0.25 * (1 + r) * (1 - s)
        n[2] = 0.25 * (1 + r) * (1 + s)
        n[3] = 0.25 * (1 - r) * (1 + s)
        return n

    @staticmethod
    def dN(r, s, t=None):
        """Shape function 1st derivatives.

        """
        dn = [np.zeros(2) for i in xrange(4)]
        dn[0][0] = -0.25 * (1 - s)
        dn[1][0] =  0.25 * (1 - s)
        dn[2][0] =  0.25 * (1 + s)
        dn[3][0] = -0.25 * (1 + s)

        dn[0][1] = -0.25 * (1 - r)
        dn[1][1] = -0.25 * (1 + r)
        dn[2][1] =  0.25 * (1 + r)
        dn[3][1] =  0.25 * (1 - r)

        return dn

    @staticmethod
    def ddN(r, s, t=None):
        """Shape function 2nd derivatives.

        """
        pass
