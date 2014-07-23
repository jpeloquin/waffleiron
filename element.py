# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import fsolve

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
    nodes := Nodal positions in reference configuration.  The order
    follows FEBio convention.

    mesh := (optional) The mesh to which the element belongs.

    ids := (optional) Nodal indices into `mesh.nodes`

    material := (optional) Material object instance.

    properties := A dictionary of nodal values.  Expected keys are
    'displacement' (indexing a list of displacement vectors).

    Notes
    -----
    When using 2d elements, they should coincide with the xy plane.
    The current implementation of the methods cannot do anything
    useful with the z dimension.  This will be fixed.

    """
    def __init__(self, nodes, material=None):
        """Create an element from a list of nodes.

        Parameters
        ----------
        nodes : n × 3 array-like or n × 1 array-like
            Coordinates for n nodes.  The second dimension is x, y,
            and z.  If `mesh` is provided, a list of node indices is
            instead expected, and the node coordinates will be
            calculated by indexing into `mesh`.

        """
        self.ids = None # indices of nodes in mesh
        self.mesh = None
        self.material = material
        self.properties = {'displacement': np.array([(0, 0, 0) for i in nodes])}
        # Nodal coordinates
        self.nodes = np.array(nodes)
        assert self.nodes.shape[1] >= 2
            
    @classmethod
    def from_ids(cls, ids, nodelist, material=None):
        """Create an element from nodal indices.

        """
        nodes = [nodelist[i] for i in ids]
        element = cls(nodes, material)
        element.ids = ids
        return element

    def apply_property(self, label, values):
        """Apply nodal properties.

        """
        assert len(values) == len(self.nodes)
        values = np.array(values)
        self.properties[label] = values

    def x(self, config='reference'):
        """Nodal positions in reference or deformed configuration.

        """
        if config == 'reference':
            x = self.nodes
        elif config == 'deformed':
            x = self.nodes + self.properties['displacement']
        else:
            raise Exception('Value "{}" for config not recognized.  Use "reference" or "deformed"'.format(config))
        return x

    def interp(self, r, prop='displacement'):
        """Interpolate node-valued data at r.

        values := A list with a 1:1 mapping to the list of nodes in
        the mesh.  The list elements can be scalar or vector valued
        (but must be consistent).

        For example, to obtain the centroid of a 2d element:

            element.interp((0,0), element.xnode_mesh)

        """
        v = self.properties[prop] # nodal values
        return np.dot(self.N(*r), v_node)

    def dinterp(self, r, prop='displacement'):
        """Evalute d/dx of node-valued data at r

        The node-valued data may be scalar or vector.

        Note: If you are using a 2d element, do not use 3d vector
        values.

        """
        v = self.properties[prop] # nodal values
        j = self.j(r)
        jinv = np.linalg.pinv(j).T
        ddr = np.vstack(self.dN(*r))
        dvdr = np.dot(v.T, ddr)
        dvdx = np.dot(jinv, dvdr.T).T
        return dvdx

    def f(self, r):
        """Calculate F tensor (convenience function).

        r := coordinate vector in element's natural basis

        """
        dudx = self.dinterp(r, prop='displacement')
        F = dudx + np.eye(3)
        return F

    def j(self, r, config='reference'):
        """Jacobian matrix (∂x_i/∂r_j) evaluated at r

        """
        ddr = self.dN(*r)
        ddr = np.vstack(self.dN(*r))
        J = np.dot(np.array(self.nodes).T, ddr)
        return J

    def integrate(self, fn, *args):
        """Integrate a function over the element.

        f := The function to integrate.  Must be callable as `f(e,
            r)`, with `e` being this element object instance and `r`
            being 3-element array-like specifing the natural basis
            coordinate at which f is evaluated.

        """
        return sum((fn(self, r, *args) * self.jdet(r) * w
                    for r, w in zip(self.gloc, self.gwt)))

    def centroid(self, config='reference'):
        """Centroid of element.

        """
        x = self.x(config)
        return self.interp((0,0,0), x)

    def face_normals(self, config='reference'):
        """List of face normals

        """
        points = self.x(config)
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
        """Indices of faces that include a node.

        The node id here is local to the element.

        """
        return [i for i, f in enumerate(self.face_nodes)
                if node_id in f]


class Element3D(Element):
    """Class for 3D elements.

    """
    is_planar = False

    def jdet(self, r, config='reference'):
        """Calculate determinant of the R3 → R3 jacobian.

        """
        return np.linalg.det(self.j(r, config))


class Element2D(Element):
    """Class for shell (2D) elements.

    All shell elements should inherit from this class.

    """
    is_planar = True

    def jdet(self, r, config='reference'):
        """Calculate determinant of the R3 → R2 jacobian.

        """
        j = self.j(r, config)
        n = np.cross(j[:,0], j[:,1])
        try:
            return n[2]
        except IndexError:
            # i.e if 2d vectors were used
            return n

    def edges_with_node(self, node_id):
        """Indices of edges that include node id.

        """
        return [i for i, l in enumerate(self.edge_nodes)
                if node_id in l]

    def edge_normals(self, config='reference'):
        """Return list of edge normals.

        The edge normals are constrained to lie in the same plane as
        the element.  The normals point outward.

        """
        points = self.x(config)
        normals = []
        # Iterate over edges
        for l in self.edge_nodes:
            v = points[l[1]] - points[l[0]]
            face_normal = self.face_normals()[0]
            normals.append(_cross(v, face_normal))
        return normals

    def to_natural(self, pt, config='reference'):
        """Return natural coordinates for p = (x, y, z)

        """
        pt = np.array(pt)
        x = self.x(config)
        x0 = np.dot(self.N(*[0]*self.r_n), x)
        v = pt - x0
        j = self.j([0]*self.r_n)
        jinv = np.linalg.pinv(j)
        nat_coords = np.dot(jinv, v)
        if (nat_coords > 1).any() or (nat_coords < -1).any():
            raise Exception("Computed natural basis coordinates "
                            "{} are outside the element's "
                            "domain.".format(nat_coords))
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
    def centroid(self, config='reference'):
        """Centroid of element.

        """
        x = self.x(config)
        return self.interp((1.0/3.0, 1.0/3.0), x)

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

    # Guass point locations
    g = 1.0 / 3.0**0.5
    gloc = ((-g, -g, -g),
            ( g, -g, -g),
            ( g,  g, -g),
            (-g,  g, -g),
            (-g, -g,  g),
            ( g, -g,  g),
            ( g,  g,  g),
            (-g,  g,  g))
    # Guass weights
    gwt = (1, 1, 1, 1, 1, 1, 1, 1)

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
        dn[2][0] =  1. / 8. * (1 + s) * (1 - t)
        dn[3][0] = -1. / 8. * (1 + s) * (1 - t)
        dn[4][0] = -1. / 8. * (1 - s) * (1 + t)
        dn[5][0] =  1. / 8. * (1 - s) * (1 + t)
        dn[6][0] =  1. / 8. * (1 + s) * (1 + t)
        dn[7][0] = -1. / 8. * (1 + s) * (1 + t)
        # d/ds
        dn[0][1] = -1. / 8. * (1 - r) * (1 - t)
        dn[1][1] = -1. / 8. * (1 + r) * (1 - t)
        dn[2][1] =  1. / 8. * (1 + r) * (1 - t)
        dn[3][1] =  1. / 8. * (1 - r) * (1 - t)
        dn[4][1] = -1. / 8. * (1 - r) * (1 + t)
        dn[5][1] = -1. / 8. * (1 + r) * (1 + t)
        dn[6][1] =  1. / 8. * (1 + r) * (1 + t)
        dn[7][1] =  1. / 8. * (1 - r) * (1 + t)
        # d/dt
        dn[0][2] = -1. / 8. * (1 - r) * (1 - s)
        dn[1][2] = -1. / 8. * (1 + r) * (1 - s)
        dn[2][2] = -1. / 8. * (1 + r) * (1 + s)
        dn[3][2] = -1. / 8. * (1 - r) * (1 + s)
        dn[4][2] =  1. / 8. * (1 - r) * (1 - s)
        dn[5][2] =  1. / 8. * (1 + r) * (1 - s)
        dn[6][2] =  1. / 8. * (1 + r) * (1 + s)
        dn[7][2] =  1. / 8. * (1 - r) * (1 + s)
        return dn

    @staticmethod
    def ddN(r, s, t):
        """"Shape function 2nd derivatives.

        """
        raise NotImplementedError()


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
    gloc = ((-a, -a),           # Gauss point locations
          ( a, -a),
          ( a, a),
          (-a, a))
    gwt = (1, 1, 1, 1)          # Gauss weights

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
        # d/dr
        dn[0][0] = -0.25 * (1 - s)
        dn[1][0] =  0.25 * (1 - s)
        dn[2][0] =  0.25 * (1 + s)
        dn[3][0] = -0.25 * (1 + s)
        # d/ds
        dn[0][1] = -0.25 * (1 - r)
        dn[1][1] = -0.25 * (1 + r)
        dn[2][1] =  0.25 * (1 + r)
        dn[3][1] =  0.25 * (1 - r)
        return dn

    @staticmethod
    def ddN(r, s, t=None):
        """Shape function 2nd derivatives.

        """
        raise NotImplementedError()
