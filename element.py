# -*- coding: utf-8 -*-
import numpy as np

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
    matl_id = 0 # material integer code (FEBio codes are 1-indexed)
    matl = None # material definition class
    inode = [] # list of node indices
    xnode_mesh = [] # list of node coordinates for whole mesh

    def __init__(self, inode, xnode, elem_id=None, matl_id=None):
        self.eid = elem_id
        self.inode = inode
        self.xnode_mesh = xnode
        self.matl_id = matl_id

    @property
    def xnode(self):
        """List of node coordinate tuples.

        """
        return [self.xnode_mesh[i] for i in self.inode]

    def f(self, r, u):
        """Calculate F tensor.

        r := coordinate vector in element's natural basis
        u := list of displacements for all the nodes in the mesh.
        
        """
        u = [v[:len(r)] for v in u]
        dudx = self.dinterp(r, u)
        if len(r) == 2:
            # pad to 3-dimensions
            dudx = np.pad(dudx, ((0,1), (0,1)), mode='constant')
        F = dudx + np.eye(3)
        return F

    def j(self, r):
        """Jacobian matrix (∂x_i/∂r_j) evaluated at r

        """
        ddr = self.dN(*r)
        ddr = np.vstack(ddr)
        x_node = [x[:len(r)] for x in self.xnode]
        x_node = np.array(x_node).T  # i over x, j over nodes
        J = np.dot(x_node, ddr)
        return J

    def integrate(self, f, *args):
        """Integrate a function over the element.

        f := The function to integrate.  Must be callable as `f(e,r)`,
            with `e` being this element object instance and `r` being
            a 2d or 3d coordinate vector (a 2- or 3-element
            array-like).

        """
        return sum((f(self, r, *args) * np.linalg.det(self.j(r)) * w 
                    for r, w in zip(self.gloc, self.gwt)))

    def interp(self, r, values):
        """Interpolate node-valued data at r.

        values := A list with a 1:1 mapping to the list of nodes in
        the mesh.  The list elements can be scalar or vector valued
        (but must be consistent).

        For example, to obtain the centroid of a 2d element:

            element.interp((0,0), element.xnode)

        """
        v_node = np.array([values[i] for i in self.inode])
        return np.dot(self.N(*r), v_node)

    def dinterp(self, r, values):
        """Evalute d/dx of node-valued data at r

        The node-valued data may be scalar or vector.

        Note: If you are using a 2d element, do not use 3d vector
        values.

        """
        v_node = np.array([values[i] for i in self.inode]).T
        j = self.j(r)
        jinv = np.linalg.inv(j)
        ddr = np.vstack(self.dN(*r))
        dvdr = np.dot(v_node, ddr)
        dvdx = np.dot(jinv, dvdr.T)
        return dvdx.T


class Hex8(Element):
    """Functions for hex8 trilinear elements.

    """
    # gwt
    # gloc
    n = 8 # number of vertices

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

class Quad4(Element):
    """Shape functions for quad4 bilinear shell element.

    This definition uses Guass point integration.  FEBio also has a
    nodal integration-type quad4 element defined; I'm not exactly sure
    which is used in the solver.

    """
    n = 4

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
    def dN(r, s):
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
    def ddN(r, s):
        """Shape function 2nd derivatives.

        """
        pass
