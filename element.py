# -*- coding: utf-8 -*-
import numpy as np

def f(r, X, u, elem_type):
    """Calculate F tensor from nodal values and shape functions.

    r = target coordinates in natural basis (tuple)
    X = nodal coordinates in reference configuration (n x 3)
    u = nodal displacements (n x 3)
    elem_type = element class (Hex8 or Quad4)

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

    """
    mat_id = 0 # material integer code (real codes are > 0)
    material = None # material definition class
    inode = [] # list of node indices

    def __init__(self, inode, xnode, elem_id=None, mat_id=None):
        self.eid = elem_id
        self.inode = inode
        self.xnode = xnode # List of node coordinate tuples
        self.mat_id = mat_id

    def j(self, r):
        """Jacobian matrix (∂x_i/∂r_j) evaluated at r

        """
        ddr = self.dN(*r)
        x_node = [self.xnode[i] for i in self.inode]
        J = sum(np.outer(x, d) for x, d in zip(x_node, ddr))
        return J

    def integrate(self, f):
        """Integrate a function over the element.

        f := The function to integrate.  Must be callable as `f(r)`,
            with r being a 2d or 3d coordinate vector.

        """
        return sum((f(r) * np.linalg.det(self.j(r)) * w 
                    for r, w in zip(self.gloc, self.gwt)))

    def interpolate(self, r, values):
        """Interpolate values (defined per node) at r

        values := A list with a 1:1 mapping to the list of nodes in
        the mesh.  The list elements can be scalar or vector valued
        (but must be consistent).

        For example, to obtain the centroid of a 2d element:

            element.interpolate((0, 0), element.xnode)

        """
        v_node = np.array([values[i] for i in self.inode])
        return np.dot(self.N(*r), v_node)

    def dinterp(self, r, values):
        """Evalute d/dx of node-valued data at r

        """
        v_node = [values[i] for i in self.inode]
        j = self.j(r)
        jinv = np.linalg.inv(j)
        ddr = self.dN(*r)
        dvdr = (d * v for d, v in zip(ddr, v_node))
        dvdx = sum(np.dot(jinv, d) for d in dvdr)
        return dvdx


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
        """Shape functions' 1st derivatives.

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
        """"Shape functions 2nd derivatives.

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
        """Shape function' 1st derivatives.

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
