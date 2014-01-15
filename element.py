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

    """
    etype = None # element type class
    mat_id = 0 # material integer code (real codes are > 0)
    material = None # material definition class
    nodes = [] # list of node indices

    def __init__(self, nodes, elem_type, mat_id):
        self.nodes = nodes
        self.etype = elem_type
        self.mat_id = mat_id


# Element type classes should implement the following:
#
#     n : number of vertices
#     N(r, s, t) : shape function
#    dN(r, s, t) : 1st derivative of shape function


class Hex8:
    """Functions for hex8 trilinear elements.

    """
    # gwt
    # gloc
    n = 8 # number of vertices

    @staticmethod
    def N(r, s, t):
        """Shape functions.

        """
        n = np.zeros((8, 1))
        n[0] = 1. / 8. * (1 - r) * (1 - s) * (1 - t)
        n[1] = 1. / 8. * (1 + r) * (1 - s) * (1 - t)
        n[2] = 1. / 8. * (1 + r) * (1 - s) * (1 - t)
        n[3] = 1. / 8. * (1 - r) * (1 - s) * (1 - t)
        n[4] = 1. / 8. * (1 - r) * (1 - s) * (1 - t)
        n[5] = 1. / 8. * (1 + r) * (1 - s) * (1 - t)
        n[6] = 1. / 8. * (1 + r) * (1 - s) * (1 - t)
        n[7] = 1. / 8. * (1 - r) * (1 - s) * (1 - t)
        return n.T

    @staticmethod
    def dN(r, s, t):
        """Shape functions 1st derivatives.

        """
        nr = np.zeros((8, 1))
        nr[0] = -1. / 8. * (1 - s) * (1 - t)
        nr[1] =  1. / 8. * (1 - s) * (1 - t)
        nr[2] =  1. / 8. * (1 - s) * (1 - t)
        nr[3] = -1. / 8. * (1 - s) * (1 - t)
        nr[4] = -1. / 8. * (1 - s) * (1 - t)
        nr[5] =  1. / 8. * (1 - s) * (1 - t)
        nr[6] =  1. / 8. * (1 - s) * (1 - t)
        nr[7] = -1. / 8. * (1 - s) * (1 - t)

        ns = np.zeros((8, 1))
        ns[0] = -1. / 8. * (1 - r) * (1 - t)
        ns[1] = -1. / 8. * (1 - r) * (1 - t)
        ns[2] =  1. / 8. * (1 - r) * (1 - t)
        ns[3] =  1. / 8. * (1 - r) * (1 - t)
        ns[4] = -1. / 8. * (1 - r) * (1 - t)
        ns[5] = -1. / 8. * (1 - r) * (1 - t)
        ns[6] =  1. / 8. * (1 - r) * (1 - t)
        ns[7] =  1. / 8. * (1 - r) * (1 - t)

        nt = np.zeros((8, 1))
        nt[0] = -1. / 8. * (1 - r) * (1 - s)
        nt[1] = -1. / 8. * (1 - r) * (1 - s)
        nt[2] = -1. / 8. * (1 - r) * (1 - s)
        nt[3] = -1. / 8. * (1 - r) * (1 - s)
        nt[4] =  1. / 8. * (1 - r) * (1 - s)
        nt[5] =  1. / 8. * (1 - r) * (1 - s)
        nt[6] =  1. / 8. * (1 - r) * (1 - s)
        nt[7] =  1. / 8. * (1 - r) * (1 - s)

        return np.hstack((nr, ns, nt))

    @staticmethod
    def ddN(r, s, t):
        """"Shape functions 2nd derivatives.

        """
        pass

class Quad4:
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
        n = np.zeros((4, 1))
        n[0] = 0.25 * (1 - r) * (1 - s)
        n[1] = 0.25 * (1 + r) * (1 - s)
        n[2] = 0.25 * (1 + r) * (1 + s)
        n[3] = 0.25 * (1 - r) * (1 + s)
        return n.T

    @staticmethod
    def dN(r, s):
        """Shape function 1st derivatives.

        """
        nr = np.zeros((4, 1))
        nr[0] = -0.25 * (1 - s)
        nr[1] =  0.25 * (1 - s)
        nr[2] =  0.25 * (1 + s)
        nr[3] = -0.25 * (1 + s)

        ns = np.zeros((4, 1))
        ns[0] = -0.25 * (1 - r)
        ns[1] = -0.25 * (1 + r)
        ns[2] =  0.25 * (1 + r)
        ns[3] =  0.25 * (1 - r)

        return np.hstack((nr, ns))

    @staticmethod
    def ddN(r, s):
        """Shape function 2nd derivatives.

        """
        pass
