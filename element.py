import numpy as np

class Hex8:
    """Shape functions for hex8 trilinear elements.

    """
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
        
        return np.vstack((nr.T, ns.T, nt.T))

    @staticmethod
    def ddN(r, s, t):
        """"Shape functions 2nd derivatives.

        """
        pass

class quad4:
    """Shape functions for quad4 bilinear shell element.

    """
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

        return np.vstack((nr.T, ns.T))

    @staticmethod
    def ddN(r, s):
        """Shape function 2nd derivatives.

        """
        pass
