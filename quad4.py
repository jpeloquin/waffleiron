import numpy as np

def shpfun(r, s):
    """Shape functions for quad4 elements."""
    n = np.zeros((4, 1))
    n[0] = 0.25 * (1 - r) * (1 - s)
    n[1] = 0.25 * (1 + r) * (1 - s)
    n[2] = 0.25 * (1 + r) * (1 + s)
    n[3] = 0.25 * (1 - r) * (1 + s)
    return n.T

def dshpfun(r, s):
    """Shape function derivatives for quad4 elements.

    The node order follows FEBio's convention.
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
    ns[3] =  0.25 * (1 -r)

    return np.vstack((nr.T, ns.T))
