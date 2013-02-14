import numpy as np

def shpfun(r, s, t):
    """Shape functions for hex8 (brick elements)."""
    

def dshpfun(r, s, t):
    """Shape function derivatives.

    The node order follows FEBio's convention.
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
