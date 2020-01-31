"""Convenient mathematical operations."""

import numpy as np


def orthonormal_basis(a, d):
    """Return basis for two vectors.

    The basis is returned as a tuple of basis vectors (e1, e2, e3) with
    each vector defined as follows:

    - e1 = Parallel to a.

    - e2 = The component of d perpendicular to a.

    - e3 = Perpendicular to a and d.

    Each basis vector is a unit vector.

    """
    g = np.cross(a, d)
    d = np.cross(g, a)
    a = a / np.linalg.norm(a)
    d = d / np.linalg.norm(d)
    g = g / np.linalg.norm(g)
    return (a, d, g)
