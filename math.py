"""Convenient mathematical operations."""
# Base packages
from math import radians, degrees, cos, sin
# Published packages
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


def sph_from_vec(vec):
    """Return spherical coordinates for a (unit) vector, in degrees."""
    vec = vec / np.linalg.norm(vec)  # ensure unit vector
    φ = np.arccos(vec[2])
    # ^ zenith / polar / declination angle
    θ = np.arctan2(vec[1], vec[0])
    return degrees(θ), degrees(φ)


def vec_from_sph(θ, φ):
    """Return unit vector from azimuth and zenith angles (in degrees)."""
    return np.array((cos(radians(θ))*sin(radians(φ)),
                     sin(radians(θ))*sin(radians(φ)),
                     cos(radians(φ))))
