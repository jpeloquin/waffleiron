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
    return np.vstack((a, d, g)).T


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


def biasrange_log(start, stop, n=10):
    """Log spaced series with n points, finer near start.

    """
    # x ∈ [0, 1]
    x = (10**np.linspace(0, np.log10(10 + 1), n) - 1) / 10
    l = stop - start
    x = [sc * l + start for sc in x]
    # fix start and endpoints to given values, as numerical
    # error will have accumulated
    x[0] = start
    x[-1] = stop
    return x


def biasrange_sqrt(start, stop, n=10):
    """Points spaced ∝ √x.

    """
    # x ∈ [0, 1]
    x0 = 0
    x1 = 1
    x = np.linspace(x0**0.5, x1**0.5, n)**2.0
    x = start + (stop - start) * x
    return x


def bias_pt_series(line, n=None, bias='log', minstep=None,
                   bias_direction=1):
    """Return a series of points on a line segment with biased spacing.

    `line` is a list of two points in 2-space or higher dimensions.

    If `n` is not `None`, that many points are returned, and `minstep`
    should be `None`.  The default is 10 points.

    If `minstep` is not `None`, the number of points will be chosen
    such that the smallest distance between points is less than or
    equal to `minstep`.

    If bias_direction is 1, the smallest interval is at the start of
    the list.  If bias_direction is -1, it is at the end.

    """
    p1 = np.array(line[0]) # origin of directed line segment
    p2 = np.array(line[1])
    v = p2 - p1
    length = np.linalg.norm(v) # line length
    if length == 0.0:
        raise ValueError("Line has zero length.")
    u = v / length # unit vector pointing in line direction

    if bias == 'log':
        fspc = biasrange_log
    elif bias == 'linear':
        fspc = np.linspace
    elif bias == 'sqrt':
        fspc = biasrange_sqrt

    # Figure out how many points to return
    if n is None and minstep is None:
        # Return 10 points (default)
        n = 10
        s = fspc(0, 1, n)
    elif n is None and minstep is not None:
        # Calculate the number of points necessary to achieve the
        # specified minimum step size
        n = 2
        s = fspc(0, 1, n)
        dmin = s[1] * length
        while dmin > minstep:
            n = n + 1
            s = fspc(0, 1, n)
            dmin = s[1] * length
    elif n is not None and minstep is None:
        s = fspc(0, 1, n)
    else:
        # Both n and minstep are defined
        raise Exception('The number of points `n` and the minimum '
                        'distance between points `minstep` are both defined; '
                        'only one can be defined at a time.')

    # Compute the points
    if bias_direction == -1:
        s = [(1 - sc) for sc in s][::-1]
    pts = [sc * length * u + p1 for sc in s]

    return pts


def densify(curve, n):
    dense_curve = []
    for i, (x0, y0) in enumerate(curve[:-1]):
        x1 = curve[i + 1][0]
        y1 = curve[i + 1][1]
        x = np.linspace(x0, x1, n + 1)
        y = np.interp(x, [x0, x1], [y0, y1])
        dense_curve += [(a, b) for a, b in zip(x[:-1], y[:-1])]
    # Add last point
    dense_curve.append((curve[-1]))
    return dense_curve


def even_pt_series(line, n):
    """Return a list of n points evenly spaced along line segment.

    """
    # This is just a stub to postpone renaming all uses of
    # `even_pt_series` to `pt_series`.
    return pt_series(line, n, f=np.linspace)


def pt_series(line, n, f=np.linspace):
    """Return a list of points spaced along a line segment.

    line := a list of n-tuples (points), or equivalent iterable

    The returned points are numpy arrays of the same dimension as A and B.

    """
    A = np.array(line[0])
    B = np.array(line[1])
    v = B - A
    AB = [A + s * v for s in f(0, 1, n)]
    AB[-1] = B # for exactness
    return AB
