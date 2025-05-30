"""Convenient mathematical operations."""

# Base packages
from math import radians, degrees, cos, sin, e, pi

# Published packages
import numpy as np


def dyad(A, B):
    """Standard dyadic product of two tensors"""
    # np.einsum("ij,kl->ijkl", A, B)
    return np.multiply.outer(A, B)


def dyad_odot(A, B):
    """⊙ dyadic product of two order-2 tensors"""
    return 0.5 * (np.einsum("ik,jl->ijkl", A, B) + np.einsum("il,jk->ijkl", A, B))


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
    return np.array(
        (
            cos(radians(θ)) * sin(radians(φ)),
            sin(radians(θ)) * sin(radians(φ)),
            cos(radians(φ)),
        )
    )


def biasrange_log(start, stop, n=10):
    """Log spaced series with n points, finer near start."""
    # x ∈ [0, 1]
    x = (10 ** np.linspace(0, np.log10(10 + 1), n) - 1) / 10
    l = stop - start
    x = [sc * l + start for sc in x]
    # fix start and endpoints to given values, as numerical
    # error will have accumulated
    x[0] = start
    x[-1] = stop
    return x


def linspaced(offset, span, n):
    """Return a series of n equally spaced values x.

    The spacing scales as x, with:

    offset := Distance between x = 0 and the first value in the series.

    span := Distance between the first and last value in the series.

    n := Number of values in the series.  Non-negative.

    """
    if n == 1 and span != 0:
        raise ValueError(f"Cannot span distance of {span} with n = {n} value.")
    return np.linspace(offset, offset + span, n)


def x_biasfactor(start, span, n, factor=1):
    """Return a sequence of values spaced with compounded scaling by a bias factor

    :param start: Starting value of the sequence.

    :param span: Total length spanned by the sequence.

    :param n: Number of desired divisions (elements) of the sequence.

    :param factor: Bias factor.  Each pair of values has spacing equal to the spacing
    of the immediately preceding pair multiplied by `factor`.

    """
    l0 = span / sum([factor**i for i in range(n)])  # first interval
    x = np.cumsum([start] + [l0 * factor**i for i in range(n)])
    return x


def logspaced(offset, span, n, dmin=None):
    """Return a series of n log-spaced values x.

    The spacing scales as ln(x), with:

    offset := Distance between x = 0 and the first value in the series.  Non-negative.

    span := Distance between the first and last value in the series.  Non-negative.

    n := Number of values in the series.  Non-negative.

    dmin := (optional) Distance between the first and second value in the series,
    applied if offset = 0.  The default is `span * (1/n)**e`, which has no particular
    rationale but seems to produce acceptable results.

    """
    if span < 0:
        raise ValueError(f"Span, {span}, must be non-negative.")
    elif span == 0:
        return np.zeros(n) + offset
    if n < 0:
        raise ValueError(f"Number of samples, n = {n}, must be non-negative.")
    elif n == 0:
        return np.array([])
    elif n == 1 and span != 0:
        raise ValueError(f"Cannot span distance of {span} with n = {n} value.")
    if offset < 0:
        raise ValueError(f"Offset, {offset}, must be non-negative.")
    elif offset == 0:
        if dmin is None:
            dmin = (1 / n) ** e * span
        x = np.zeros(n)
        x[1:] = np.geomspace(span, dmin, n - 1)[::-1]
        # ^ work reversed so geomspace generates end point for n = 2.
    else:
        x = np.geomspace(offset, offset + span, n)
    # numpy.geomspace introduces some extra floating point error, so
    # set the endpoints to input values
    x[0] = offset
    x[-1] = offset + span
    return x


def powerspaced(offset, span, n, power, dmin=None):
    """Return a series of n power-spaced values x.

    The spacing scales as x^power, with:

    offset := Distance between x = 0 and the first value in the series.  Non-negative.

    span := Distance between the first and last value in the series.  Non-negative.

    n := Number of values in the series.  Non-negative.

    power := the exponent in the spacing ~ x^power relationship.

    dmin := (optional) Distance between the first and second value in
    the series, applied if offset = 0.  The default is `span *
    (1/n)**(-1/power)`, which has no particular rationale but seems to
    produce acceptable results.

    """
    if offset < 0:
        raise ValueError(f"Offset, {offset}, must be non-negative.")
    if span < 0:
        raise ValueError(f"Span, {span}, must be non-negative.")
    elif span == 0:
        return np.zeros(n) + offset
    if n < 0:
        raise ValueError(f"Number of samples, n = {n}, must be non-negative.")
    elif n == 0:
        return np.array([])
    elif n == 1 and span != 0:
        raise ValueError(f"Cannot span distance of {span} with n = {n} value.")
    if power == 0:
        # The normal code path assumes that `f(x) = x^power` has an
        # inverse.
        return linspaced(offset, span, n)
    elif power < 0 and offset == 0:
        # Handle singularity at x = 0 for negative powers
        if dmin is None:
            dmin = (1 / n) ** (-1 / power) * span
        x = np.zeros(n)
        x[1:] = np.linspace((offset + span) ** power, dmin**power, n - 1)[::-1] ** (
            1 / power
        )
        # ^ work reversed so linspace generates end point for n = 2.
    else:  # power > 0
        x = np.linspace(offset**power, (offset + span) ** power, n) ** (1 / power)
    return x


def densify(curve, n):
    """Return curve with more points in between each pair of points

    :param n: For each pair of points A and B in the input curve, n is the number of
    points in the half-open interval (A, B] in the output curve.

    """
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
