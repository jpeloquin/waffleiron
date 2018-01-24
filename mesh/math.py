import numpy as np

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

def bias_pt_series(line, n=None, type='log', minstep=None,
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

    if type == 'log':
        fspc = biasrange_log
    elif type == 'linear':
        fspc = np.linspace

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

def even_pt_series(line, n):
    """Return a list of n points evenly spaced along line segment.

    line := a list of n-tuples (points), or equivalent iterable

    The returned points are numpy arrays of dimension equal to A and B.

    """
    A = np.array(line[0])
    B = np.array(line[1])
    v = B - A
    return [A + s * v for s in np.linspace(0, 1, n)]
