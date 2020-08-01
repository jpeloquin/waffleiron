from math import acos

import numpy as np

tol = np.finfo(float).eps


def cross(u, v):
    """Cross product for two vectors in R3.

    """
    w = np.array(
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    )
    return w


def face_normal(face, mesh, config="reference"):
    """Return the face normal vector.

    config := only 'reference' supported right now

    """
    points = np.vstack([mesh.nodes[i] for i in face])
    # Define vectors for two face edges, using the first face node
    # as the origin.  For quadrilateral faces, one node is left
    # unused.
    v1 = points[1] - points[0]
    v2 = points[-1] - points[0]
    # compute the face normal
    normal = cross(v1, v2)
    return normal


def inter_face_angle(f1, f2, mesh, tol=2 * np.finfo(float).eps):
    """Compute angle (radians) between two faces.

    The faces are oriented tuples of node ids.  The angle returned is
    in [0, π].

    """
    v1 = face_normal(f1, mesh)
    v2 = face_normal(f2, mesh)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    b = np.dot(v1, v2) / n1 / n2
    if -1 - tol < b < -1:
        b = -1
    elif 1 < b < 1 + tol:
        b = 1
    angle = acos(b)
    return angle


def point_in_element(e, p, config="reference", dtol=tol, rtol=0.05):
    """Return true if element encloses point.

    Points on the boundary of the element are considered to be inside
    the element.

    """
    p = np.array(p)
    # Find the normal and a point on each boundary edge/face.  The
    # normals point outward by convention.
    if e.is_planar:
        normals = e.edge_normals()
        bdry_pts = [e.x(config=config)[ids[0]] for ids in e.edge_nodes]
        # Find the distance to each boundary face by projection onto
        # the face normal.  If any projection is positive, the point
        # must lie outside that face.  If all projections are negative
        # or zero, the point is inside the element.
        for o, n in zip(bdry_pts, normals):
            v = p - o
            d = np.dot(v, n)
            if d > tol:
                return False
        return True
    else:
        r = e.to_natural(p, config=config)
        return np.all(
            np.logical_and(np.greater_equal(r, -1 - rtol), np.less_equal(r, 1 + rtol))
        )


def pt_series(line, n=None, bias="linear", minstep=None, bias_direction=1):
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
    p1 = np.array(line[0])  # origin of directed line segment
    p2 = np.array(line[1])
    v = p2 - p1
    length = np.linalg.norm(v)  # line length
    if length == 0.0:
        raise ValueError("Line has zero length.")
    u = v / length  # unit vector pointing in line direction

    if bias == "log":
        fspc = biasrange_log
    elif bias == "linear":
        fspc = np.linspace
    elif bias == "sqrt":
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
        raise Exception(
            "The number of points `n` and the minimum "
            "distance between points `minstep` are both defined; "
            "only one can be defined at a time."
        )

    # Compute the points
    if bias_direction == -1:
        s = [(1 - sc) for sc in s][::-1]
    pts = [sc * length * u + p1 for sc in s]

    return pts
