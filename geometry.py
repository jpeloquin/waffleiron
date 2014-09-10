# -*- coding: utf-8 -*-
from math import acos

import numpy as np

def _cross(u, v):
    """Cross product for two vectors in R3.

    """
    w = np.array([u[1]*v[2] - u[2]*v[1],
                  u[2]*v[0] - u[0]*v[2],
                  u[0]*v[1] - u[1]*v[0]])
    return w

def face_normal(face, mesh, config='reference'):
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
    normal = _cross(v1, v2)
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


def point_in_element(e, p):
    """Return true if element encloses point.

    Points on the boundary of the element are considered to be inside
    the element.

    """
    p = np.array(p)
    # Find the normal and a point on each boundary edge/face.  The
    # normals point outward by convention.
    if e.is_planar:
        normals = e.edge_normals()
        bdry_pts = [e.x()[ids[0]] for ids in e.edge_nodes]
    else:
        normals = e.face_normals()
        # these ids are intra-element
        bdry_pts = [e.x()[ids[0]] for ids in e.face_nodes]
    # Find the distance to each boundary face by projection onto the
    # face normal.  If any projection is positive, the point must lie
    # outside that face.  If all projections are negative or zero, the
    # point is inside the element.
    for o, n in zip(bdry_pts, normals):
        v = p - o
        d = np.dot(v, n)
        if d > 0:
            return False
    return True
