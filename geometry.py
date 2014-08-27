import numpy as np

def _cross(u, v):
    """Cross product for two vectors in R3.

    """
    w = np.array([u[1]*v[2] - u[2]*v[1],
                  u[2]*v[0] - u[0]*v[2],
                  u[0]*v[1] - u[1]*v[0]])
    return w

def face_normal(mesh, face):
    """Return a vector normal to a face.

    """
    pts = [mesh.nodes[i] for i in face]
    v1 = points[f[1]] - points[f[0]]
    v2 = points[f[-1]] - points[f[0]]
    n = _cross(v1, v2)
    return n

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
        raise Exception("Element2D.edges() needs implementation.")
        bdry_pts = [e.x()[ids[0]] for ids in e.edges()]
    else:
        normals = e.face_normals()
        bdry_pts = [e.x()[ids[0]] for ids in e.faces()]
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
