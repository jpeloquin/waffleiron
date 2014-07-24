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
