import numpy as np
import sys

def scalar_field(mesh, fn, pts):
    """Return a field evaluated over a grid of points.

    Inputs
    ------
    mesh := A Mesh object.

    fn := A function that takes an F tensor and an Element object and
    returns a scalar.

    pts := An n x m x 3 array of x, y, z values.  The z value may be
    omitted (in which case pts.shape == (n, m, 2)); if so, it will be
    assumed to be zero.

    Returns
    -------
    An n x m array of scalar values calculated by calling `fn` with
    the F tensor and `Element` for each point in `pts`.

    """
    # add z coordinates if omitted
    if pts.shape[2] == 2:
        zv = np.zeros(pts[:,:,0].shape)
        pts = np.concatenate([pts, zv[...,np.newaxis]], axis=2)
    
    field = np.empty(pts.shape[0:2])
    for i in xrange(pts.shape[0]):
        for j in xrange(pts.shape[1]):
            x = pts[i, j, 0]
            y = pts[i, j, 1]
            z = pts[i, j, 2]
            elems = mesh.elements_containing_point((x, y, z))
            if not elems:
                field[i, j] = None
            else:
                e = elems[0]
                r = e.to_natural((x, y, z))
                f = e.f(r)
                field[i, j] = fn(f, e)
        sys.stdout.write("\rLine {}/{}".format(i+1, field.shape[0]))
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()

    return field
