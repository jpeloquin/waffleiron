from math import ceil
import numpy as np
from numpy.linalg import norm
from .quad import quadrilateral
from .util import zstack
from .math import even_pt_series
from .. import Model

def rectangular_prism(length, width, thickness, hmin):
    """Create an FE mesh of a rectangular prism.

    The origin is in the center of the rectangle.

    """
    if type(hmin) in [float, int]:
        hmin = [hmin]*3
    # Create rectangle in xy plane
    A = np.array([-length/2, -width/2])
    B = np.array([ length/2, -width/2])
    C = np.array([ length/2,  width/2])
    D = np.array([-length/2,  width/2])
    n_AB = ceil(norm(A-B)/hmin[0]) + 1
    AB = even_pt_series([A, B], n_AB)
    DC = even_pt_series([D, C], n_AB)
    n_BC = ceil(norm(B-C)/hmin[1]) + 1
    AD = even_pt_series([A, D], n_BC)
    BC = even_pt_series([B, C], n_BC)
    mesh = quadrilateral(AD, BC, AB, DC)
    # Create rectangular prism
    zi = np.linspace(-thickness/2, thickness/2, ceil(thickness/hmin[2]) + 1)
    mesh = zstack(mesh, zi)
    return mesh
