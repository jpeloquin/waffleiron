from math import ceil

# Public repo packages
import numpy as np
from numpy.linalg import norm
from shapely.geometry import LineString, Point, Polygon

# Febtools modules
from .core import NodeSet
from .geometry import pt_series
from .element import Hex8, Quad4
from .model import Mesh


def zstack(mesh, zcoords):
    """Stack a 2d mesh in the z direction to make a 3d mesh.

    Arguments
    ---------
    zcoords -- The z-coordinate of each layer of nodes in the stacked
    mesh.  The number of element layers will be one less than the
    length of zcoords.

    Material properties are preserved.  Boundary conditions are not.

    """
    # `zstack` is here instead of in util.py because a it's used
    # extensively in this module to build 3D meshes.
    #
    # Create 3d node list
    nodes = []
    for z in zcoords:
        node_layer = [(pt[0], pt[1], z) for pt in mesh.nodes]
        nodes = nodes + node_layer

    # Create elements
    eid = 0
    elements = []
    # Iterate over element layers
    for i in range(len(zcoords) - 1):
        # Iterate over elements in 2d mesh
        for e2d in mesh.elements:
            nids = [a + i * len(mesh.nodes) for a in e2d.ids] + [
                a + (i + 1) * len(mesh.nodes) for a in e2d.ids
            ]
            if isinstance(e2d, Quad4):
                cls = Hex8
            else:
                raise ValueError("Only Quad4 meshes can be used in zstack right now.")
            e3d = cls.from_ids(nids, nodes, mat=e2d.material)
            elements.append(e3d)

    mesh3d = Mesh(nodes=nodes, elements=elements)
    return mesh3d


def rectangular_prism(length, width, thickness, material=None):
    """Create an FE mesh of a rectangular prism.

    Each key dimension is a tuple of (length, number of elements).
    Element spacing is linear.

    The origin is in the center of the rectangle.

    """
    l = length[0]
    nl = length[1]
    w = width[0]
    nw = width[1]
    t = thickness[0]
    nt = thickness[1]
    # Create rectangle in xy plane
    A = np.array([-l / 2, -w / 2])
    B = np.array([l / 2, -w / 2])
    C = np.array([l / 2, w / 2])
    D = np.array([-l / 2, w / 2])
    AB = pt_series([A, B], nl + 1)
    DC = pt_series([D, C], nl + 1)
    AD = pt_series([A, D], nw + 1)
    BC = pt_series([B, C], nw + 1)
    mesh = quadrilateral(AD, BC, AB, DC)
    # Create rectangular prism
    zi = np.linspace(-t / 2, t / 2, nt + 1)
    mesh = zstack(mesh, zi)
    # Label the faces
    nodes = np.array(mesh.nodes)  # remove when array is default
    # ±x faces
    ns = NodeSet(
        np.nonzero(np.isclose(nodes[:, 0], -l / 2, rtol=0, atol=np.spacing(l / 2)))[0]
    )
    assert len(ns) == (nw + 1) * (nt + 1)
    mesh.named["node sets"].add("−x1 face", ns)
    ns = NodeSet(
        np.nonzero(np.isclose(nodes[:, 0], l / 2, rtol=0, atol=np.spacing(l / 2)))[0]
    )
    assert len(ns) == (nw + 1) * (nt + 1)
    mesh.named["node sets"].add("+x1 face", ns)
    # ±y faces
    ns = NodeSet(
        np.nonzero(np.isclose(nodes[:, 1], -w / 2, rtol=0, atol=np.spacing(w / 2)))[0]
    )
    assert len(ns) == (nl + 1) * (nt + 1)
    mesh.named["node sets"].add("−x2 face", ns)
    ns = NodeSet(
        np.nonzero(np.isclose(nodes[:, 1], w / 2, rtol=0, atol=np.spacing(w / 2)))[0]
    )
    assert len(ns) == (nl + 1) * (nt + 1)
    mesh.named["node sets"].add("+x2 face", ns)
    # ±z faces
    ns = NodeSet(
        np.nonzero(np.isclose(nodes[:, 2], -t / 2, rtol=0, atol=np.spacing(t / 2)))[0]
    )
    assert len(ns) == (nl + 1) * (nw + 1)
    mesh.named["node sets"].add("−x3 face", ns)
    ns = NodeSet(
        np.nonzero(np.isclose(nodes[:, 2], t / 2, rtol=0, atol=np.spacing(t / 2)))[0]
    )
    assert len(ns) == (nl + 1) * (nw + 1)
    mesh.named["node sets"].add("+x3 face", ns)
    # Assign material
    if material is not None:
        for e in mesh.elements:
            e.material = material
    return mesh


def quadrilateral(col1, col2, row1, row2):
    """Mesh a quadrilateral with quad elements.

    Each input variable is a list of (x, y) points specifying node
    locations on the boundary of the mesh domain.  `col1` and `col2` are
    opposite edges, as are `row1` and `row2`.  Row1 must be below (−y)
    row2 and col1 should be left (−x) of col2 in a right hand coordinate
    system.

    """

    def pdist(p1, p2):
        """Distance between two points."""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    # Ensure col1 lines up with the start of row1
    d_row1p0_col1 = min(pdist(row1[0], col1[i]) for i in [0, -1])
    d_row1p0_col2 = min(pdist(row1[0], col2[i]) for i in [0, -1])
    if d_row1p0_col1 > d_row1p0_col2:
        # col2 intersects the start of row1, so swap col2 and col1
        col1, col2 = col2, col1

    # Ensure both columns start on row1
    if pdist(row1[0], col1[0]) > pdist(row1[0], col1[-1]):
        col1 = col1[::-1]
    if pdist(row1[-1], col2[0]) > pdist(row1[-1], col2[-1]):
        col2 = col2[::-1]

    # Ensure the row2 starts on col1 (parallels row1)
    if pdist(col1[-1], row2[0]) > pdist(col1[-1], row2[-1]):
        row2 = row2[::-1]

    # Compute corresponding normalized path length for each point
    def spts_norm(pts):
        s = [0]
        for i in range(1, len(pts)):
            p0 = np.array(pts[i - 1])
            p1 = np.array(pts[i])
            s.append(s[i - 1] + np.linalg.norm(p1 - p0))
        stot = s[-1]
        snorm = [a / stot for a in s]
        return snorm

    col1_s = spts_norm(col1)
    col2_s = spts_norm(col2)
    row1_s = spts_norm(row1)
    row2_s = spts_norm(row2)

    # Ensure the the node counts are compatible
    assert len(row1) == len(row2)
    assert len(col1) == len(col2)
    nr = len(col1)
    nc = len(row1)

    # For debugging
    #
    # import matplotlib.pyplot as plt
    # plt.ion()
    # plt.plot(np.array(row1)[:,0], np.array(row1)[:,1], 'k-o')
    # plt.plot(np.array(row2)[:,0], np.array(row2)[:,1], 'k-*')
    # plt.plot(np.array(col1)[:,0], np.array(col1)[:,1], 'r-o')
    # plt.plot(np.array(col2)[:,0], np.array(col2)[:,1], 'r-*')
    # plt.show()

    # Create nodes

    # initialize loop
    prevrow_pts = row1
    elements = []
    nodes = prevrow_pts
    # Loop over rows of nodes, creating one row at a time, from -x2 to
    # +x2.  We already initialized the -x2 most row above.
    for i in range(1, nr):

        # Add interpolated points for this row
        thisrow_pts = [col1[i]]
        v_c = np.array(col2[i]) - np.array(col1[i])
        u_c = v_c / np.linalg.norm(v_c)
        for j in range(1, nc - 1):
            v_r = np.array(row2[j] - row1[j])
            u_v = v_r / np.linalg.norm(v_r)
            ln_r = LineString([col1[i] - v_c, col2[i] + v_c])
            ln_c = LineString([row1[j] - v_r, row2[j] + v_r])
            # ^ extend the lines a little bit to avoid missing an
            # intersection due to finite precision
            pt = ln_r.intersection(ln_c)
            if pt.is_empty:
                raise Exception(
                    "When placing a node, no "
                    "intersection was found between "
                    "lines drawn between gridlines. "
                    "Check the input geometry."
                )
            pt = pt.coords[0]
            thisrow_pts.append(pt)
        thisrow_pts.append(col2[i])
        nodes = nodes + thisrow_pts

    # Stitch elements
    for i in range(1, nr):
        # some setup
        n_base = (i - 1) * nc  # number of nodes already fully stitched
        prevrow_pts = thisrow_pts
        # do the actual stitching
        for j in range(nc - 1):
            e = np.array([0, 1, nc + 1, nc]) + n_base + j
            elements.append(e)

    mesh = Mesh.from_ids(nodes, elements, Quad4)

    return mesh
