# Distributed packages
import numpy as np
from shapely.geometry import LineString, Point, Polygon
# Locally developed packages
import febtools as feb

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
            p0 = np.array(pts[i-1])
            p1 = np.array(pts[i])
            s.append(s[i-1] + np.linalg.norm(p1 - p0))
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

    ## Create nodes

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
                raise Exception("When placing a node, no "
                                "intersection was found between "
                                "lines drawn between gridlines. "
                                "Check the input geometry.")
            pt = pt.coords[0]
            thisrow_pts.append(pt)
        thisrow_pts.append(col2[i])
        nodes = nodes + thisrow_pts

    ## Stitch elements
    for i in range(1, nr):
        # some setup
        n_base = (i - 1) * nc # number of nodes already fully stitched
        prevrow_pts = thisrow_pts
        # do the actual stitching
        for j in range(nc - 1):
            e = np.array([0, 1, nc + 1, nc]) + n_base + j
            elements.append(e)

    mesh = feb.Mesh.from_ids(nodes, elements, feb.element.Quad4)

    return mesh
