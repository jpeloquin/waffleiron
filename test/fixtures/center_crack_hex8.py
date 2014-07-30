# -*- coding: utf-8 -*-
"""Create a center crack mesh with a low number of elements.

"""
import distmesh as dm
from distmesh import distmesh2d, huniform, dpoly
import numpy as np
from math import cos, sin, pi, radians
import matplotlib.pyplot as plt
from shapely.ops import polygonize, polygonize_full
import shapely.geometry as geo
from shapely.geometry import LineString, Point, Polygon
import febtools as feb
import os
import triangle
from copy import deepcopy

from operator import itemgetter

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
        raise ValueError
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


def biased_range(start, stop, minstep, type='log'):
    """Log spaced series with minimum step size, finer near start.

    """
    n = 2
    x = biasrange_log(start, stop, n)
    while abs(x[1] - x[0]) > minstep:
        n = n + 1
        x = biasrange_log(start, stop, n)
    return x

def mesh_quad_quad(col1, col2, row1, row2):
    """Mesh a quadrilateral with quad elements.

    Each input variable is a list of (x, y) points specifying node locations
    on the boundary of the mesh domain.  `col1` and `col2` are
    opposite edges, as are `row1` and `row2`.

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
    if pdist(row1[0], col2[0]) > pdist(row1[0], col2[-1]):
        col2 = col2[::-1]

    # Ensure the row2 starts on col1 (parallels row1)
    if pdist(col1[-1], row2[0]) > pdist(col1[-1], row2[-1]):
        row2 = row2[::-1]

    # Compute corresponding normalized path length for each point
    def spts_norm(pts):
        s = [0]
        for i in xrange(1, len(pts)):
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

    ## Create nodes

    # initialize loop
    prevrow_pts = row1
    elements = []
    nodes = prevrow_pts
    # loop over rows of nodes, creating one row at a time
    for i in xrange(1, nr - 1):

        # Add interpolated points for this row
        thisrow_pts = [col1[i]]
        v = np.array(col1[i]) - np.array(col2[i])
        for j in xrange(1, nc - 1):
            ln_r = LineString([col1[i], col2[i]])
            ln_c = LineString([row1[j], row2[j]])
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
    nodes = nodes + row2

    ## Stitch elements
    for i in xrange(1, nr):
        # some setup
        n_base = (i - 1) * nc # number of nodes already fully stitched
        prevrow_pts = thisrow_pts
        # do the actual stitching
        for j in xrange(nc - 1):
            e = np.array([0, 1, nc + 1, nc]) + n_base + j
            elements.append(e)

    mesh = feb.Mesh.from_ids(nodes, elements, feb.element.Quad4)

    return mesh

def center_mesh_quad(height=20.0e-3, width=10.0e-3,
                     notch_length=2.0e-3,
                     hmin=0.05e-3, tol=1e-8):
    """Mesh a center-cracked specimen with quad4 elements.

    """
    # bb = [xmin, ymin, xmax, ymax]
    bb = [-0.5 * width, -0.5 * height, 0.5 * width, 0.5 * height]
    pt_tip_left = (-0.5 * notch_length, 0.0, 0.0)
    pt_tip_right = (0.5 * notch_length, 0.0, 0.0)
    r1 = bias_pt_series([(bb[0], 0, 0), pt_tip_left],
                        minstep=hmin, bias_direction=-1)
    r2 = bias_pt_series([pt_tip_left, (0, 0, 0)],
                        minstep=hmin, bias_direction=1)
    row1 = r1[:-1] + \
           r2[:-1] + \
           [(-x, y, z) for (x, y, z) in r2[:0:-1]] + \
           [(-x, y, z) for (x, y, z) in r1[::-1]]
    row2 = [(x, bb[3], z) for (x, y, z) in row1]
    col1 = bias_pt_series([(bb[0], 0, 0), (bb[0], bb[3], 0)],
                          minstep=hmin, bias_direction=1)
    col2 = [(-x, y, z) for (x, y, z) in col1]
    mesh = mesh_quad_quad(col1, col2, row1, row2)
    col1 = [(x, -y, z) for (x, y, z) in col1]
    col2 = [(x, -y, z) for (x, y, z) in col2]
    row2 = [(x, -y, z) for (x, y, z) in row2]
    mesh2 = mesh_quad_quad(col1, col2, row2, row1)
    ids = [i for i, (x, y, z) in enumerate(mesh2.nodes)
           if (y == 0.0 and
               (x <= (pt_tip_left[0] + tol) or
                x >= (pt_tip_right[0] - tol)))]
    mesh.merge(mesh2, candidates=ids)
    return mesh

def sent_mesh_quad(angle, notch_length=2.0e-3,
                   height=20.0e-3, width=10.0e-3,
                   hmin=0.05e-3, tol=1e-8):
    """Notch a SENT specimen with quadrilateral elements.

    angle := the notch angle in degrees, measured ccw from +x

    The notch is on the the right (+x).

    """

    # all units in mks

    bb_xmin = 0.0
    bb_xmax = width
    bb_ymax = 0.5 * height
    bb_ymin = -0.5 * height

    bbox = [(width, 0.5 * height), # upper right
            (0.0, 0.5 * height), # upper left
            (0.0, -0.5 * height), # lower left
            (width, -0.5 * height)] # lower right

    poly_specimen = Polygon(bbox)

    # Define notch line
    # notch origin
    pt_og = (width, 0.0)
    # notch tip
    pt_tip = (pt_og[0] - (notch_length * cos(radians(angle))),
              pt_og[1] - (notch_length * sin(radians(angle))))
    # notch line
    ln_notch = LineString([pt_tip, pt_og])

    # Define biased grid points that are common between subdomains

    # to left of notch tip
    gridl = bias_pt_series([(bb_xmin, pt_tip[1]), pt_tip],
                           minstep=hmin, bias_direction=-1)
    # to right of notch tip
    gridr = bias_pt_series([pt_tip, pt_og],
                           minstep=hmin)
    # up from notch tip
    gridu = bias_pt_series([pt_tip, (pt_tip[0], bb_ymax)],
                           minstep=hmin)
    # down from notch tip
    gridd = bias_pt_series([pt_tip, (pt_tip[0], bb_ymin)],
                           minstep=hmin)

    # Subdomain 1
    # quadrant I with respect to the notch tip
    col1 = gridu
    col2 = bias_pt_series([pt_og, bbox[0]], n=len(gridu))
    row1 = gridr
    row2 = bias_pt_series([(pt_tip[0], bb_ymax), bbox[0]],
                          n=len(gridr))
    mesh1 = mesh_quad_quad(col1, col2, row1, row2)
    # Subdomain 2
    # quadrant II with respect to the notch tip
    col1 = bias_pt_series([(bb_xmin, pt_tip[1]), bbox[1]],
                          n=len(gridu))
    col2 = gridu
    row1 = gridl
    row2 = bias_pt_series([bbox[1], (pt_tip[0], bb_ymax)],
                          n=len(gridl), bias_direction=-1)
    mesh2 = mesh_quad_quad(col1, col2, row1, row2)
    # Subdomain 3
    # quadrant III with respect to the notch tip
    col1 = bias_pt_series([bbox[2], (bb_xmin, pt_tip[1])],
                          n=len(gridd), bias_direction=-1)
    col2 = gridd
    row1 = bias_pt_series([bbox[2], (pt_tip[0], bb_ymin)],
                          n=len(gridl), bias_direction=-1)
    row2 = gridl
    mesh3 = mesh_quad_quad(col1, col2, row1, row2)
    # Subdomain 4
    # quadrant IV with respect to the notch tip
    col1 = gridd
    col2 = bias_pt_series([bbox[3], pt_og],
                          n=len(gridd), bias_direction=-1)
    row1 = bias_pt_series([(pt_tip[0], bb_ymin), bbox[3]],
                          n=len(gridr))
    row2 = gridr
    mesh4 = mesh_quad_quad(col1, col2, row1, row2)

    ## Merge meshes
    mesh = mesh2
    mesh.merge(mesh3)
    mesh.merge(mesh1)
    # here, we need to avoid merging the notch face
    tol=1e-6
    idx_tipcolumn = [i for i, node in enumerate(mesh4.nodes)
                     if abs(node[0] - pt_tip[0]) < tol]
    mesh.merge(mesh4, candidates=idx_tipcolumn)
    return mesh


##############################
#  Create center crack mesh  #
##############################

thisdir = os.path.dirname(os.path.abspath(__file__))
zcoords = np.linspace(-0.5e-3, 0.5e-3, 5)
mesh = center_mesh_quad(width=10e-3, height=20e-3,
                        notch_length=2e-3, hmin=50e-6)
mesh = feb.meshing.zstack(mesh, zcoords)
model = feb.Model(mesh)

# Define material
props = {'E': 1e7,
         'v': 0.3}
mat = feb.material.IsotropicElastic(props)
for e in model.mesh.elements:
    e.material = mat

# Load curve for boundary conditions
seq_bc = feb.Sequence([(0, 0), (1, 1)])
# Load curve for must points
seq_dt = feb.Sequence([(0, 0), (1, 0.5)], typ='step')

# Identify faces
maxima = np.max(model.mesh.nodes, 0)
minima = np.min(model.mesh.nodes, 0)
up_face = [i for i, pt in enumerate(model.mesh.nodes)
           if pt[1] == maxima[1]]
down_face = [i for i, pt in enumerate(model.mesh.nodes)
             if pt[1] == minima[1]]

# Uniaxial stretch nodes
c = 0.001
dy = [c for i in up_face]
model.apply_nodal_displacement(up_face, values=dy, axis='y',
                               sequence=seq_bc)
# Fixed nodes
model.fixed_nodes['y'].update(down_face)
model.fixed_nodes['x'].update([i for i in up_face + down_face
                               if np.allclose(model.mesh.nodes[i][0], 0)])
back_corners = [i for i in feb.selection.corner_nodes(model.mesh)
                if model.mesh.nodes[i][2] < 0]
model.fixed_nodes['z'].update(back_corners)

# Save only final solution
model.steps[0]['control']['plot level'] = 'PLOT_MUST_POINTS'

# Adjust time steps
nsteps = 20
#model.steps[0]['control']['time steps'] = nsteps
#model.steps[0]['control']['step size'] = 1.0 / nsteps
model.steps[0]['control']['time stepper']['dtmin'] = 0.01
model.steps[0]['control']['time stepper']['dtmax'] = seq_dt

# Write .feb
fp_out = os.path.join(thisdir,
    'center_crack_uniax_isotropic_elastic_hex8.feb')
feb.output.write_feb(model, fp_out)
