# -*- coding: utf-8 -*-
"""Create a center crack mesh with a low number of elements.

"""
# Standard packages
import os
from math import cos, sin, pi, radians
# Distributed packages
import numpy as np
# Locally developed packages
import febtools as feb
from febtools.mesh.math import bias_pt_series
from febtools.mesh.quad import mesh_quad_quad

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


##############################
#  Create center crack mesh  #
##############################

thisdir = os.path.dirname(os.path.abspath(__file__))
zcoords = np.linspace(-0.5e-3, 0.5e-3, 5)
mesh = center_mesh_quad(width=10e-3, height=20e-3,
                        notch_length=2e-3, hmin=100e-6)
mesh = feb.meshing.zstack(mesh, zcoords)
model = feb.Model(mesh)

# Define material
props = {'E': 1e9,
         'v': 0.0}
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
c = 0.0001
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
