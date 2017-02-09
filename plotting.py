# -*- coding: utf-8 -*-
import sys

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tvtk.api import tvtk

import febtools as feb

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

    bb = feb.core._e_bb(mesh.elements)

    field = np.empty(pts.shape[0:2])
    for i in range(pts.shape[0]):
        for j in range(pts.shape[1]):
            x = pts[i, j, 0]
            y = pts[i, j, 1]
            z = pts[i, j, 2]
            elems = feb.selection.elements_containing_point((x, y, z), mesh.elements, bb=bb)
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

def plot_q(elements, length=1.0):
    """Plot nodal q vectors in elements.

    Requires matplotlib ≥ 1.4.0

    """
    # get subset of elements that actually has q values
    qelements = [e for e in elements if 'q' in e.properties]
    # get list of q values
    q = np.array([v for e in qelements for v in e.properties['q']])
    qnodes = np.array([x for e in qelements for x in e.nodes])
    nodes = np.array([x for e in elements for x in e.nodes])
    # plot vectors
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(qnodes[:,0], qnodes[:,1], qnodes[:,2],
               c='k', edgecolor='k', s=4)
    # the vector length is added to the node locations because
    # matplotlib draws the arrows such that the head is at the
    # provided point
    ax.quiver(qnodes[:,0] + length*q[:,0],
              qnodes[:,1] + length*q[:,1],
              qnodes[:,2] + length*q[:,2],
              q[:,0], q[:,1], q[:,2],
              color='r', length=length)
    # plot 0-values
    #xyz = nodes[~np.any(q, axis=1)]
    #ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2],
#               s=16, c='r', marker='*', edgecolor='r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig, ax
