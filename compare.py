#!/usr/bin/python

"""Compares two mesh solutions with equivalent geometry.

Features:
* Calculate relative stress difference between two solutions
* Calculate relative strain ' '
* Plot relative differences

"""

import os
import re
import matplotlib.pyplot as plt
import febtools
import matplotlib.mlab as mlab
import numpy as np
import math
import gc

def midplane_stress(soln, type='1pk'):
    """Finds stress at z = 0 plane from MeshSolution.

    """
    v = []
    if type == '1pk':
        s = soln.s()
    elif type == 'cauchy':
        s = soln.t()
    else:
        raise Exception("Stress type %s not recognized." % type)
    for stress, centroid in zip(s, soln.elemcentroid()):
        x, y, z = centroid
        if abs(z) < 1e-7:
            v.append((stress, x, y, z))
    return v

def relative_stress(va, vb):
    """Computes stress va / vb.

    Input:

    `va` and `vb` are lists of tuples (`stress`, `x`, `y`, `z`) where
    `stress` is a 3x3 `numpy` array and `x`, `y`, and `z` are numeric
    coordinates.

    Make sure the stresses are defined in comparable coordinate
    systems (i.e. 1st P-K stress).

    Output:

    A list of tuples in the same format and the input lists.

    """
    x1 = np.array([c[1:] for c in va])
    x2 = np.array([c[1:] for c in vb])
    d = abs(x1 - x2)
    tol = 1e-15
    if d.max() > tol:
        raise Exception('Element coordinates do not match between '
                        'inputs (tolerance %s)' % str(tol))
    stress = [(a[0] / b[0]) for a, b in zip(va, vb)]
    return zip(stress, x2[:,0], x2[:,1], x2[:,2])

def plot_stress(s, x, y, title, fsave, delta=50e-6, clabel=""):
    """Shaded plot of stress.

    """
    xi = np.arange(min(x), max(x), delta)
    yi = np.arange(min(y), max(y), delta)
    v = mlab.griddata(x, y, s, xi, yi, interp='nn')
    if clabel:
        i = int(math.log(v.max(), 10) / 3)
        if i >= 1:
            v = v * (i * 1e-3)
        scaleprefix = ["", 'k', 'M', 'G', 'T'][i]
    else:
        scaleprefix = ""
    hf = plt.figure()
    plt.imshow(v, interpolation='bilinear',
               extent=[min(x), max(x), min(y), max(y)],
               origin='lower')
    # plt.contourf(xi*1e3, yi*1e3, v, 100, antialiased=False)
    plt.colorbar().set_label(scaleprefix + clabel)
    plt.title(title)
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    fout = fsave + '.png'
    plt.savefig(fout, dpi=300)
    print('Wrote ' + fout)
    fout = fsave + '.pdf'
    plt.savefig(fout, dpi=300)
    print('Wrote ' + fout)
    plt.close()

def stressplot(v, title, fsave, clabel=""):
    """Plots s_11, s_22, and s_33 in a 2D plane.

    Usage:

    `stressplot(v, 'outfile')`

    `v` is a list of tuples (`stress`, `x`, `y`, `z`) where `stress`
    is a 3x3 `numpy` array and `x`, `y`, and `z` are numeric
    coordinates.

    `'outfile'` is a file name for the output plots.  The appropriate
    file extension will be added automatically if not present.

    """
    a = re.search(r"(?P<name>.+)(?P<ext>\.xplt)?", fsave)
    outname = a.group('name')
    x, y, z = zip(*[t[1:] for t in v])
    idx = ((0,0), (1,1), (2,2))
    sub = ('_sxx', '_syy', '_szz')
    tstr = (' x', ' y', ' z')
    for i, s_f, s_title in zip(idx, sub, tstr):
        thisf = os.path.join(fsave + s_f)
        s = [t[0][i] for t in v]
        plot_stress(s, np.array(x)*10**3, np.array(y)*10**3,
                    title + s_title, thisf,
                    delta=50e-3, clabel=clabel)
