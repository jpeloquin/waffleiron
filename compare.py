#!/usr/bin/python

"""Compares two mesh solutions with equivalent geometry.

Features:
* Calculate relative stress difference between two solutions
* Calculate relative strain ' '
* Plot relative differences

"""

import os
import matplotlib.pyplot as plt
import febtools
import matplotlib.mlab as mlab
import numpy as np

def midplane_stress(soln):
    """Finds stress at z = 0 plane from MeshSolution.

    """
    v = []
    for stress, centroid in zip(soln.s(), soln.elemcentroid()):
        x, y, z = centroid
        if abs(z) < 1e-7:
            v.append((stress, x, y, z))
    return v

def plot_stress(s, x, y, title, fsave, delta=50e-6):
    """Shaded plot of stress.

    """
    xi = np.arange(min(x), max(x), delta)
    yi = np.arange(min(y), max(y), delta)
    v = mlab.griddata(x, y, s, xi, yi, interp='nn')
    plt.figure()
    #plt.imshow(v, interpolation='bilinear',
    #           extent=[min(x), max(x), min(y), max(y)],
    #           origin='lower', vmin=0)
    plt.contourf(xi*1e3, yi*1e3, v, 100, antialiased=True)
    plt.colorbar()
    plt.title(title)
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    fout = fsave + '.png'
    plt.savefig(fout, dpi=300)
    print('Wrote ' + fout)
    fout = fsave + '.pdf'
    plt.savefig(fout)
    print('Wrote ' + fout)

def relplot(fa, fb, outname):
    """Plots stresses from file A relative to B.

    """

    def sol(f):
        """Get MeshSolution object from f"""
        if isinstance(f, str):
            return febtools.MeshSolution(f)
        elif isinstance(f, febtools.MeshSolution):
            return f
        else:
            raise Exception("f is neither a string nor a "
                            "MeshSolution object.")
    asol = sol(fa)
    bsol = sol(fb)
    adata = midplane_stress(asol)
    bdata = midplane_stress(bsol)
    x, y, z = zip(*[v[1:] for v in adata])
    idx = ((0,0), (1,1), (2,2))
    sub = ('_sxx', '_syy', '_szz')
    tstr = ('$s_{11}$', '$s_{22}$', '$s_{33}$')
    for i, s, ts in zip(idx, sub, tstr):
        fsave = os.path.join(outname + s)
        v = [a[0][i] / b[0][i] for a, b in zip(adata, bdata)]
        plot_stress(v, x, y, 'Relative stress: ' + ts, fsave)
