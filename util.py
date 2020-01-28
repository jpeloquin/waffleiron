from math import inf
from warnings import warn
import numpy as np

from febtools import Sequence


def mesh_from_elems(elems):
    """Get nodes, elements representation for a list of elements.

    Positional arguments:

    elems -- A list of elements, each a tuple of points.

    Returns:

    nodes -- An (m, n) array of m node coordinates in n-space.
    Duplicate nodes are merged.

    elements -- Array.  `elements[i, j]` is the jth node id in the ith
    element.  The node ids index into `nodes`.

    """
    seen = {}  # node ids indexed by point coords
    nodes = []
    # `seen` and `nodes` could be combined with an ordered set (needs
    # implementation) to save on memory.
    p_elems = elems  # elements as lists of points coords
    c_elems = []  # elements as node indices
    for p_e in p_elems:
        c_e = []
        for p in p_e:
            if p in seen:
                c_e.append(seen[p])
            else:
                i = len(nodes)
                nodes.append(p)
                seen[p] = i
                c_e.append(i)
        c_elems.append(c_e)
    return nodes, c_elems


def apply_uniax_stretch(model, stretches, axis='x1'):
    """Apply stretch with must points and a fixed width grip line.

    axis := 'x1' or 'x2'; the direction along which to stretch the mesh.
    Displacement is prescribed to the node(s) furthest +ve and -ve along
    the specified axis.

    """
    axis1 = axis
    if axis == 'x1':
        axis2 = 'x2'
        iaxis1 = 0
        iaxis2 = 1
    elif axis == 'x2':
        axis2 = 'x1'
        iaxis1 = 1
        iaxis2 = 0
    else:
        msg = "`axis` must be 'x1' or 'x2'; {} was provided"
        raise ValueError(msg.format(axis))
    maxima = np.max(model.mesh.nodes, 0)
    minima = np.min(model.mesh.nodes, 0)
    # Moving (gripped) nodes
    gripped_nodes = [i for i, x in enumerate(model.mesh.nodes)
                     if x[iaxis1] == minima[iaxis1] or
                     x[iaxis1] == maxima[iaxis1]]
    u1 = [(stretches[-1] - 1) * model.mesh.nodes[i][iaxis1]
          for i in gripped_nodes]
    # ^ gripped node displacements at final timepoint
    seq_bc = Sequence([(0, 0), (1, 1)])
    model.apply_nodal_displacement(gripped_nodes, values=u1, axis=axis1,
                                   sequence=seq_bc)
    # Fixed nodes
    model.fixed['node'][axis2].update(gripped_nodes)
    gripped_nodes_back = [i for i in gripped_nodes
                          if model.mesh.nodes[i][2] == minima[2]]
    model.fixed['node']['x3'].update(gripped_nodes_back)
    # Define number of steps
    nsteps = len(stretches) * 1 + 1
    nmust = len(stretches) * 1 + 1
    dtmax = 1.0 / nsteps
    # Calculate must points to match input stretches
    t_must = [(u - 1) / (stretches[-1] - 1) for u in stretches]
    seq_must = Sequence([(t, dtmax) for t in t_must],
                        typ="step")
    model.steps[0]['control']['plot level'] = 'PLOT_MUST_POINTS'
    model.steps[0]['control']['time stepper']['dtmax'] = seq_must


def find_closest_timestep(target, times, steps, rtol=0.01, atol="auto"):
    """Return step index closest to given time."""
    times = np.array(times)
    if atol == "auto":
        atol = max(abs(np.nextafter(target, 0) - target),
                   abs(np.nextafter(target, target**2) - target))
    if len(steps) != len(times):
        raise ValueError("len(steps) ≠ len(times).  All steps must have a corresponding time value and vice versa.")
    idx = np.argmin(np.abs(times - target))
    step = steps[idx]
    time = times[idx]
    # Raise a warning if the specified value is not close to a step.  In
    # the future this function may support interpolation or other fixes.
    t_error = time - target
    if t_error == 0:
        t_relerror = 0
    elif idx == 0 and t_error < 0:
        # The selection specifies a time point before the first given step
        t_interval = (times[idx + 1] - times[idx])
        t_relerror = t_error / t_interval
    elif step == steps[-1] and t_error > 0:
        # The selection specifies a time point after the
        # last time step.  It might only be a little
        # after, within acceptable tolerance when
        # working with floating point values, so we do
        # not raise an error until further checks.
        t_interval = (times[idx] - times[idx-1])
        t_relerror = t_error / t_interval
    else:
        t_interval = abs(times[idx] -
                        times[idx + int(np.sign(t_error))])
        t_relerror = t_error / t_interval
    # Check error tolerance
    if abs(t_error) > atol:
        raise ValueError(f"Time step selection absolute error > atol; target time — selected time = {t_error}; atol = {atol}.")
    if abs(t_relerror) > abs(rtol):
        raise ValueError(f"Time step selection relative error > rtol; target time — selected time = {t_error}; step interval = {t_interval}; relative error = {t_relerror}; rtol = {rtol}.")
    return step
