"""Module for adding initial and boundary conditions to models."""
# Third party modules
import numpy as np
# Same-package modules
from .core import Sequence, NodeSet


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


def cyclic_stretch_sequence(targets, rate, n=1, baseline=1.0):
    """Return a Sequence representing cyclic stretch.

    targets := List of numbers.  Each value specifies the peak stretch
    ratio for a block of cycles, in order.

    rate := Number, or list of numbers.  The strain rate for each block
    of cycles.  If a list, there must be one value per block, in the
    same order as `targets`.

    n := integer, or list of integers (optional).  The number of cycles
    in each block of cyclic stretches.  If an integer is provided, it
    will be applied to all blocks.  If a list is provided, it must be
    the same length as `targets`, and the values will be applied to each
    block in turn.

    baseline := number.  The starting and ending stretch ratio for each cycle.

    """
    # Expand strain rate
    if not hasattr(rate, '__iter__'):
        rate = [rate for y in targets]

    # Expand n
    elif not hasattr(n, '__iter__'):
        n_by_block = [n for y in targets]
    else:
        n_by_block = n

    # Compute a list of peak and return-to-baseline points for every
    # cycle in every block, and their corresponding times and rates.
    stretches = [baseline]
    times = [0.0]
    for v, r, n in zip(targets, rate, n_by_block):
        for i in range(n):
            # Ascending ramp to peak
            dv = v - stretches[-1]
            dt = abs(dv / r)
            stretches.append(v)
            times.append(times[-1] + dt)
            # Descending ramp to baseline
            dv = baseline - stretches[-1]
            dt = abs(dv / r)
            stretches.append(baseline)
            times.append(times[-1] + dt)

    # Create [(time, eng. strain), ...] curve
    curve = [(t, y - 1.0) for t, y in zip(times, stretches)]
    sequence = Sequence(curve, extend='constant', typ='linear')

    return sequence


def prescribe_deformation(model, node_ids, F, sequence, **kwargs):
    """Prescribe nodal displacements to match given F tensor."""
    # Need list of node IDs with stable order; we might have been given
    # a set
    id_list = np.array([i for i in node_ids])
    x_old = model.mesh.nodes[id_list]
    x_new = (F @ x_old.T).T
    u = x_new - x_old
    for i, node_id in enumerate(id_list):
        for iax, ax in enumerate(("x1", "x2", "x3")):
            model.apply_nodal_bc(NodeSet([node_id]), ax, "displacement",
                                 sequence=sequence,
                                 scales={node_id: u[i, iax]},
                                 **kwargs)
