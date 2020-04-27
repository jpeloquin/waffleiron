"""Module for adding initial and boundary conditions to models."""
# Third party modules
import numpy as np
# Same-package modules
from .core import Sequence, NodeSet


def densify(curve, n):
    dense_curve = []
    for i, (x0, y0) in enumerate(curve[:-1]):
        x1 = curve[i + 1][0]
        y1 = curve[i + 1][1]
        x = np.linspace(x0, x1, n + 1)
        y = np.interp(x, [x0, x1], [y0, y1])
        dense_curve += [(a, b) for a, b in zip(x[:-1], y[:-1])]
    # Add last point
    dense_curve.append((curve[-1]))
    return dense_curve


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
