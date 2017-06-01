import numpy as np

from .core import Sequence

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
    seen = {} # node ids indexed by point coords
    nodes = []
    # `seen` and `nodes` could be combined with an ordered set (needs
    # implementation) to save on memory.
    p_elems = elems # elements as lists of points coords
    c_elems = [] # elements as node indices
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

def apply_uniax_stretch(model, stretches, axis='x'):
    """Apply stretch with must points and a fixed width grip line.

    axis := 'x' or 'y'; the direction along which to stretch the mesh.
    Displacement is prescribed to the node(s) furthest +ve and -ve along
    the specified axis.

    """
    axis1 = axis
    if axis == 'x':
        axis2 = 'y'
        iaxis1 = 0
        iaxis2 = 1
    elif axis == 'y':
        axis2 = 'x'
        iaxis1 = 1
        iaxis2 = 0
    else:
        raise ValueError("`axis` must be 'x' or 'y'; {} was provided".format(axis))
    maxima = np.max(model.mesh.nodes, 0)
    minima = np.min(model.mesh.nodes, 0)
    length = maxima[iaxis1] - minima[iaxis1]
    # Moving (gripped) nodes
    tol = np.spacing(0.5 * length)
    gripped_nodes = [i for i, x in enumerate(model.mesh.nodes)
                     if x[iaxis1] == minima[iaxis1] or x[iaxis1] == maxima[iaxis1]]
    u1 = [(stretches[-1] - 1) * model.mesh.nodes[i][iaxis1] for i in gripped_nodes]
    # ^ gripped node displacements at final timepoint
    seq_bc = Sequence([(0, 0), (1, 1)])
    model.apply_nodal_displacement(gripped_nodes, values=u1, axis=axis1, sequence=seq_bc)
    # Fixed nodes
    model.fixed_nodes[axis2].update(gripped_nodes)
    gripped_nodes_back = [i for i in gripped_nodes
                          if model.mesh.nodes[i][2] == minima[2]]
    model.fixed_nodes['z'].update(gripped_nodes_back)
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
