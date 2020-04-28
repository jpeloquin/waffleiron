from warnings import warn
import numpy as np

from .core import Sequence, _DEFAULT_TOL


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


def merge_node_ids(mesh, nodes):
    """Merge node ids in nodes in `mesh`

    Note that this function does not clean up any orphaned nodes.

    """
    # Adjust references in elements
    nodes = np.array(nodes)
    for e in mesh.elements:
        for j, k in enumerate(e.ids):
            if k in nodes:
                e.ids[j] = nodes[0]


def merge_node_positions(mesh, candidates=None, tol=_DEFAULT_TOL):
    """Merge overlapping nodes.

    candidates := list-like of integer node ids.

    """
    if candidates is None:
        candidates = range(len(mesh.nodes))
    for i in candidates:
        # Find nodes in mesh closest to i's position
        x_i = np.array(mesh.nodes[i])
        imatch = set(mesh.find_nearest_nodes(*x_i)) - set([i])
        to_merge = [j for j in imatch if np.linalg.norm(x_i - np.array(mesh.nodes[j])) < tol]
        merge_node_ids(mesh, [i] + to_merge)
