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
