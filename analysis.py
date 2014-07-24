import numpy as np
import febtools as feb

def select_elems_around_node(mesh, i, n=3):
    """Select elements centered on node i.

    Parameters
    ----------

    mesh : febtools.mesh.Mesh object

    i : integer
        The index of the central node.

    r : integer, optional
        The number of concentric rings of elements to select.

    """
    nodelist = set([i])
    elements = set([])
    for n in xrange(n):
        for i in nodelist:
            elements = elements | set(mesh.elem_with_node[i])
        nodelist = set(i for e in elements
                       for i in e.ids)
        # ^ This needlessly re-adds elements already in the domain;
        # there's probably a better way; see undirected graph search.
        # Doing this efficiently requires transforming the mesh into a
        # graph data structure.
    return elements

def eval_fn_x(soln, fn, pt):
    """Evaluate a function at a specific point.

    fn := A function that takes an F tensor and an Element object.

    soln := A MeshSolution object.

    pt : (x, y, z)
        The coordinates of the point at which to evaluate `fn`.  The z
        value may be omitted; if so, it is assumed to be 0.

    Returns
    -------
    The return value of fn evaluated at pt or, if no element contains
    pt, None.

    """
    e = soln.element_containing_point(pt)
    if e is None:
        return None
    else:
        r = e.to_natural(pt)
        f = e.f(r, soln.data['displacement'])
        return fn(f, e)

def apply_q(mesh, crack_line, n=3, qtype='plateau'):
    """Define q for for the J integral.

    crack_line := list of node ids comprising the crack line

    Notes:

    Zero crack face tractions are assumed.

    """
    active_nodes = set(crack_line)
    all_nodes = set(crack_line)
    inner_nodes = set()
    elements = set()
    for ring in xrange(n):
        # The current nodeset becomes interior nodes
        inner_nodes.update(all_nodes)
        # Add connected elements
        for i in active_nodes:
            elems = mesh.elem_with_node[i]
            elements.update(elems)
            for e in elems:
                all_nodes.update(e.ids)
        # Update list of active nodes to the set of exterior nodes
        # that were just added to the full nodeset.
        active_nodes = all_nodes - inner_nodes
    elements = set(elements)

    # The final active node set forms the exterior ring of the domain,
    # which must have q = 0.  In the 3D case, the faces on either end
    # of the crack line must also have q = 0.
    outer_nodes = active_nodes
    q = [None] * len(mesh.nodes)
    if qtype == 'plateau':
        for i in inner_nodes:
            q[i] = 1.0
        for i in outer_nodes:
            q[i] = 0.0
    else:
        raise NotImplemented('{}-type q functions are not '
                             'implemented yet.'.format(qtype))

    # Apply q to the elements
    for e in elements:
        e.properties['q'] = np.array([q[i] for i in e.ids])

    return elements

def jintegral(elements):
    """Calculate J integral.

    Parameters
    ----------
    elements : list of element objects
       The q function should be pre-applied as nodal properties.

    """

    def integrand(e, r):
        """Integrate over a single element.

        Parameters
        ----------
        e : Element object

        """
        F = e.f(r)
        p = e.material.pstress(F) # 1st Piola-Kirchoff stress
        dudx = e.dinterp(r, prop='displacement')
        dudx1 = dudx[:,0]
        w = e.material.w(F) # strain energy
        dqdx = e.dinterp(r, prop='q') # 1 x 2 or 1 x 3
        return -w * dqdx[0] + sum(p[i][j] * dudx[i,0] * dqdx[j]
                                 for i in xrange(len(r))
                                 for j in xrange(len(r)))
    j = 0
    for e in elements:
        j += e.integrate(integrand)
    return j
