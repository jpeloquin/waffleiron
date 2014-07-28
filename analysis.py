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

from febtools.selection import adj_faces

def apply_q(mesh, crack_line, n=3, qtype='plateau', dimension='2d'):
    """Define q for for the J integral.

    crack_line := list of node ids comprising the crack line

    Notes:

    Zero crack face tractions are assumed.

    In 3D, classification of the crack faces works only for hexahedral
    elements.

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
    outer_nodes = active_nodes

    # The final active node set forms the exterior ring of the domain,
    # which must have q = 0.  In the 3D case, the faces on either end
    # of the crack line must also have q = 0.

    if dimension == '3d':
        # Find surface faces
        surface_faces = feb.selection.surface_faces(mesh)
        # Find crack faces
        crack_faces = set()
        adv_front = set(crack_line)
        processed_nodes = set()
        for j in xrange(n):
            candidates = (f for i in adv_front
                          for f in mesh.faces_with_node[i])
            on_surface = (f for f in candidates
                          if len(adj_faces(mesh, f, mode='face')) == 0)
            new = [f for f in on_surface
                   if len(set(f.ids) & adv_front) > 1]
            crack_faces.update(new)
            processed_nodes.update(adv_front)
            adv_front = set.difference(set([i for f in crack_faces
                                            for i in f.ids]),
                                       processed_nodes)

    q = [None] * len(mesh.nodes)
    if qtype == 'plateau':
        for i in inner_nodes:
            q[i] = 1.0
        for i in outer_nodes:
            q[i] = 0.0
    else:
        raise NotImplemented('{}-type q functions are not '
                             'implemented yet.'.format(qtype))
    # Set ends of 3d cyclinder to q = 0
    if dimension == '3d':
        candidates = set(i for f in surface_faces - crack_faces
                         for i in f.ids)
        inner_cap_nodes = candidates & inner_nodes
        for i in inner_cap_nodes:
            q[i] = 0.0

    # Apply q to all elements
    for e in mesh.elements:
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
