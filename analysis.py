import numpy as np
import febtools as feb
from collections import defaultdict

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

def apply_q_2d(mesh, crack_tip, n=3, qtype='plateau'):
    """Define q for for the J integral.

    crack_tip := list of node ids comprising the crack line

    Notes:

    Zero crack face tractions are assumed.

    In 3D, classification of the crack faces works only for hexahedral
    elements.

    """
    active_nodes = set(crack_tip)
    all_nodes = set(crack_tip)
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

    q = [None] * len(mesh.nodes)
    if qtype == 'plateau':
        for i in inner_nodes:
            q[i] = 1.0
        for i in outer_nodes:
            q[i] = 0.0
    else:
        raise NotImplemented('{}-type q functions are not '
                             'implemented yet.'.format(qtype))

    # Apply q to all elements
    for e in mesh.elements:
        e.properties['q'] = np.array([q[i] for i in e.ids])

    return elements

from febtools.selection import expand_element_set

def apply_q_3d(elements, crack_tip, n=3, qtype='plateau'):
    """Define q for for the J integral.

    crack_tip := list of node ids comprising the crack tip line

    Notes:

    Zero crack face tractions are assumed.

    This function is only applicable to hexahedral meshes for which
    the crack face surfaces together form a quad mesh that is (1) a
    manifold and (2) regular.

    The hexahedral mesh must also be regular; i.e. all interior nodes
    are connected to 6 edges (8 elements).

    """
    all_nodes = set(i for e in elements for i in e.ids)
    crack_tip = set(crack_tip) & all_nodes

    # Grow a sub-selection of elements from the crack tip
    tip_elements = [e for e in elements if set(e.ids) & crack_tip]
    elements = expand_element_set(elements, tip_elements, n - 1)

    # Find boundary nodes
    refcount = {} # how many elements each node is connected to
    for e in elements:
        for i in e.ids:
            refcount[i] = refcount.setdefault(i, 0) + 1
    boundary_nodes = (set(k for k, v in refcount.iteritems()
                         if v < 8)
                      | crack_tip)
    interior_nodes = all_nodes - boundary_nodes
    if not interior_nodes:
        raise ValueError("All nodes in `elements` are boundary nodes.")

    # Find the moving nodes on the crack line
    surface_faces = set(frozenset(f) for e in elements
                        for f in e.faces()
                        if set(f) <= boundary_nodes)
    crack_faces = set()
    active_nodes = set(crack_tip)
    # ^ faces connected to these nodes will be added to crack_faces
    candidates = surface_faces
    for j in xrange(n):
        new = [f for f in candidates
               if len(f & active_nodes) > 1
               and f not in crack_faces]
        crack_faces.update(new)
        # move the active front
        active_nodes = set(i for f in new for i in f) - active_nodes
    crack_nodes_q0 = active_nodes

    # Find crack nodes on interior of crack face surface mesh
    refcount = {}
    for f in crack_faces:
        for i in f:
            refcount[i] = refcount.setdefault(i, 0) + 1
    crack_surface_nodes_interior = set(k for k, v
                                       in refcount.iteritems()
                                       if v == 4)
    crack_nodes_q1 = crack_surface_nodes_interior - crack_nodes_q0
    # Apply
    if qtype == 'plateau':
        q0_nodes = boundary_nodes - crack_nodes_q1
        q1_nodes = all_nodes - q0_nodes
        for e in elements:
            qprop = np.array([np.nan] * len(e.ids))
            for i, node_id in enumerate(e.ids):
                if node_id in q0_nodes:
                    qprop[i] = 0.0
                elif node_id in q1_nodes:
                    qprop[i]= 1.0
                else:
                    raise Exception("Element {} node {} does not have q defined.".format(e, i))
            e.apply_property('q', qprop)
    else:
        raise NotImplemented('{}-type q functions are not '
                             'implemented.'.format(qtype))

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
        return -w * dqdx[0] + sum(p[i][j] * dudx[j,0] * dqdx[i]
                                 for i in xrange(len(r))
                                 for j in xrange(len(r)))
    j = 0
    for e in elements:
        j += e.integrate(integrand)
    return j
