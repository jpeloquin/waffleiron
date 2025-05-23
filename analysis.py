import numpy as np

from waffleiron.core import NodeSet
from waffleiron.model import Model
from waffleiron.exceptions import InvalidSelectionError


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
        f = e.f(r, soln.data["displacement"])
        return fn(f, e)


def apply_q_2d(mesh, crack_tip, q=[1, 0, 0], n=3, qtype="plateau"):
    """Define q for for the J integral.

    crack_tip := list of node ids comprising the crack line

    Notes:

    Zero crack face tractions are assumed.

    In 3D, classification of the crack faces works only for hexahedral
    elements.

    """
    q = np.array(q)

    active_nodes = set(crack_tip)
    all_nodes = set(crack_tip)
    inner_nodes = set()
    elements = set()
    for ring in range(n):
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

    q_nodes = [None] * len(mesh.nodes)
    if qtype == "plateau":
        for i in inner_nodes:
            q_nodes[i] = 1.0 * q
        for i in outer_nodes:
            q_nodes[i] = 0.0 * q
    else:
        raise NotImplemented(
            "{}-type q functions are not " "implemented yet.".format(qtype)
        )

    # Apply q to all elements
    for e in mesh.elements:
        q_local = [q_nodes[i] for i in e.ids]
        if all([a is not None for a in q_local]):
            e.properties["q"] = np.array(q_local)
        else:
            # Can't make an array if some nodes' values are None and
            # others' are 3-element arrays.  Numpy ≥ 1.19 will issue a
            # warning.
            e.properties["q"] = q_local

    return elements


def apply_q_3d(domain, crack_faces, tip_nodes, q=[1, 0, 0], qtype="plateau"):
    """Define q for for the J integral.

    domain := A list or set of elements comprising the domain.

    q := The q vector to be applied to the domain's internal nodes.

    crack_tip := list of node ids comprising the crack tip line

    Notes:

    Zero crack face tractions are assumed.

    This function is only applicable to hexahedral meshes for which
    the crack face surfaces together form a quad mesh that is (1) a
    manifold and (2) regular.

    The hexahedral mesh must also be regular; i.e. all interior nodes
    are connected to 6 edges (8 elements).

    """
    if not domain:
        raise InvalidSelectionError("Integration domain is an empty set.")

    # Define q vector
    q = np.array(q)

    # Initialize node selections
    all_nodes = set(i for e in domain for i in e.ids)
    tip_nodes = set(tip_nodes) & all_nodes

    # Find boundary nodes
    refcount = {}  # how many elements each node is connected to
    for e in domain:
        for i in e.ids:
            refcount[i] = refcount.setdefault(i, 0) + 1
    boundary_nodes = set(k for k, v in refcount.items() if v < 8) | tip_nodes
    interior_nodes = all_nodes - boundary_nodes
    if not interior_nodes:
        raise InvalidSelectionError("All nodes in `elements` are boundary nodes.")

    # Find the moving nodes on the crack line.  Find adjacent faces to
    # the crack tip line (sharing at least 2 nodes), then grow the
    # selection by the same adjacency rules until reaching the edge of
    # the domain.
    surface_faces = set(
        f for e in domain for f in e.faces() if set(f) <= boundary_nodes
    )
    non_crack_faces = surface_faces - crack_faces
    crack_surface_nodes = set([i for f in crack_faces for i in f])
    non_crack_face_nodes = set([i for f in non_crack_faces for i in f])
    crack_nodes_q0 = non_crack_face_nodes & crack_surface_nodes

    crack_nodes_q1 = crack_surface_nodes - crack_nodes_q0

    # Apply
    if qtype == "plateau":
        q0_nodes = boundary_nodes - crack_nodes_q1
        q1_nodes = all_nodes - q0_nodes
        for e in domain:
            qprop = np.array([[np.nan, np.nan, np.nan]] * len(e.ids))
            for i, node_id in enumerate(e.ids):
                if node_id in q0_nodes:
                    qprop[i] = 0.0 * q
                elif node_id in q1_nodes:
                    qprop[i] = 1.0 * q
                else:
                    msg = "Element {} node {} does not have q defined."
                    raise Exception(msg.format(e, i))
            e.apply_property("q", qprop)
    else:
        raise NotImplemented(
            "{}-type q functions are not " "implemented.".format(qtype)
        )

    return domain


def jintegral(domain, infinitessimal=False):
    """Calculate J integral.

    Parameters
    ----------
    domain : list of element objects
       The q function should be pre-applied as nodal properties.

    """

    def integrand(e, r, debug=False):
        """Integrate over a single element.

        Parameters
        ----------
        e : Element object

        """
        F = e.f(r)
        s = e.material.sstress(F)  # 2nd Piola-Kirchoff stress
        dudx = e.dinterp(r, prop="displacement")
        w = e.material.w(F)  # strain energy
        q = e.interp(r, prop="q")
        dqdx = e.dinterp(r, prop="q")

        # compute ∂s/∂x at this point
        dsdx = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                dsdx[i, j, :] = e.dinterp(r, prop=e.properties["S"][:, i, j])

        # compute ∂²u/∂x²
        d2udx2 = e.ddinterp(r, prop="displacement")

        # compute ∂ψ/∂x
        dwdx = e.dinterp(r, prop="w")

        kd = np.eye(3)  # kronecker delta

        # The usual part.
        work = sum(
            s[i][j] * dudx[j, k] * dqdx[k][i]
            for i in range(3)
            for j in range(3)
            for k in range(3)
        )
        pe = sum(-w * dqdx[ik][ik] for ik in range(3))
        igrand1 = work + pe

        # The non-infinitessimal strain part

        # The Hakimelahi_Ghazavi_2008 version
        igrand2 = sum(
            q[k]
            * (
                dsdx[i, 2, 2] * dudx[i, k]
                + s[i, 2] * d2udx2[i, k, 2]
                - dwdx[2] * kd[i, 2]
            )
            for i in range(3)
            for k in range(3)
        )

        # The Anderson version, corrected to use the divergence
        # operator correctly
        igrand2_1 = sum(
            q[k] * (dsdx[i, j, j] * dudx[j, k])
            for i in range(3)
            for j in range(3)
            for k in range(3)
        )

        igrand2_2 = sum(
            q[k] * (s[i, j] * d2udx2[j, k, j])
            for i in range(3)
            for j in range(3)
            for k in range(3)
        )

        igrand2_3 = sum(q[i] * (-dwdx[i]) for i in range(3))

        igrand2 = igrand2_1 + igrand2_2 + igrand2_3

        if infinitessimal:
            return igrand1
        else:
            return igrand1 + igrand2

    # apply stress and strain energy to element nodes
    for e in domain:
        # calculate 2nd piola-kirchoff stress for each node so we can
        # take its derivative
        nodal_f = [e.f(r) for r in e.vloc]
        nodal_s = [e.material.sstress(f) for f in nodal_f]
        nodal_s = np.array(nodal_s)
        e.properties["S"] = nodal_s
        # same for strain energy
        nodal_w = np.array([e.material.w(f) for f in nodal_f])
        e.properties["w"] = nodal_w

    for e in domain:
        dsdx = e.dinterp((0, 0, 0), prop="S")
        # print dsdx[0, 0, 0] + dsdx[0,1,1] + dsdx[0,2,2]
        # print sum([integrand(e, r, debug=True)[1] for r in e.gloc])

    # compute j
    j = 0.0
    for e in domain:
        j += e.integrate(integrand)

    return j


def distance(model: Model, a: NodeSet, b: NodeSet):
    """Return distance between a and b as a time series.

    Inputs:
    -------

    model := waffleiron Model object with a solution.

    a := First distance marker

    b := Second distance marker

    Distance is calculated from the center of one marker to the center
    of the other marker.  The order of markers does not matter.  For
    collective markers (e.g., node sets) the order of entities within
    each marker does not matter.

    For a node set marker, the center of the marker is the arithmetic
    mean of the positions of all nodes in the node set.

    Returns:
    --------

    scalar time series of distance values

    """
    d = np.empty(len(model.solution.step_times))
    d[:] = np.nan  # make any indexing errors more obvious
    for i in range(len(model.solution.step_times)):
        δ = np.array(model.solution.step_data(i)[("displacement", "node")])
        δ_a = np.mean([δ[j] for j in a], axis=0)
        δ_b = np.mean([δ[j] for j in b], axis=0)
        x0_a = np.mean([model.mesh.nodes[j] for j in a], axis=0)
        x0_b = np.mean([model.mesh.nodes[j] for j in b], axis=0)
        d[i] = np.linalg.norm((x0_a + δ_a) - (x0_b + δ_b))
    return d


def strain_gauge(model: Model, a: NodeSet, b: NodeSet):
    """Return strain between a and b as a time series.

    Inputs:
    -------

    model := waffleiron Model object with a solution.

    a := First strain gauge marker / anchor.

    b := Second strain gauge marker / anchor.

    Strain is calculated from the center of one marker to the center of
    the other marker.  The order of markers does not matter.  For
    collective markers (e.g., node sets) the order of entities within
    each marker does not matter.

    For a node set marker, the center of the marker is the arithmetic
    mean of the positions of all nodes in the node set.

    Returns:
    --------

    scalar time series of engineering strain; for all times t,
    (displacement between a and b at t) / (distance between a and b at t
    = 0 s).

    """
    λ = np.empty(len(model.solution.step_times))
    λ[:] = np.nan  # make any indexing errors more obvious
    for i in range(len(model.solution.step_times)):
        δ = np.array(model.solution.step_data(i)[("displacement", "node")])
        δ_a = np.mean([δ[j] for j in a], axis=0)
        δ_b = np.mean([δ[j] for j in b], axis=0)
        x0_a = np.mean([model.mesh.nodes[j] for j in a], axis=0)
        x0_b = np.mean([model.mesh.nodes[j] for j in b], axis=0)
        λ[i] = np.linalg.norm((x0_b + δ_b) - (x0_a + δ_a)) / np.linalg.norm(x0_b - x0_a)
    return λ
