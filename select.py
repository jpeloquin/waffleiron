"""Functions for conveniently selecting nodes and elements.

"""
from math import degrees, radians
import operator
from copy import copy

import numpy as np

import waffleiron as wfl
from .core import _canonical_face
from waffleiron.geometry import inter_face_angle, face_normal

default_tol = 10 * np.finfo(float).eps


def find_closest_timestep(target, times, steps, rtol=0.01, atol="auto"):
    """Return step index closest to given time."""
    times = np.array(times)
    if atol == "auto":
        atol = max(
            abs(np.nextafter(target, 0) - target),
            abs(np.nextafter(target, target**2) - target),
        )
    if len(steps) != len(times):
        raise ValueError(
            "len(steps) ≠ len(times).  All steps must have a corresponding time value and vice versa."
        )
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
        t_interval = times[idx + 1] - times[idx]
        t_relerror = t_error / t_interval
    elif step == steps[-1] and t_error > 0:
        # The selection specifies a time point after the
        # last time step.  It might only be a little
        # after, within acceptable tolerance when
        # working with floating point values, so we do
        # not raise an error until further checks.
        t_interval = times[idx] - times[idx - 1]
        t_relerror = t_error / t_interval
    else:
        t_interval = abs(times[idx] - times[idx + int(np.sign(t_error))])
        t_relerror = t_error / t_interval
    # Check error tolerance
    if abs(t_error) > atol:
        raise ValueError(
            f"Time step selection absolute error > atol; target time — selected time = {t_error}; atol = {atol}."
        )
    if abs(t_relerror) > abs(rtol):
        raise ValueError(
            f"Time step selection relative error > rtol; target time — selected time = {t_error}; step interval = {t_interval}; relative error = {t_relerror}; rtol = {rtol}."
        )
    return step


class ElementSelection:
    def __init__(self, mesh, elements=None):
        """Create a selection of elements.

        mesh := the `mesh` to which the elements belong

        elements := A `set` of elements. Optional. If unset, all
        elements in the mesh are selected.

        This class permits chaining of selection functions, many of
        which need a reference to the parent mesh in addition to the
        element selection.

        """
        self.mesh = mesh
        if elements is None:
            self.elements = set(mesh.elements)
        else:
            self.elements = elements


def elements_containing_point(point, elements, bb=None, tol=default_tol):
    """Return element(s) containing a point

    point := (x, y, z)

    elements : list of elements
        The ordering of `elements` must match the ordering of
        `bb`.

    bb : array (optional)
        An array with shape (n, 3, 2), where n is the length of
        `elements`.  The second dimension is x, y, z.  The third
        dimension indexes the minimum and maximum bounding box
        coordinates, in that order.

    Returns [] if no elements contain point.

    """
    p = np.array(point)
    if not bb:
        bb = wfl.core._e_bb(elements)
    in_bb = np.all(
        np.logical_and((point + tol) >= bb[0], (point - tol) <= bb[1]), axis=1
    )
    inds = np.nonzero(in_bb)[0]
    candidates = [elements[i] for i in inds]
    elements = [e for e in candidates if wfl.geometry.point_in_element(e, p, dtol=tol)]
    return elements


def select_elems_around_node(mesh, i, n=1):
    """Select elements centered on node i.

    Parameters
    ----------

    mesh : waffleiron.mesh.Mesh object

    i : integer
        The index of the central node.

    n : integer, optional
        The number of concentric rings of elements to select.

    """
    nodelist = set([i])
    elements = set([])
    for n in range(n):
        for i in nodelist:
            elements = elements | set(mesh.elem_with_node[i])
        nodelist = set(i for e in elements for i in e.ids)
        # ^ This needlessly re-adds elements already in the domain;
        # there's probably a better way; see undirected graph search.
        # Doing this efficiently requires transforming the mesh into a
        # graph data structure.
    return elements


def corner_nodes(mesh):
    """Return ids of corner nodes."""
    ids = [i for i in range(len(mesh.nodes)) if len(mesh.elem_with_node[i]) == 1]
    return ids


def surface_faces(mesh):
    """Return surface faces."""
    surf_faces = set()
    for body in mesh.bodies:
        # Pick a node to start. Nodes with minimum/maximum coordinate
        # values must be surfaces nodes.
        nids, xnodes = body.nodes()
        i, j, k = np.argmin(xnodes, axis=0)
        i = nids[i]  # translate index to node index in mesh
        # Advance across the surface with a "front" of active nodes
        adv_front = set([i])
        processed_nodes = set()
        while adv_front:
            candidate_faces = (
                f
                for i in adv_front
                for e in mesh.elem_with_node[i]
                for f in e.faces()
                if i in f
            )
            on_surface = [
                f for f in candidate_faces if len(adj_faces(f, mesh, mode="face")) == 0
            ]
            surf_faces.update(on_surface)
            processed_nodes.update(adv_front)
            adv_front = set.difference(
                set([i for f in surf_faces for i in f]), processed_nodes
            )
    return surf_faces


def bisect(elements, p, v):
    """Return elements on one side of of a plane.

    p := A point (x, y, z) on the plane.

    v := A vector (vx, vy, vz) normal to the plane.

    The elements returned are those either intersected by the cut
    plane or on the side of the plane towards which `v` points.

    """
    # sanitize inputs
    v = np.array(v)
    p = np.array(p)

    # find distance from plane

    def on_pside(e, p=p, v=v):
        """Returns true if element touches (or intersects) plane."""
        dpv = np.dot(p, v)
        d = np.dot(e.nodes, v)
        return any(d >= dpv)

    eset = [e for e in elements if on_pside(e)]
    return set(eset)


def element_slice(elements, v, extent=default_tol, axis=(0, 0, 1)):
    """Return a slice of elements.

    v := The distance along `axis` at which the slice plane is
    located.

    axis := A normal vector to the slice plane.  Must coincide with
    the cartesian coordinate system; i.e. two values must be 0 and
    the third 1.

    Any element within +/- `extent` of `v` along `axis` is included in
    the slice.  The default extent is the floating point precision.
    Therefore, if the selection plane coincides with a node, the
    elements on both sides of the plane will be included.

    """
    # sanitize inputs
    axis = np.abs(np.array(axis), dtype=float)
    # figure out which axis we're using
    idx = np.array(axis).nonzero()[0]
    assert len(idx) == 1
    iax = idx[0]
    # Select all above and intersecting lower bound
    v1 = v - extent
    elements = bisect(elements, p=v1 * axis, v=axis)
    # Select all below and intersecting upper bound
    v2 = v + extent
    elements = bisect(elements, p=v2 * axis, v=-axis)
    return set(elements)


def e_grow(selection, candidates, n):
    """Grow element selection by n elements.

    seed := The growing selection.

    candidates := The set of elements that are candidates for growing
    the seed.

    """
    seed = set(selection)
    inactive_nodes = set([])
    active_nodes = set([i for e in seed for i in e.ids])
    candidates = set(candidates) - seed
    n_iter = 0  # number of completed iterations
    while active_nodes and n_iter != n:
        # Find adjacent elements
        adjacent = set(e for e in candidates if any(i in active_nodes for i in e.ids))
        # Grow the seed
        seed = seed | adjacent
        # Inactivate former boundary nodes
        inactive_nodes.update(active_nodes)
        # Get new boundary (active) nodes
        nodes = set(i for e in adjacent for i in e.ids)
        active_nodes = nodes - inactive_nodes
        # Update list of candidates
        candidates = candidates - adjacent
        n_iter += 1
    return seed


def faces_by_normal(elements, normal, delta=default_tol):
    """Return all faces with target normal."""
    target = 1.0 - delta
    faces = []
    for e in elements:
        for n, f in zip(e.face_normals(), e.faces()):
            n = n / np.linalg.norm(n)  # make unit vector
            if np.dot(n, normal) > target:
                faces.append(f)
    return faces


def f_grow_to_edge(faces, mesh, delta=30):
    """Select all adjacent faces with similar normals.

    The selection is restricted to faces on the surface of the mesh.

    faces := sequence of face tuples

    delta := the angle change that defines an edge between adjacent
    faces (degrees)

    """
    if len(faces) == 0:
        raise ValueError("Empty set of faces provided.")
    # work in radians internally
    delta_deg = delta
    delta_rad = radians(delta_deg)

    f_surface = surface_faces(mesh)
    faces = [_canonical_face(f) for f in faces]
    # make sure given faces are on surface
    for f in faces:
        assert f in f_surface

    f_subsurface = set(faces)
    new_faces = set(faces)
    while new_faces:
        seed_faces = copy(new_faces)
        new_faces = set()
        for seed_face in seed_faces:
            af = adj_faces(seed_face, mesh, mode="edge")
            af = [f for f in af if (inter_face_angle(f, seed_face, mesh) < delta_rad)]
            new_faces.update(af)
        new_faces.difference_update(f_subsurface)
        f_subsurface.update(new_faces)
    return f_subsurface


def adj_faces(face, mesh, mode="all"):
    """Return faces connected to a face.

    face : tuple of integers
        The face for which adjacent faces will be identified, defined
        by the ids of its nodes.

    mode : {'all', 'edge', 'face'}
        The type of adjacency desired. Specifying 'edge' returns only
        faces which share an edge with the input face. Specifying
        'face' returns only faces that share every node with the input
        face.

    """

    def overlap(a, b):
        """Return the number of shared nodes between two faces."""
        o = len(set(a) & set(b))
        return o

    face = tuple(face)
    face = _canonical_face(face)

    # faces sharing at least one node, sans the queried face
    nc_faces = set([b for i in face for b in mesh.faces_with_node(i) if b != face])

    if mode == "all":
        return nc_faces
    elif mode == "edge":
        return [b for b in nc_faces if overlap(face, b) == 2]
    elif mode == "face":
        n = len(face)
        return [b for b in nc_faces if overlap(face, b) == n]
