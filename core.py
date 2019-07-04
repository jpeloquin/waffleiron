from copy import copy, deepcopy
from collections import defaultdict
from math import inf
from operator import itemgetter
import sys
from warnings import warn

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from copy import deepcopy

import febtools as feb

# Set tolerances
_default_tol = 10*np.finfo(float).eps

# Increase recursion limit for kdtree
sys.setrecursionlimit(10000)


def _validate_axis(axis, rigid=False):
    allowed_axes = ['x1', 'x2', 'x3', 'fluid', 'temperature', 'charge']
    if rigid:
       allowed_axes += ['α1', 'α2', 'α3']
    if not axis in allowed_axes:
        msg = f"{axis} is not a supported axis type.  The supported axis types are " +\
            ", ".join(allowed_axes) + "."
        raise ValueError(msg)


class Body:
    """A geometric body of elements.

    Explicit rigid bodies should be constructed using this type.

    """
    def __init__(self, elements):
        self.elements = elements
        # self.master_inode = elements[0].ids[0]

    def nodes(self):
        """Return (node_ids, xnodes) for this body."""
        nids = set()
        for e in self.elements:
            nids.update(e.ids)
        nids = np.array([i for i in nids])
        parent_mesh = next(iter(self.elements)).mesh
        # ^ assumption: all elements are from same mesh
        xnodes = np.array([parent_mesh.nodes[i] for i in nids])
        return nids, xnodes


class ImplicitBody:
    """A geometric body defined by its interface with a mesh."""
    def __init__(self, mesh, interface, material=None):
        """Constructor for ImplicitBody object.

        interface := a collection of node ids

        At some some point in the future, support may be added to define
        an interface using a list of faces.

        """
        self.mesh = mesh
        self.interface = set(interface)
        self.material = material


class ContactConstraint:
    """A constraint defining contact between two surfaces."""
    def __init__(self, leader, follower, algorithm=None,
                 auto_penalty=True,
                 auto_penalty_scale=1,
                 penalty_factor=None,
                 augmented_lagrange=False,
                 passes=1,
                 tension=False,
                 **kwargs):
        self.leader = leader
        self.follower = follower
        self.algorithm = algorithm
        self.tension = tension
        self.augmented_lagrange = augmented_lagrange
        self.passes = passes
        if auto_penalty:
            self.penalty = {'type': 'auto',
                            'factor': auto_penalty_scale}
            if penalty_factor is not None:
                warn("A value for `penalty_factor` was provided, but automatic penalty factor calculation was also requested.  The `penalty_factor` value will not be used.  You may want to set `auto_penalty_scale` instead.")
        else:
            assert penalty_factor is not None
            self.penalty = {'type': 'manual',
                            'factor': penalty_factor}
        # TODO: Warn if two passes are specified and at least one of the
        # surfaces belongs to a rigid body.


class Model:
    """An FE model: geometry, boundary conditions, solution.

    """
    def __init__(self, mesh):
        if type(mesh) is not feb.core.Mesh:
            raise TypeError("{} is not of type"
                            "febtools.core.Model".format(mesh))

        self.mesh = mesh  # The model geometry
        self.solution = None  # The solution for the model

        # If a sequence is assigned to dtmax in this default dictionary,
        # it will be copied when `default_control` is used to initialize a
        # control step.  This may not be desirable.
        self.default_control = feb.control.default_control_section()

        fixed_template = {'x1': set(),
                          'x2': set(),
                          'x3': set(),
                          'pressure': set(),
                          'concentration': set()}
        self.fixed = {'node': deepcopy(fixed_template),
                      'body': deepcopy(fixed_template)}
        self.fixed['body'].update({'α1': set(),
                                   'α2': set(),
                                   'α3': set()})
        # Note: for multiphasic problems, concentration is a list of
        # sets
        #
        # TODO: Make the specification of fixed-for-all-time BCs have
        # the same format as for BCs in steps.

        # Contact
        self.constraints = []

        # Initialize dictionaries to hold named named entities
        self.named = {"materials": NameRegistry(),
                      "node sets": NameRegistry(),
                      "facet sets": NameRegistry(),
                      "element sets": NameRegistry(),
                      "sequences": NameRegistry()}

        # initial nodal values
        self.initial_values = {'velocity': [],
                               'fluid_pressure': [],
                               'concentration': []}
        # Note: for multiphasic problems, concentration is a list of
        # lists.
        self.steps = []

    def add_contact(self, constraint):
        self.constraints.append(constraint)

    def add_step(self, module='solid', control=None):
        """Add a step with default control values and no BCs.

        """
        if control is None:
            control = feb.control.default_control_section()
        step = {'module': module,
                'control': control,
                'bc': {'node': {},
                       'body': {}}}
        self.steps.append(step)


    def apply_nodal_bc(self, node_ids, axis, variable, sequence,
                       scales=None, step_id=-1):
        """Apply a boundary condition to a set of nodes.

        The boundary condition is applied to the selected solution step,
        which by default is the last step.  (Note that if you export a
        model to FEBio, it will automatically leak the boundary
        condition to all subsequent steps.)

        axis := 'x1', 'x2', 'x3', 'fluid', 'temperature', or 'charge'.

        variable := 'displacement', 'force', 'pressure', or 'flow'

        sequence := conditions.Sequence object or 'fixed'

        """
        _validate_axis(axis)
        if sequence == 'fixed':
            scales = [None]*len(node_ids)
        elif scales is None:  # variable BC
            scales = [1]*len(node_ids)
        for i, scale in zip(node_ids, scales):
            bc_node = self.steps[step_id]['bc']['node'].setdefault(i, {})
            bc_node[axis] = {'variable': variable,
                             'sequence': sequence,
                             'scale': scale}


    def apply_body_bc(self, body, axis, variable, sequence, scale=1,
                      step_id=-1):
        """Apply a variable displacement boundary condition to a body.

        The boundary condition is applied to the selected solution step,
        which by default is the last step.  (Note that if you export a
        model to FEBio, it will automatically leak the boundary
        condition to all subsequent steps.)

        axis := 'x1', 'x2', 'x3', 'fluid', 'temperature', or 'charge'.

        sequence := conditions.Sequence object or 'fixed'

        body := Body or ImplicitBody object

        """
        _validate_axis(axis)
        if sequence == 'fixed':
            scale = None
        bc = self.steps[step_id]['bc']['body'].setdefault(body, {})
        bc[axis] = {'variable': variable,
                    'sequence': sequence,
                    'scale': scale}


    def apply_solution(self, solution, t=None, step=None):
        """Attach a solution to the model.

        By default, the last timepoint in the solution is applied.

        """
        self.solution = solution
        # apply node data
        if t is None and step is None:  # use last timestep
            data = solution.step_data(-1)
        elif t is not None and step is None:
            data = solution.step_data(time=t)
        elif t is None and step is not None:
            data = solution.step_data(step=step)
        else:
            raise ValueError("Provide either `t` or `step`, not both.")
        properties = data['node variables']
        for k, v in properties.items():
            self.apply_nodal_properties(k, v)

    def apply_nodal_properties(self, key, values):
        """Apply nodal properties to each element.

        """
        for e in self.mesh.elements:
            e.properties[key] = np.array([values[i] for i in e.ids])


class Mesh:
    """Stores a mesh geometry.

    """
    # nodetree = kd-tree for quick lookup of nodes
    # elem_with_node = For each node, list the parent elements.

    def __init__(self, nodes, elements):
        """Create mesh from nodes and element objects.

        nodes := list of (x, y, z) points
        elements := list of nodal indices for each element

        """
        # Nodes
        # if nodes are 2D, add z = 0
        if len(nodes[0]) == 2:
            self.nodes = [(x, y, 0.0) for (x, y) in nodes]
        else:
            self.nodes = nodes
        # Elements
        for e in elements:
            # Make sure node ids are consistent with the nodal
            # coordinates
            if len(e.ids) == 0:
                pts_e = np.array(e.nodes)
                pts_ind = np.array([nodes[i] for i in e.ids])
                assert np.all(pts_e == pts_ind)
            # Store reference to this mesh
            e.mesh = self
        self.elements = elements
        # Bodies
        self.bodies = set()

        # Precompute derived properties
        self.prepare()

    @classmethod
    def from_ids(cls, nodes, elements, element_class):
        """Create mesh from nodes and element nodal indices.

        element_class := The Element subclass (or equivalent) to use
        for the elements.

        This function can only create meshes of homogeneous element
        type.  Meshes with heterogeneous element types may be created
        by merging two or more meshes.

        """
        element_objects = []
        for ids in elements:
            e = element_class.from_ids(ids, nodes)
            element_objects.append(e)
        mesh = cls(nodes, element_objects)
        return mesh

    def update_elements(self):
        """Update elements with current node coordinates.

        """
        for e in self.elements:
            nodes = np.array([self.nodes[i] for i in e.ids])
            e.nodes = nodes

    def prepare(self):
        """Calculate all derived properties.

        This should be called every time the mesh geometry changes.

        """
        # Create KDTree for fast node lookup
        self.nodetree = KDTree(self.nodes)

        # Create list of parent elements by node
        elem_with_node = [[] for i in range(len(self.nodes))]
        for e in self.elements:
            for i in e.ids:
                elem_with_node[i].append(e)
        self.elem_with_node = elem_with_node

        # Create list of bodies.  Each body is a set of elements that
        # are connected to each other via shared nodes.
        self.bodies = set()
        untouched_elements = set(self.elements)
        while untouched_elements:
            e = untouched_elements.pop()
            body_elements = feb.selection.e_grow([e], untouched_elements, inf)
            self.bodies.add(Body(body_elements))
            # TODO: It's a little odd to have a list of "bodies" each
            # defined as the maximal set of connected elements when
            # other bodies that are not maximal sets also exist to
            # support rigid body constraints, and these (rigid) bodies
            # aren't in self.bodies.
            untouched_elements = untouched_elements - set(body_elements)

    def faces_with_node(self, idx):
        """Return face tuples containing node index.

        The faces are tuples of integers.

        """
        candidate_elements = self.elem_with_node[idx]
        faces = [f for e in candidate_elements
                 for f in e.faces()
                 if idx in f]
        return faces

    def clean_nodes(self):
        """Remove any nodes that are not part of an element.

        """
        refcount = self.node_connectivity()
        for i in reversed(range(len(self.nodes))):
            if refcount[i] == 0:
                self.remove_node(i)

    def remove_node(self, nid_remove):
        """Remove node i from the mesh.

        The indexing of the elements into the node list is updated to
        account for the removal of the node.  An exception is thrown
        if an element refers to the removed node, since removing the
        node would then invalidate the mesh.  Remove or modify the
        element first.

        """
        def nodemap(i):
            if i < nid_remove:
                return i
            elif i > nid_remove:
                return i - 1
            else:
                return None

        def removal(e):
            return [nodemap(i) for i in e]

        elems = [removal(e.ids) for e in self.elements]
        for i, ids in enumerate(elems):
            self.elements[i].ids = ids
        self.nodes = [x for i, x in enumerate(self.nodes)
                      if i != nid_remove]

    def node_connectivity(self):
        """Count how many elements each node belongs to.

        """
        refcount = [0] * len(self.nodes)
        for e in self.elements:
            for i in e.ids:
                refcount[i] += 1
        return refcount

    def find_nearest_nodes(self, x, y, z=None):
        """Return array of node id(s) nearest (x, y, z)

        This function returns an array so that in cases where nodes are
        equidistant (and possibly superimposed), all the relevant node
        ids are returned.

        """
        if z is None:
            p = (x, y, 0)
        else:
            p = (x, y, z)
        d = np.array(self.nodes) - p
        d = np.sum(d**2., axis=1)
        idx = np.nonzero(d == np.min(d))[0]
        return idx

    def conn_elem(self, elements):
        """Find elements connected to elements.

        """
        nodes = set([i for e in elements
                     for i in e.ids])
        elements = []
        for idx in nodes:
            elements = elements + self.elem_with_node[idx]
        return set(elements)

    def merge(self, other, candidates='auto', tol=_default_tol):
        """Merge this mesh with another

        Inputs
        ------
        other : Mesh object
            The mesh to merge with this one.
        candidates : {'auto', list of int}
            If 'auto' (the default), combine all nodes in the `other`
            mesh that are distance < `tol` from a node in the current
            mesh.  This is recommended for small meshes only.  The
            simplices of `other` will be updated accordingly.  If
            `nodes` is a list, use only the node indices in the list
            as candidates for combination.  These indexes are in the
            domain of `other`.

        Returns
        -------
        Mesh object
            The merged mesh

        """
        if candidates == 'auto':
            candidates = range(len(other.nodes))
        # copy nodes so any error will not corrupt the original mesh
        nodelist = deepcopy(self.nodes)
        nodes_cd = [other.nodes[i] for i in candidates]
        dist = cdist(nodes_cd, self.nodes, 'euclidean')
        # ^ i indexes candidate list (other), j indexes nodes in self
        newind = []  # new indices for 'other' nodes after merge
        # Iterate over nodes in 'other'
        for i, p in enumerate(other.nodes):
            if i in candidates:
                # Find node in 'self' closest to p
                imatch = self.find_nearest_nodes(*p)
                for node_id in imatch:
                    # pmatch = self.nodes[node_id]
                    i_cd = candidates.index(i)
                    if dist[i_cd, node_id] < tol:
                        # Make all references in 'other' to p use pmatch
                        # instead
                        newind.append(node_id)
                    else:
                        newind.append(len(nodelist))
                        nodelist.append(p)
            else:
                # This node will not be combined; just append it
                # to the 'self' nodelist
                newind.append(len(nodelist))
                nodelist.append(p)
        # Update this mesh's node list
        self.nodes = nodelist
        # Define new elements for "other" mesh
        # new_simplices = [list(e.ids) for e in other.elements]
        for i, e in enumerate(other.elements):
            new_ids = [newind[j] for j in e.ids]
            e.ids = new_ids
            # Merge the element to the first mesh
            self.elements.append(e)
            # Update the reference to the element's parent mesh
            e.mesh = self
        self.prepare()

    def _build_node_graph(self):
        """Create a node connectivity graph for faster node lookup.

        The connectivity graph is a list of lists.
        `node_graph[i][j]` returns the jth node connected to node
        i.

        """
        for nid in self.nodes:
            pass


def _canonical_face(face):
    """Return the canonical face tuple.

    The canonical face tuple is the ordering of the face nodes such
    that the lowest node id is first.  To achieve this ordering, only
    rotational shifts are allowed.

    """
    i, inode = min(enumerate(face), key=itemgetter(1))
    face = tuple(face[i:] + face[:i])
    return face


def _e_bb(elements):
    """Create bounding box array from element list.

    """
    bb_max = np.vstack(np.max(e.nodes, axis=0)
                       for e in elements)
    bb_min = np.vstack(np.min(e.nodes, axis=0)
                       for e in elements)
    bb = (bb_min, bb_max)
    return bb


class Sequence:
    """A basic time-varying sequence.

    Defined by control points + interpolation method + extrapolation
    method.

    """
    def __init__(self, seq, typ='smooth', extend='extrapolate'):
        # Input checking
        assert extend in ['extrapolate', 'constant', 'repeat',
                          'repeat continuous']
        assert typ in ['step', 'linear', 'smooth']
        # Parameters
        self.points = seq
        self.typ = typ
        self.extend = extend


class ScaledSequence:
    """A time-varying sequence proportional to another sequence.

    Defined by a scaling factor + another sequence.

    """
    def __init__(self, sequence: Sequence, scale: float):
        self.scale = scale
        self.sequence = sequence


class NameRegistry:
    """Mapping between names and objects.

    Provides `name + nametype → object` for all of an object's names and
    `object → (name + nametype)s`.

    """
    def __init__(self):
        self._from_name = defaultdict(dict)
        self._from_name["canonical"] = {}
        self._from_object = defaultdict(dict)

    def add(self, name, obj, nametype="canonical"):
        """Add a name for an object."""
        # Add the name to the name → object map
        self._from_name[nametype][name] = obj
        # Add the name to the object → names map
        self._from_object[obj][nametype] = name

    def remove_name(self, name, nametype="canonical"):
        """Remove a name for an object."""
        obj = self._from_name[nametype][name]
        # Remove the name from the name → object map
        del self._from_name[nametype][name]
        # Remove the name from the object → names map
        del self._from_object[obj][nametype]

    def remove_object(self, obj):
        """Remove all names for an object"""
        names = self._from_object[obj]
        # Remove all applicable names from the name → object map
        for nametype in names:
            del self._from_name[nametype]
        # Remove all applicable names from the object → names map
        del self._from_object[obj]

    def name(self, obj, nametype="canonical"):
        """Return canonical name for object."""
        return self._from_object[obj][nametype]

    def obj(self, name, nametype="canonical"):
        """Return object by name (and type of name)."""
        return self._from_name[nametype][name]

    def nametypes(self):
        """Return nametypes"""
        return self._from_name.keys()

    def names(self, nametype="canonical"):
        """Return all names of nametype in registry.

        This function is analogous to dict.keys().

        """
        return self._from_name[nametype].keys()

    def objects(self):
        """Return all named objects.

        This function is analogous to dict.values().

        """
        return self._from_object.keys()

    def pairs(self, nametype="canonical"):
        """Return iterable over (name, obj) pairs for nametype.

        This function is analogous to dict.items().

        """
        return self._from_name[nametype].items()

    def __copy__(self):
        """Copy dicts but not named objects."""
        new = type(self)()
        for nametype in self._from_name:
            new._from_name[nametype] = copy(self._from_name[nametype])
        new._from_object = copy(self._from_object)
        return new
