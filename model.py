"""Module for high-level model-related classes."""
# Built-in packages
from copy import deepcopy
from math import inf
import sys
# Public packages
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
# Intra-package imports
from .control import default_control_section
from .core import _default_tol, Body, NodeSet, NameRegistry, _validate_dof
from .selection import e_grow
from . import util

# Increase recursion limit for kdtree
sys.setrecursionlimit(10000)


class Model:
    """An FE model: geometry, boundary conditions, solution.

    """
    def __init__(self, mesh):
        if type(mesh) is not Mesh:
            raise TypeError("{} is not of type"
                            "febtools.core.Model".format(mesh))

        self.mesh = mesh  # The model geometry
        self.solution = None  # The solution for the model

        # If a sequence is assigned to dtmax in this default dictionary,
        # it will be copied when `default_control` is used to initialize a
        # control step.  This may not be desirable.
        self.default_control = default_control_section()

        self.output = {"variables": None}

        self.environment = {"temperature": 294}  # K
        self.constants = {"R": 8.31446261815324,  # J/mol·K
                          "F": 96485.33212}  # C/mol

        axes = ('x1', 'x2', 'x3', 'pressure', 'concentration')
        self.fixed = {'node': {k: NodeSet() for k in axes},
                      'body': {k: set() for k in axes}}
        self.fixed['body'].update({'α1': set(),
                                   'α2': set(),
                                   'α3': set()})
        axes = ('x1', 'x2', 'x3', 'pressure', 'concentration')
        # Note: for multiphasic problems, concentration is a list of
        # sets
        #
        # TODO: Make the specification of fixed-for-all-time BCs have
        # the same format as for BCs in steps.

        # Global "boundary conditions" that apply to all steps.  Same
        # format as self.steps[i]["bc"]
        self.varying = {'node': {},
                        'body': {}}

        # Contact
        self.constraints = []
        # ^ this might be better split between interactions (two parts
        # of the model interacting with each other according to some
        # equality) and true constraints (one part of the model is
        # constrained by some equality).

        # Initialize dictionaries to hold named named entities
        self.named = {"materials": NameRegistry(),
                      "node sets": NameRegistry(),
                      "face sets": NameRegistry(),
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
            control = default_control_section()
        step = {'module': module,
                'control': control,
                'bc': {'node': {},
                       'body': {}}}
        self.steps.append(step)


    def apply_nodal_bc(self, node_ids, dof, variable, sequence,
                       scales=None, step_id=-1):
        """Apply a boundary condition to a set of nodes.

        The boundary condition is applied to the selected solution step,
        which by default is the last step.  (Note that if you export a
        model to FEBio, it will automatically leak the boundary
        condition to all subsequent steps.)

        node_ids := collection of integer node IDs

        dof := 'x1', 'x2', 'x3', 'fluid', 'temperature', or 'charge'.

        variable := 'displacement', 'force', 'pressure', or 'flow'

        sequence := conditions.Sequence object or 'fixed'

        scales := dict of integer node ID → scale.  If None, scale will
        be set to 1 for all nodes.

        """
        default_scale = 1
        _validate_dof(dof)
        if sequence == 'fixed':
            scales = [None]*len(node_ids)
        elif scales is None:  # variable BC
            scales = {i: default_scale for i in node_ids}
        for i in node_ids:
            bc_node = self.steps[step_id]['bc']['node'].setdefault(i, {})
            bc_node[dof] = {'variable': variable,
                             'sequence': sequence,
                             'scale': scales[i]}
        # TODO: Support global nodal BCs.


    def apply_body_bc(self, body, dof, variable, sequence, scale=1,
                      step_id=-1):
        """Apply a variable displacement boundary condition to a body.

        The boundary condition is applied to the selected solution step,
        which by default is the last step.  (Note that if you export a
        model to FEBio, it will automatically leak the boundary
        condition to all subsequent steps.)

        dof := 'x1', 'x2', 'x3', 'fluid', 'temperature', or 'charge'.

        sequence := conditions.Sequence object or 'fixed'

        body := Body or ImplicitBody object

        step_id := int or None.  The step ID to which to apply the body
        constraint.  If None, the body constraint will be applied to
        model.constraints, which is sort of an initial or universal
        step.

        """
        _validate_dof(dof, body=True)
        if sequence == 'fixed':
            scale = None
        if step_id is None:
            # Setting global BC
            bc_dict = self.varying["body"]
        else:
            # Setting step-local BC
            bc_dict = self.steps[step_id]["bc"]["body"]
        bc = bc_dict.setdefault(body, {})
        bc[dof] = {'variable': variable,
                    'sequence': sequence,
                    'scale': scale}
        # TODO: Remove scale; just used ScaledSequence for that case


    def apply_solution(self, solution, t=None, step=None):
        """Attach a solution to the model.

        By default, the last timepoint in the solution is applied.

        """
        self.solution = solution
        # apply node data
        if t is None and step is None:  # use last timestep
            step = -1
        elif t is not None and step is None:
            try:
                step_times = solution.times
            except AttributeError:
                # `solution` is probably an XpltReader instance (which
                # is sort of deprecated)
                step_times = solution.step_times
            step_ids = [a for a in range(len(step_times))]
            step = util.find_closest_timestep(t, step_times, step_ids)
        elif t is not None and step is not None:
            raise ValueError("Provide either `t` or `step`, not both.")
        data = solution.step_data(step)
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
        elements := list of Element objects that index into `nodes`

        """
        # Nodes
        if np.array(nodes).size == 0:
            self.nodes = nodes
        # if nodes are 2D, add z = 0
        elif len(nodes[0]) == 2:
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
        ## A mesh might have an empty node and element list.  Commit
        ## 25146ae specifically made prepare() tolerate empty meshes, so
        ## presumably someone needs empty meshes.
        if (self.nodes is not None) and (len(self.nodes) > 0):
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
            body_elements = e_grow([e], untouched_elements, inf)
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
