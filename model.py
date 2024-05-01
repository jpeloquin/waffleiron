"""Module for high-level model-related classes."""

# Built-in packages
from collections import namedtuple
from math import inf
import sys
from numbers import Real
from typing import Iterable, Optional, Union, Dict

# Public packages
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

# Intra-package imports
from .control import Step, Ticker, IterController, Solver
from .core import (
    _DEFAULT_TOL,
    Body,
    NodeSet,
    ScaledSequence,
    NameRegistry,
    _validate_dof,
)
from .element import Element
from .select import e_grow, find_closest_timestep
from . import util

# Increase recursion limit for kdtree
sys.setrecursionlimit(10000)


NamedStep = namedtuple("NamedStep", ["step", "name"])


class Model:
    """An FE model: geometry, boundary conditions, solution."""
    constants: dict[str, Real]

    def __init__(self, mesh):
        if type(mesh) is not Mesh:
            raise TypeError("{} is not of type" "waffleiron.core.Model".format(mesh))

        self.mesh = mesh  # The model geometry
        self.solution = None  # The solution for the model

        self.name = None  # model name, to facilitate error messages

        self.output = {"variables": []}
        # ^ TODO: probably should be a set; doesn't make sense to have an output
        #  variable twice
        # TODO: write_feb only populates this output variable list if it
        # is empty, which is bad if you just want to force one output
        # variable to be included while retaining the auto-export

        self.environment = {"temperature": 294}  # K
        self.constants = {"R": 8.31446261815324, "F": 96485.33212}  # J/mol·K  # C/mol

        # Fixed conditions
        self.fixed = {
            "node": {
                ("x1", "displacement"): set(),
                ("x2", "displacement"): set(),
                ("x3", "displacement"): set(),
                ("fluid", "pressure"): set(),
                ("solute", "concentration"): set(),
            },
            "body": {
                ("x1", "displacement"): set(),
                ("x2", "displacement"): set(),
                ("x3", "displacement"): set(),
                ("fluid", "pressure"): set(),
                ("solute", "concentration"): set(),
                ("α1", "rotation"): set(),
                ("α2", "rotation"): set(),
                ("α3", "rotation"): set(),
            },
        }
        # Note: for multiphasic problems, concentration is a list of sets
        #
        # TODO: Make the specification of fixed-for-all-time BCs have
        # the same format as for BCs in steps.

        # Global "boundary conditions" that apply to all steps.  Same
        # format as self.steps[i]["bc"], which is dict[dof: str →
        # condition: dict]
        self.varying = {"node": {}, "body": {}}

        # Contact
        self.constraints = []
        # ^ this might be better split between interactions (two parts
        # of the model interacting with each other according to some
        # equality) and true constraints (one part of the model is
        # constrained by some equality).

        # TODO: Compose all aspects of mesh into model
        self.named = self.mesh.named

        # initial nodal values
        self.initial_values = {
            "velocity": [],
            "fluid_pressure": [],
            "concentration": [],
        }
        # Note: for multiphasic problems, concentration is a list of
        # lists.
        self.steps = []

    def add_contact(self, constraint, step_idx=None):
        if step_idx is None:
            # Apply the contact as an atemporal constraint
            self.constraints.append(constraint)
        else:
            # Apply the contact as a step-specific constraint
            self.steps[step_idx].step.bc["contact"].append(constraint)

    def add_step(self, step, name: Union[str, None] = None):
        """Add a step with default control values and no BCs."""
        step = NamedStep(step, name=name)
        self.steps.append(step)

    def apply_nodal_bc(
        self,
        node_ids,
        dof,
        variable,
        sequence,
        step: Step,
        scales: Optional[dict] = None,
        relative=False,
    ):
        """Apply a boundary condition to a set of nodes.

        The boundary condition is applied to the selected solution step,
        which by default is the last step.  (Note that if you export a
        model to FEBio, it will automatically leak the boundary
        condition to all subsequent steps.)

        node_ids := collection of integer node IDs

        dof := 'x1', 'x2', 'x3', 'fluid', 'temperature', or 'charge'.

        variable := 'displacement', 'force', 'pressure', or 'flow'

        sequence := conditions.Sequence object or 'fixed'

        step := The step to which the nodal boundary condition applies.
        This is a required argument because a time-varying nodal
        boundary condition cannot (yet) be defined outside of a
        simulation step.

        scales := dict of integer node ID → scale.  If None, scale will
        be set to 1 for all nodes.  If you want to apply a different
        scaling factor to all nodes, use a ScaledSequence.

        """
        default_scale = 1
        _validate_dof(dof)
        if sequence == "fixed":
            scales = {i: None for i in node_ids}
        else:
            # variable BC
            if scales is None:
                scales = {i: default_scale for i in node_ids}
            if isinstance(sequence, ScaledSequence):
                scales = {i: sequence.scale * scales[i] for i in node_ids}
        for i in node_ids:
            bc_node = step.bc["node"].setdefault(i, {})
            bc_node[dof] = {
                "variable": variable,
                "sequence": sequence,
                "scale": scales[i],
                "relative": relative,
            }
        # TODO: Support global nodal BCs.

    def apply_body_bc(
        self, body, dof, variable, sequence, scale=1, relative=False, step=None
    ):
        """Apply a variable displacement boundary condition to a body.

        The boundary condition is applied to the selected solution step,
        which by default is the last step.  (Note that if you export a
        model to FEBio, it will automatically leak the boundary
        condition to all subsequent steps.)

        :param dof: 'x1', 'x2', 'x3', 'fluid', 'temperature', or 'charge'.

        :param variable: "displacement", "rotation", "force", or "moment", etc.

        :param sequence: conditions.Sequence object or 'fixed'

        :param body: Body or ImplicitBody object

        :param step: Step object or None.  The step to which the rigid body boundary
        condition applies.  If None, the condition will be applied globally.
        """
        _validate_dof(dof, body=True)
        if step is None:
            # Setting global BC
            bc_dict = self.varying["body"]
        else:
            # Setting step-local BC
            bc_dict = step.bc["body"]
        bc = bc_dict.setdefault(body, {})
        bc[dof] = {
            "variable": variable,
            "sequence": sequence,
            "scale": scale,
            "relative": relative,
        }
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
            step = find_closest_timestep(t, step_times, step_ids)
        elif t is not None and step is not None:
            raise ValueError("Provide either `t` or `step`, not both.")
        data = solution.step_data(step)
        for (varname, entity_type), values in data.items():
            if not entity_type == "node":
                continue
            self.apply_nodal_properties(varname, values)

    def apply_nodal_properties(self, key, values):
        """Apply nodal properties to each element."""
        for e in self.mesh.elements:
            e.properties[key] = np.array([values[i] for i in e.ids])


class Mesh:
    """Stores a mesh geometry."""

    # nodetree = kd-tree for quick lookup of nodes
    # elem_with_node = For each node, list the parent elements.

    def __init__(self, nodes, elements):
        """Create mesh from nodes and element objects.

        :param nodes: Array of (x, y, z) points
        :param elements: List of Element objects that index into `nodes`

        """
        nodes = np.array(nodes)
        # Nodes
        if nodes.size == 0:
            self._nodes = nodes
        # if nodes are 2D, add z = 0
        elif len(nodes[0]) == 2:
            self._nodes = np.hstack([nodes, np.zeros((len(nodes), 1))])
        else:
            self._nodes = nodes
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

        # Initialize dictionaries to hold named entities
        self.named = {
            "materials": NameRegistry(),
            "node sets": NameRegistry(),
            "face sets": NameRegistry(),
            "element sets": NameRegistry(),
            "sequences": NameRegistry(),
        }

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

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, value):
        self._nodes = np.array(value)

    def add_elements(self, nodes, elements: Iterable[Element]):
        """Add elements to the mesh

        Use this function rather than appending directly to self.elements to ensure
        proper housekeeping.

        """
        # TODO: Add option to copy rather than modify elements.  Or warn if elements
        # already belong to a different mesh.
        offset = len(self.nodes)
        self.nodes = np.vstack([self.nodes, nodes])
        for e in elements:
            e.mesh = self
            e.ids += offset
            self.elements.append(e)

    def update_elements(self):
        """Update elements with current node coordinates."""
        for e in self.elements:
            nodes = np.array([self.nodes[i] for i in e.ids])
            e.nodes = nodes

    def prepare(self):
        """Calculate all derived properties.

        This should be called every time the mesh geometry changes.

        """
        # Create KDTree for fast node lookup
        # A mesh might have an empty node and element list.  Commit
        # 25146ae specifically made prepare() tolerate empty meshes, so
        # presumably someone needs empty meshes.
        if (self.nodes is not None) and (len(self.nodes) > 0):
            self.nodetree = KDTree(self.nodes)

        # Create list of parent elements by node
        elem_with_node = [[] for i in range(len(self.nodes))]
        for e in self.elements:
            for i in e.ids:
                elem_with_node[i].append(e)
        self.elem_with_node = elem_with_node

    def faces_with_node(self, idx):
        """Return face tuples containing node index.

        The faces are tuples of integers.

        """
        candidate_elements = self.elem_with_node[idx]
        faces = [f for e in candidate_elements for f in e.faces() if idx in f]
        return faces

    def clean_nodes(self):
        """Remove any nodes that are not part of an element."""
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
        self.nodes = [x for i, x in enumerate(self.nodes) if i != nid_remove]

    def node_connectivity(self):
        """Count how many elements each node belongs to."""
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
        d = np.sum(d**2.0, axis=1)
        idx = np.nonzero(d == np.min(d))[0]
        return idx

    def conn_elem(self, elements):
        """Find elements connected to elements."""
        nodes = set([i for e in elements for i in e.ids])
        elements = []
        for idx in nodes:
            elements = elements + self.elem_with_node[idx]
        return set(elements)

    def merge(self, other, candidates="auto", tol=_DEFAULT_TOL):
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
        if candidates == "auto":
            candidates = range(len(other.nodes))
        # Copy the node list so errors will not corrupt the original
        nodelist = [node for node in self.nodes]
        nodes_cd = [other.nodes[i] for i in candidates]
        dist = cdist(nodes_cd, self.nodes, "euclidean")
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
        self.nodes = np.array(nodelist)
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
