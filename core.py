from operator import itemgetter

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from lxml import etree as ET
from copy import deepcopy

import febtools as feb
from febtools.input import XpltReader
from febtools.element import elem_obj
from febtools.geometry import _cross

# Set tolerances
default_tol = 10*np.finfo(float).eps

# Increase recursion limit for kdtree
import sys
sys.setrecursionlimit(10000)

class Model:
    """An FE model: geometry, boundary conditions, solution.

    """
    # If a sequence is assigned to dtmax in this default dictionary,
    # it will be copied when `default_control` is used to initialize a
    # control step.  This may not be desirable.
    default_control = {'time steps': 10,
                       'step size': 0.1,
                       'max refs': 15,
                       'max ups': 10,
                       'dtol': 0.001,
                       'etol': 0.01,
                       'rtol': 0,
                       'lstol': 0.9,
                       'time stepper': {'dtmin': 0.01,
                                        'dtmax': 0.1,
                                        'max retries': 5,
                                        'opt iter': 10},
                       'analysis type': 'static',
                       'plot level': 'PLOT_MAJOR_ITRS'}

    def __init__(self, mesh):
        self.mesh = mesh

        self.materials = {}
        self.material_names = {}
        self.solution = None # the solution for the model

        self.fixed_nodes = {'x': set(),
                            'y': set(),
                            'z': set(),
                            'pressure': set(),
                            'concentration': set()}
        # Note: for multiphasic problems, concentration is a list of
        # sets

        # initial nodal values
        self.initial_values = {'velocity': [],
                               'fluid_pressure': [],
                               'concentration': []}
        # Note: for multiphasic problems, concentration is a list of
        # lists.
        self.steps = []
        self.add_step()

    def add_step(self, module='solid', control=None):
        """Add a step with default control values and no BCs.

        """
        if control is None:
            control = deepcopy(self.default_control)
        step = {'module': module,
                'control': control,
                'bc': {}}
        self.steps.append(step)

    def apply_nodal_displacement(self, node_ids, values, sequence,
                                 axis, step_id=-1):
        """Apply a boundary condition to a step.

        """
        for i, v in zip(node_ids, values):
            bc_node = self.steps[step_id]['bc'].setdefault(i, {})
            bc_node[axis] = {'sequence': sequence,
                             'value': v}

    def apply_solution(self, solution, t=None):
        """Attach a solution to the model.

        By default, the last timepoint in the solution is applied.

        """
        self.solution = solution
        # apply node data
        if t is None: # use last timestep
            t = solution.times[-1]
        data = solution.stepdata(time=t)
        properties = data['node']
        for k,v  in properties.items():
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
        removal = lambda e: [nodemap(i) for i in e]
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

    def find_nearest_node(self, x, y, z=None):
        """Find node nearest (x, y, z)

        Notes
        -----
        Does not handle the case where nodes are superimposed.

        """
        if z is None:
            p = (x, y, 0)
        else:
            p = (x, y, z)
        d = np.array(self.nodes) - p
        d = np.sum(d**2., axis=1) # don't need square root if just
                                  # finding nearest
        idx = np.argmin(abs(d))
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

    def merge(self, other, candidates='auto', tol=default_tol):
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
        newind = [] # new indices for 'other' nodes after merge
        # Iterate over nodes in 'other'
        for i, p in enumerate(other.nodes):
            if i in candidates:
                # Find node in 'self' closest to p
                imatch = self.find_nearest_node(*p)
                pmatch = self.nodes[imatch]
                i_cd = candidates.index(i)
                if dist[i_cd, imatch] < tol:
                    # Make all references in 'other' to p use pmatch
                    # instead
                    newind.append(imatch)
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
        new_simplices = [list(e.ids) for e in other.elements]
        for i, e in enumerate(other.elements):
            new_ids = [newind[j] for j in e.ids]
            e.ids = new_ids
            # Merge the element to the first mesh
            self.elements.append(e)

    def _build_node_graph(self):
        """Create a node connectivity graph for faster node lookup.

        The connectivity graph is a list of lists.
        `node_graph[i][j]` returns the jth node connected to node
        i.

        """
        for nid in self.nodes:
            pass


class Sequence:
    """A time-varying sequence for step control.

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
