import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from lxml import etree as ET
from copy import deepcopy

import febtools as feb
from febtools.input import XpltReader
from febtools.element import elem_obj

# Set tolerances
default_tol = 10*np.finfo(float).eps

# Increase recursion limit for kdtree
import sys
sys.setrecursionlimit(10000)

class Model:
    """An FE model: geometry, boundary conditions, solution.

    """
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
        self.solution = None # the solution for the model

        self.sequences = []

        self.fixed_nodes = {'x': set(),
                            'y': set(),
                            'z': set(),
                            'pressure': set(),
                            'concentration': set()}
        # Note: for multiphasic problems, concentration is a list of
        # sets

        self.steps = None # list

        # initial nodal values
        self.initial_values = {'velocity': [],
                               'fluid_pressure': [],
                               'concentration': []}
        # Note: for multiphasic problems, concentration is a list of
        # lists.
        self.steps = []
        self.add_step()

    def add_sequence(self, points, typ='smooth',
                     extend='extrapolate'):
        """Define a sequence.

        """
        seq = {'type': typ,
               'extend': extend,
               'points': points}
        self.sequences.append(seq)

    def add_step(self, module='solid', control=None):
        """Add a step with default control values and no BCs.

        """
        if control is None:
            control = deepcopy(self.default_control)
        step = {'module': module,
                'control': control,
                'bc': {}}
        self.steps.append(step)

    def apply_bc(self, node_ids, values, sequence_id,
                 axis, step_id=-1):
        """Apply a boundary condition to a step.

        """
        for i, v in zip(node_ids, values):
            bc_node = self.steps[step_id]['bc'].setdefault(i, {})
            bc_node[axis] = {'sequence': sequence_id,
                             'value': v}

    def apply_solution(self, solution, t=None):
        """Attach a solution to the model.

        By default, the last timepoint in the solution is applied.

        """
        self.solution = solution
        # apply node data
        if t is None: # use last timestep
            t = solution.time[-1]
        data = solution.stepdata(time=t)
        properties = data['node']
        for k,v  in properties.iteritems():
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
            nodes = [self.nodes[i] for i in e.ids]
            e.nodes = nodes

    def prepare(self):
        """Calculate all derived properties.

        This should be called every time the mesh geometry changes.

        """
        # Create KDTree for fast node lookup
        self.nodetree = KDTree(self.nodes)

        # Create list of parent elements by node
        elem_with_node = [[] for i in xrange(len(self.nodes))]
        for e in self.elements:
            for i in e.ids:
                elem_with_node[i].append(e)
        self.elem_with_node = elem_with_node

        # Faces
        self.faces = [Face(f) for e in self.elements
                      for f in e.faces()]

        # Create list of parent faces by node
        faces_with_node = [set() for i in xrange(len(self.nodes))]
        for f in self.faces:
            for i in f.ids:
                faces_with_node[i].add(f)
        self.faces_with_node = faces_with_node

    def clean_nodes(self):
        """Remove any nodes that are not part of an element.

        """
        refcount = self.node_connectivity()
        for i in reversed(xrange(len(self.nodes))):
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

    def element_containing_point(self, point):
        """Return element containing a point

        """
        # Provide 2 dimensions so iteration over the first will always
        # iterate over points
        point = np.array(point)

        # Determine closest node (point Q) to each point P
        d, node_idx = self.nodetree.query(point, k=2)
        # Handle superimposed points
        if abs(d[0] - d[1]) < np.finfo('float').eps:
            # These two points are superimposed.  There may be more.
            node_idx = self.nodetree.query_ball_point(point,
                            r=d[0] + np.finfo('float').eps)
        else:
            # The two points are not superimposed; we only want one.
            node_idx = [node_idx[0]]

        # Test each element connected to closest node(s) for
        # containing the point
        elems = []
        for nid in node_idx:
            # Iterate over connected elements
            for e in self.elem_with_node[nid]:
                pt_q = self.nodes[nid] # the closest node
                v_pq = point - pt_q # vector from Q to point of
                                    # interest
                # Check if point is in element
                lid = e.ids.index(nid) # node id w/in element
                if e.is_planar:
                    edge_ids = e.edges_with_node(lid)
                    normals = e.edge_normals()
                    normals = [normals[i] for i in edge_ids]
                else:
                    face_ids = e.faces_with_node(lid)
                    normals = e.face_normals()
                    normals = [normals[i] for i in face_ids]
                # Test if line PQ is perpindicular or antiparallel to
                # the normal of each face connected to Q; if yes, P is
                # interior to all three faces and the element contains
                # P.
                if np.all([np.dot(n, v_pq) <= 0 for n in normals]):
                    return e
        # If no element contains the point, the point is outside the
        # mesh or inside a hole.
        return None

    def merge(self, other, candidates='auto', tol=default_tol):
        """Merge this mesh with another

        Inputs
        ------
        other : Mesh object
            The mesh to merge with this one.
        candidates : {'auto', list of int}
            If 'auto' (the default), combine all nodes in the `other`
            mesh that are distance < `tol` from a node in the current
            mesh.  The simplices of `other` will be updated
            accordingly.  If `nodes` is a list, use only the node
            indices in the list as candidates for combination.  These
            indexes are in the domain of `other`.

        Returns
        -------
        Mesh object
            The merged mesh

        """
        dist = cdist(other.nodes, self.nodes, 'euclidean')
        newind = [] # new indices for 'other' nodes after merge
        # copy nodelist so any error will not corrupt the original mesh
        nodelist = deepcopy(self.nodes)
        # Iterate over nodes in 'other'
        for i, p in enumerate(other.nodes):
            try_combine = ((candidates == 'auto') or
                           (candidates != 'auto' and i in candidates))
            if try_combine:
                # Find node in 'self' closest to p
                imatch = self.find_nearest_node(*p)
                pmatch = self.nodes[imatch]
                if dist[i, imatch] < tol:
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

class Face:
    """An oriented list of nodes representing a face.

    """
    def __init__(self, ids, mesh=None):
        """Create a face from a list of node ids.

        """
        # Defaults
        self.fc_faces = set() # neighboring faces sharing all nodes;
                              # i.e. fully connected
        self.ec_faces = set() # neighboring faces sharing an edge;
                              # i.e. edge connected
        self.normal = None # To be used by a mesh objected to store
                           # normals.
        # Set ids
        self.ids = tuple(ids)
