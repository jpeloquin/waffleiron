from copy import deepcopy
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import febtools as feb
import febtools.element
from febtools import XpltReader
from febtools.element import elem_obj
# import xml.etree.ElementTree as ET
from lxml import etree as ET

default_tol = 10*np.finfo(float).eps

# Increase recursion limit for kdtree
import sys
sys.setrecursionlimit(10000)

class Mesh:
    """Stores a mesh geometry."""

    nodes = []
    elements = []
    # nodetree = kd-tree for quick lookup of nodes
    # elem_with_node = For each node, list the parent elements.

    def __init__(self, nodes, elements):

        # Nodes (list of tuples)
        if len(nodes[0]) == 2:
            nodes = [(x, y, 0.0) for (x, y) in nodes]
        self.nodes = nodes

        self.nodetree = KDTree(self.nodes)

        # Elements (list of tuples)
        if elements !=[] and nodes !=[]:
            if isinstance(elements[0], febtools.element.Element):
                self.elements = elements
            else:
                self.elements = [elem_obj(nid, nodes, eid=i)
                                 for i, nid in enumerate(elements)]

        # Precompute derived properties
        self._precompute()


    def _precompute(self):
        """Calculate all derived properties.

        This should be called every time the mesh geometry changes.

        """
        # Create list of parent elements by node
        elem_with_node = [[] for i in xrange(len(self.nodes))]
        for e in self.elements:
            for nid in e.inode:
                elem_with_node[nid].append(e)
        self.elem_with_node = elem_with_node


    def writefeb(self, fpath, materials=None):
        """Write mesh to .feb file.

        Inputs
        ------
        fpath : string
            Path for output file.

        materials : list of Material objects

        """
        root = feb.output.feb_skeleton()
        Material = root.find('Material')
        Geometry = root.find('Geometry')
        Nodes = root.find('Geometry/Nodes')
        Elements = root.find('Geometry/Elements')

        formatted = lambda n: "{:e}".format(n)

        # write nodes
        for i, x in enumerate(self.nodes):
            feb_nid = i + 1 # 1-indexed
            e = ET.SubElement(Nodes, 'node', id="{}".format(feb_nid))
            e.text = ",".join("{:e}".format(n) for n in x)
            Nodes.append(e)

        # write elements
        for i, elem in enumerate(self.elements):
            label = elem.__class__.__name__.lower()
            feb_eid = i + 1 # 1-indexed
            if elem.matl_id is None:
                matl_id = 1
            else:
                matl_id = elem.matl_id + 1
            e = ET.SubElement(Elements, label,
                              id=str(feb_eid),
                              mat=str(matl_id))
            # remember, 1-indexed
            e.text = ",".join("{:d}".format(n + 1) for n in elem.inode)

            # write thicknesses for shells
            if label == 'tri3' or label == 'quad4':
                ElementData = root.find('Geometry/ElementData')
                if ElementData is None:
                    ElementData = ET.SubElement(Geometry, 'ElementData')
            e = ET.SubElement(ElementData, 'element', id=str(feb_eid))
            t = ET.SubElement(e, 'thickness')
            t.text = '0.001, 0.001, 0.001'

        # Write materials

        def default_mat(mat_id):
            m = ET.Element('material',
                           id=str(mat_id),
                           name="Material" + str(mat_id),
                           type="isotropic elastic")
            ET.SubElement(m, 'E').text = "1"
            ET.SubElement(m, 'v').text = "0.3"
            return m

        if materials is None:
            # Count how many materials are defined
            mids = set(e.matl_id for e in self.elements)
            for mid in mids:
                m = default_mat(mid)
                Material.append(m)

        tree = ET.ElementTree(root)
        with open(fpath, 'w') as f:
            tree.write(f, pretty_print=True, xml_declaration=True,
                       encoding='us-ascii')

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

        Remember that the nodes are indexed starting with 0.

        """
        def nodemap(i):
            if i < nid_remove:
                return i
            elif i > nid_remove:
                return i - 1
            else:
                return None
        removal = lambda e: [nodemap(i) for i in e]
        elems = [removal(e.inode) for e in self.elements]
        for i, inode in enumerate(elems):
            self.elements[i].inode = inode
        self.nodes = [x for i, x in enumerate(self.nodes)
                     if i != nid_remove]

    def node_connectivity(self):

        """Count how many elements each node belongs to.

        """
        refcount = [0] * len(self.nodes)
        for e in self.elements:
            for i in e.inode:
                refcount[i] += 1
        return refcount

    def elemcentroid(self):
        """List of element centroids (reference coordinates).

        """
        centroid = []
        for i in range(len(self.elements)):
            x = [self.nodes[inode] for inode in self.elements[i]]
            c = [sum(v) / len(v) for v in zip(*x)]
            yield c

    def elemcoord(self):
        """Generator for element coordinates."""
        for idx in self.elements:
            yield tuple([self.nodes[i] for i in idx])

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
                     for i in e.inode])
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
                lid = e.inode.index(nid) # node id w/in element
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
        # Define new simplices for "other" mesh
        new_simplices = [list(e.inode) for e in other.elements]
        for i, elem in enumerate(other.elements):
            elem.xnode_mesh = nodelist
            for j, nodeid in enumerate(elem.inode):
                inode = np.array(elem.inode)
                inode[j] = newind[nodeid]
                elem.inode = inode
            # Add the new element
            self.elements.append(elem)

    def _build_node_graph(self):
        """Create a node connectivity graph for faster node lookup.

        The connectivity graph is a list of lists.
        `node_graph[i][j]` returns the jth node connected to node
        i.

        """
        for nid in self.nodes:
            pass


class MeshSolution(Mesh):
    """Analysis of a solution step"""

    nodes = []
    elements = []
    materials = {}
    data = {}
    reader = None

    def __init__(self, f=None, step=-1, materials=None):
        if f is None:
            # This is a minimal instance for debugging.
            pass
        else:
            if isinstance(f, str):
                self.reader = XpltReader(f)
            elif isinstance(f, XpltReader):
                self.reader = f
            self.nodes, self.elements = self.reader.mesh()
            self.nodetree = KDTree(self.nodes) # TODO: make this an
                                              # automatic consequence
                                              # of adding a Mesh
                                              # geometry
            self.data = self.reader.stepdata(step)
            self.material_index_xplt = self.reader.material()
            if materials:
                self.assign_materials(materials)
                self.materials = materials

        # Create list of parent elements by node
        elem_with_node = [[] for i in xrange(len(self.nodes))]
        for e in self.elements:
            for nid in e.inode:
                elem_with_node[nid].append(e)
        self.elem_with_node = elem_with_node

    def assign_materials(self, materials):
        """Assign materials from integer codes.

        Parameters
        ----------
        materials : dictionary
            A mapping from integer ids to material objects.  This
            dictionary is usually obtained from
            `input.FebReader.materials()`.

        """
        for i, e in enumerate(self.elements):
            self.elements[i].material = materials[e.matl_id]

    def f(self, istep = -1, r = 0, s = 0, t = 0):
        """Generator for F tensors for each element.

        Global coordinates: x, y, z
        Natural coordinates: r, s, t
        Displacements (global): u, v, w

        """
        for e in self.elements:
            u = np.array([self.data['displacement'][i]
                          for i in e.inode])
            # displacements are exported for each node
            du_dR = np.dot(u.T, e.dN(r, s, t))
            du_dX = np.dot(np.linalg.inv(e.j((r, s, t))), du_dR)
            f = du_dX + np.eye(3)
            yield f

    def s(self):
        """1st Piola-Kirchoff stress for each element.

        The stress is calculated at the center of each element by
        transforming FEBio's Cauchy stress output.
        """
        s = []
        for t, f in zip(self.data['stress'], self.f()):
            # 1/J F S transpose(F) = t
            J = np.linalg.det(f)
            x = np.linalg.solve(1 / J * f, t)
            s = np.linalg.solve(f, x.T).T
            yield s

    def t(self):
        """Cauchy stress tensor for each element.

        The Cauchy stress is read directly from the xplt file.
        """
        for v in self.data['stress']:
            yield v
