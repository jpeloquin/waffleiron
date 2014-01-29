import numpy as np
import febtools as feb
import febtools.element
from febtools import XpltReader
from febtools.element import elem_obj
# import xml.etree.ElementTree as ET
from lxml import etree as ET

class Mesh:
    """Stores a mesh geometry."""

    node = []
    element = []

    def __init__(self, node=None, element=None):

        # Node list
        if node is None:
            self.node = []
        else:
            self.node = node

        # Element list
        if element is not None and node is not None:
            if element[0] is febtools.element.Element:
                self.element = element
            else:
                self.element = [elem_obj(nid, node, eid=i)
                                for i, nid in enumerate(element)]

    def readfeb(self, f):
        """Read a mesh from an .feb file.

        """
        root = ET.parse(f).getroot()
        if root.tag != "febio_spec":
            raise Exception("Root node is not 'febio_spec': '" +
                            fpath + "' is not a valid .feb file.")
        self.node = [tuple([float(a) for a in b.text.split(",")])
                     for b in root.findall("./Geometry/Nodes/*")]
        element = [tuple([int(a) - 1 for a in b.text.split(",")])
                   for b in root.findall("./Geometry/Elements/*")]
        element = [elem_obj(nid, self.node, eid=i)
                   for i, nid in enumerate(element)]
        self.element = element

    def writefeb(self, fpath):
        """Write mesh to .feb file.

        """
        root = feb.output.feb_skeleton()
        Geometry = root.find('Geometry')
        Nodes = root.find('Geometry/Nodes')
        Elements = root.find('Geometry/Elements')

        formatted = lambda n: "{:e}".format(n)

        # write nodes
        for i, x in enumerate(self.node):
            feb_nid = i + 1 # 1-indexed
            e = ET.SubElement(Nodes, 'node', id="{}".format(feb_nid))
            e.text = ",".join("{:e}".format(n) for n in x)
            Nodes.append(e)
        
        # write elements
        for i, elem in enumerate(self.element):
            label = elem.__class__.__name__.lower()
            feb_eid = i + 1 # 1-indexed
            matl_id = elem.matl_id
            if matl_id is None:
                matl_id = 1
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

        tree = ET.ElementTree(root)
        with open(fpath, 'w') as f:
            tree.write(f, pretty_print=True, xml_declaration=True,
                       encoding='us-ascii')

    def clean_nodes(self):
        """Remove any nodes that are not part of an element.

        """
        refcount = self.node_connectivity()
        for i in reversed(xrange(len(self.node))):
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
        elems = [removal(e.inode) for e in self.element]
        for i, inode in enumerate(elems):
            self.element[i].inode = inode
        self.node = [x for i, x in enumerate(self.node)
                     if i != nid_remove]

    def node_connectivity(self):

        """Count how many elements each node belongs to.

        """
        refcount = [0] * len(self.node)
        for e in self.element:
            for i in e.inode:
                refcount[i] += 1
        return refcount

    def elemcentroid(self):

        """List of element centroids (reference coordinates)."""
        centroid = []
        for i in range(len(self.element)):
            x = [self.node[inode] for inode in self.element[i]]
            c = [sum(v) / len(v) for v in zip(*x)]
            yield c

    def elemcoord(self):
        """Generator for element coordinates."""
        for idx in self.element:
            yield tuple([self.node[i] for i in idx])

    def find_nearest_node(self, x, y, z=None):
        """Find node nearest (x, y, z)

        """
        if z is None:
            p = (x, y, 0)
        else:
            p = (x, y, z)
        d = np.array(self.node) - p
        d = np.sum(d**2., axis=1)**0.5
        idx = np.argmin(abs(d))
        return idx

    def elem_with_node(self, idx):
        """Return elements containing a node

        idx := node id

        """
        eid = set(ii for (ii, r) in enumerate(self.element)
                  if idx in r.inode)
        elements = [self.element[i] for i in eid]
        return set(elements)

    def conn_elem(self, elements):
        """Find elements connected to elements.

        """
        nodes = set([i for e in elements
                     for i in e.inode])
        elements = []
        for idx in nodes:
            elements = elements + list(self.elem_with_node(idx))
        return set(elements)

    def _build_node_graph(self):
        """Create a node connectivity graph for faster node lookup.

        The connectivity graph is a list of lists.
        `node_graph[i][j]` returns the jth node connected to node
        i.

        """
        for nid in self.node:
            pass


class MeshSolution(Mesh):
    """Analysis of a solution step"""

    node = []
    element = []
    data = {}
    reader = None
    material_index = []

    
    def __init__(self, f=None, step=-1, matl_map=None):
        if f is None:
            # This is a minimal instance for debugging.
            pass
        else:
            if isinstance(f, str):
                self.reader = XpltReader(f)
            elif isinstance(f, XpltReader):
                self.reader = f
            self.node, self.element = self.reader.mesh()
            self.data = self.reader.stepdata(step)
            self.material_index_xplt = self.reader.material()
            self.material_map = matl_map

    def assign_materials(self, matl_map):
        """Assign materials from integer codes.

        mat_map : a dictionary mapping integers to material classes

        """
        for i, e in enumerate(self.element):
            matname = matl_map[e.mat_id]
            self.element[i].material = feb.material.getclass(matname)

    def f(self, istep = -1, r = 0, s = 0, t = 0):
        """Generator for F tensors for each element.
        
        Global coordinates: x, y, z
        Natural coordinates: r, s, t
        Displacements (global): u, v, w
    
        """
        for e in self.element:
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
