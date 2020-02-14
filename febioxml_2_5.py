# Base packages
from collections import defaultdict
# System packages
from lxml import etree as ET
# Same-package modules
from .core import Sequence
from .control import step_duration
from .febioxml import *
from .febioxml import _to_number, _find_unique_tag

feb_version = 2.5

def parts(model):
    """Return list of parts.

    Here, 1 part = all elements of the same type with the same material.

    PreView has a notion of parts that is separate from both material
    and element type and named element sets.  This doesn't seem to
    matter to FEBio itself, though.  It may interfere with
    postprocessing, in which case the definition of a "part" from the
    perspective of febtools will have to be changed.

    """
    # TODO: Modify the definition of parts such that 1 part = all
    # *connected* elements with the same material.
    #
    # Assemble elements into blocks with like type and material.
    # Elemsets uses material instances as keys.  Each item is a
    # dictionary using element classes as keys, with items being tuples
    # of (element_id, element).
    by_mat_type = {}
    for i, elem in enumerate(model.mesh.elements):
        subdict = by_mat_type.setdefault(elem.material, {})
        like_elements = subdict.setdefault(elem.__class__, [])
        like_elements.append((i, elem))
    # Convert nested dictionaries to a list
    parts = []
    for mat in by_mat_type:
        for typ in by_mat_type[mat]:
            parts.append({'label': None,
                          'material': mat,
                          'element_type': typ,
                          'elements': by_mat_type[mat][typ]})
    return parts

def geometry_section(model, parts, material_registry):
    """Return XML tree for Geometry section."""
    e_geometry = ET.Element('Geometry')
    # Write <nodes>
    e_nodes = ET.SubElement(e_geometry, 'Nodes')
    for i, x in enumerate(model.mesh.nodes):
        feb_nid = i + 1  # 1-indexed
        e = ET.SubElement(e_nodes, 'node', id="{}".format(feb_nid))
        e.text = vec_to_text(x)
        e_nodes.append(e)
    # Write <elements> for each part
    for part in parts:
        e_elements = ET.SubElement(e_geometry, 'Elements')
        e_elements.attrib['type'] = part['element_type'].feb_name
        mat_id = material_registry.name(part['material'], "ordinal_id")
        e_elements.attrib['mat'] = str(mat_id + 1)
        for i, e in part['elements']:
            e_element = ET.SubElement(e_elements, 'elem')
            e_element.attrib['id'] = str(i + 1)
            e_element.text = ', '.join(str(i+1) for i in e.ids)
    return e_geometry


def elem_var_fiber_xml(e):
    tag = ET.Element('elem')
    raise NotImplementedError
    # TODO: Implement this.  But it is not clear how the fiber direction
    # element property is supposed to be written in FEBio XML 2.5.
    # PreView won't export it.

def elem_var_thickness_xml(e):
    raise NotImplementedError

def elem_var_vonmises_xml(e):
    raise NotImplementedError

def elem_var_prestretch_xml(e):
    raise NotImplementedError

element_var_feb = {'v_fiber': {'name': 'fiber',
                               'fn': elem_var_fiber_xml},
                   'thickness': {'name': 'shell thickness',
                                 'fn': elem_var_thickness_xml},
                   'von Mises': {'name': 'MRVonMisesParameters',
                                 'fn': elem_var_vonmises_xml},
                   'prestretch': {'name': 'pre_stretch',
                                  'fn': elem_var_prestretch_xml}}


def iter_node_conditions(root):
    """Return generator over prescribed nodal condition info.

    Returns dict of property names → values.  All properties are
    guaranteed to be not-None, except "nodal values", which will be None
    if the condition applies the same condition to all nodes.

    All returned IDs are 0-indexed for consistency with febtools.

    """
    step_id = -1  # Curent step ID (0-indexed)
    for e_Step in root.findall("Step"):
        step_id += 1
        for e_prescribe in e_Step.findall("Boundary/prescribe"):
            # Re-initialize output
            info = {"node set name": None,
                    "axis": None,  # x1, fluid, charge, etc.
                    "variable": None,  # displacement, force, etc.
                    "sequence ID": None,
                    "scale": 1.0,
                    "nodal values": None,
                    "step ID": None}
            # Read values
            info["node set name"] = e_prescribe.attrib["node_set"]
            info["axis"] = axis_from_febio[e_prescribe.attrib["bc"]]
            info["variable"] = "displacement"
            e_scale = e_prescribe.find("scale")
            seq_scale = 0.0  # FEBio default
            if e_scale.text is not None:
                seq_scale = _to_number(e_scale.text)
            info["sequence ID"] =  _to_number(e_scale.attrib["lc"]) - 1
            e_value = e_prescribe.find("value")
            if e_value is not None:
                if "node_data" in e_value.attrib:
                    # Node-specific data; look up the data in MeshData
                    info["scale"] = seq_scale
                    e_NodeSet = _find_unique_tag(root,
                                                 "Geometry/NodeSet[@name='" +
                                                 info['node set name'] +
                                                 "']")
                    e_NodeData = _find_unique_tag(root,
                                                  "MeshData/NodeData[@name='" +
                                                  e_value.attrib["node_data"] +
                                                  "']")
                    info["nodal values"] = {}
                    for e_node, e_value in zip(e_NodeSet.findall("node"),
                                               e_NodeData.findall("node")):
                        id_ = int(e_node.attrib["id"]) - 1
                        info["nodal values"][id_] = _to_number(e_value.text)
                else:
                    # One value for all nodes; redundant with "scale"
                    val_scale = _to_number(e_value.text)
                    info["scale"] = seq_scale * val_scale
            e_relative = e_prescribe.find("relative")
            if e_relative is not None and e_relative.text == "1":
                raise ValueError("Relative prescribed nodal displacements are not yet supported.")
            info["step ID"] = step_id
            yield info


def meshdata_section(model):
    """Return XML tree for MeshData section."""
    e_meshdata = ET.Element('MeshData')
    e_edata_mat_axis = ET.Element("ElementData", var="mat_axis",
                                  elem_set="autogen-mat_axis")
    e_elemset = ET.Element("ElementSet", name="autogen-mat_axis")
    i_elemset = 0
    # ^ index into the extra element set we're forced to construct
    for i, e in enumerate(model.mesh.elements):
        # Write local basis if defined
        if e.local_basis is not None:
            e_elem = ET.SubElement(e_edata_mat_axis, "elem", lid=str(i_elemset+1))
            e_elem.append(ET.Comment(f"Element {i + 1}"))
            i_elemset += 1
            ET.SubElement(e_elem, "a").text = bvec_to_text(e.local_basis[0])
            ET.SubElement(e_elem, "d").text = bvec_to_text(e.local_basis[1])
            ET.SubElement(e_elemset, "elem", id=str(i+1))
    if len(e_edata_mat_axis) != 0:
        e_meshdata.append(e_edata_mat_axis)
    return e_meshdata, e_elemset


def split_bc_names(s):
    """Split boundary condition names.

    In FEBio XML 2.5, each boundary condition is separated by a comma.

    """
    return [bc.strip() for bc in s.split(",")]
