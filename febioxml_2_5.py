# Base packages
from collections import defaultdict

# System packages
from lxml import etree as ET

# Same-package modules
from .core import Sequence
from .control import step_duration
from .febioxml import *


# Facts about FEBio XML 2.5

BC_TYPE_TAG = {
    "node": {"variable": "prescribe", "fixed": "fix"},
    "body": {"variable": "prescribed", "fixed": "fixed"},
}


# Functions for reading XML


def elem_var_fiber_xml(e):
    tag = ET.Element("elem")
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


element_var_feb = {
    "v_fiber": {"name": "fiber", "fn": elem_var_fiber_xml},
    "thickness": {"name": "shell thickness", "fn": elem_var_thickness_xml},
    "von Mises": {"name": "MRVonMisesParameters", "fn": elem_var_vonmises_xml},
    "prestretch": {"name": "pre_stretch", "fn": elem_var_prestretch_xml},
}


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
            info = {
                "node set name": None,
                "axis": None,  # x1, fluid, charge, etc.
                "variable": None,  # displacement, force, pressure, etc.
                "sequence ID": None,
                "scale": 0.0,  # FEBio default; it really should be 1.0
                "relative": False,
                "nodal values": None,
                "step ID": None,
            }
            # Read values
            info["node set name"] = e_prescribe.attrib["node_set"]
            info["dof"] = DOF_NAME_FROM_XML_NODE_BC[e_prescribe.attrib["bc"]]
            info["variable"] = VAR_FROM_XML_NODE_BC[e_prescribe.attrib["bc"]]
            e_scale = e_prescribe.find("scale")
            if e_scale.text is not None:
                info["scale"] = to_number(e_scale.text)
            info["sequence ID"] = to_number(e_scale.attrib["lc"]) - 1
            # Node-specific values
            e_value = e_prescribe.find("value")
            if e_value is not None:
                if "node_data" in e_value.attrib:
                    # Node-specific data; look up the data in MeshData
                    e_NodeSet = find_unique_tag(
                        root, "Geometry/NodeSet[@name='" + info["node set name"] + "']"
                    )
                    e_NodeData = find_unique_tag(
                        root,
                        "MeshData/NodeData[@name='"
                        + e_value.attrib["node_data"]
                        + "']",
                    )
                    info["nodal values"] = {}
                    for e_node, e_value in zip(
                        e_NodeSet.findall("node"), e_NodeData.findall("node")
                    ):
                        id_ = int(e_node.attrib["id"]) - 1
                        info["nodal values"][id_] = to_number(e_value.text)
                else:
                    # One value for all nodes; redundant with "scale"
                    val_scale = to_number(e_value.text)
                    info["scale"] = seq_scale * val_scale
            e_relative = e_prescribe.find("relative")
            if e_relative is not None:
                info["relative"] = True
            info["step ID"] = step_id
            yield info


# Functions for writing XML


def geometry_section(model, parts, material_registry):
    """Return XML tree for Geometry section."""
    e_geometry = ET.Element("Geometry")
    # Write <nodes>
    e_nodes = ET.SubElement(e_geometry, "Nodes")
    for i, x in enumerate(model.mesh.nodes):
        feb_nid = i + 1  # 1-indexed
        e = ET.SubElement(e_nodes, "node", id="{}".format(feb_nid))
        e.text = vec_to_text(x)
        e_nodes.append(e)
    # Write <elements> for each part
    for part in parts:
        e_elements = ET.SubElement(e_geometry, "Elements")
        e_elements.attrib["type"] = part["element_type"].feb_name
        mat_id = material_registry.names(part["material"], "ordinal_id")[0]
        e_elements.attrib["mat"] = str(mat_id + 1)
        for i, e in part["elements"]:
            e_element = ET.SubElement(e_elements, "elem")
            e_element.attrib["id"] = str(i + 1)
            e_element.text = ", ".join(str(i + 1) for i in e.ids)
    return e_geometry


def meshdata_section(model):
    """Return XML tree for some of MeshData section

    Currently this function only generates the part of the MesData
    section that deals with material axis element data.

    """
    e_meshdata = ET.Element("MeshData")
    e_edata_mat_axis = ET.Element(
        "ElementData", var="mat_axis", elem_set="autogen-mat_axis"
    )
    e_elemset = ET.Element("ElementSet", name="autogen-mat_axis")
    i_elemset = 0
    # ^ index into the extra element set we're forced to construct
    for i, e in enumerate(model.mesh.elements):
        # Write local basis if defined
        if e.basis is not None:
            e_elem = ET.SubElement(e_edata_mat_axis, "elem", lid=str(i_elemset + 1))
            e_elem.append(ET.Comment(f"Element {i + 1}"))
            i_elemset += 1
            ET.SubElement(e_elem, "a").text = bvec_to_text(e.basis[:, 0])
            ET.SubElement(e_elem, "d").text = bvec_to_text(e.basis[:, 1])
            ET.SubElement(e_elemset, "elem", id=str(i + 1))
    if len(e_edata_mat_axis) != 0:
        e_meshdata.append(e_edata_mat_axis)
    return e_meshdata, e_elemset


def split_bc_attrib(s):
    """Split boundary condition names.

    In FEBio XML 2.5, each boundary condition is separated by a comma.

    """
    return [bc.strip() for bc in s.split(",")]


def node_data_xml(nodes, data, data_name, nodeset_name):
    """Construct NodeData XML element"""
    e_NodeData = ET.Element("NodeData")
    e_NodeData.attrib["name"] = data_name
    e_NodeData.attrib["node_set"] = nodeset_name
    # Write NodeData/node elements.  To specify a node, FEBio XML, for
    # some reason, uses the 1-indexed position of the node in the node
    # set as a "local ID", as opposed to just using the node's ID.  Our
    # node sets, being sets, are unordered.  To be able to generate the
    # local ID, we write NodeSet/node elements in FEBio XML in ascending
    # order of node ID.
    lid_from_node_id = {node_id: i + 1 for i, node_id in enumerate(sorted(nodes))}
    for i, v in zip(nodes, data):
        ET.SubElement(
            e_NodeData,
            "node",
            lid=str(lid_from_node_id[i]),
        ).text = float_to_text(v)
    return e_NodeData


def node_var_disp_xml(
    model, xmlroot, nodes, scales, seq, dof, var, relative, step_id=0
):
    """Return XML elements for nodal variable displacement

    model := Model object.  Needed for the name registry.

    Returns tuple of (<bc> element, <NodeData> element)

    """
    # Hierarchy: <Boundary><prescribe node_set="set_name">
    e_bc = ET.Element(BC_TYPE_TAG["node"]["variable"], bc=XML_BC_FROM_DOF[(dof, var)])
    seq_id = get_or_create_seq_id(model.named["sequences"], seq)
    e_sc = ET.SubElement(e_bc, "scale", lc=str(seq_id + 1))
    e_sc.text = "1.0"
    # Get or create a name for the node set
    nm_base = "nodal_bc_" f"step{step_id + 1}_variable_{var[0]}_seq{seq_id}_autogen"
    nodeset = NodeSet(nodes)
    nodeset_name = model.named["node sets"].get_or_create_name(nm_base, nodeset)
    e_bc.attrib["node_set"] = nodeset_name
    # Generate a non-duplicate name for the Geometry/MeshData/NodeData
    # element, which will contain the node-specific scaling factors.
    stem = "nodal_bc_" f"step{step_id + 1}_{dof}_seq{seq_id}_autogen"
    i = 0
    data_name = f"{stem}{i}"
    e_MeshData = find_unique_tag(xmlroot, "MeshData")
    while e_MeshData.find(f"NodeData[@name='{data_name}']") is not None:
        i += 1
        data_name = f"{stem}{i}"
    # Create the <NodeData> element
    e_NodeData = node_data_xml(nodes, scales, data_name, nodeset_name)
    # Reference the node-specific boundary condition scaling factors
    ET.SubElement(e_bc, "value", node_data=data_name)
    # Other attributes
    ET.SubElement(e_bc, "relative").text = str(int(relative))
    return e_bc, e_NodeData
