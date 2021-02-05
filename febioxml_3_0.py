# Same-package modules
from .febioxml import *
from .febioxml_2_5 import geometry_section, meshdata_section

# Facts about FEBio XML 3.0

BC_TYPE_TAG = {
    "node": {"variable": "prescribe", "fixed": "fix"},
    "body": {"variable": "prescribe", "fixed": "fix"},
}


# Functions for reading XML

# Functions for writing XML


def node_data_xml(nodes, data, data_name, nodeset_name):
    """Construct NodeData XML element"""
    e_NodeData = ET.Element("NodeData")
    e_NodeData.attrib["data_type"] = "scalar"
    # TODO: support other data types
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
    # Hierarchy: <Boundary><bc type="prescribe" node_set="set_name">
    e_bc = ET.Element("bc", type=BC_TYPE_TAG["node"]["variable"])
    e_dof = ET.SubElement(e_bc, "dof").text = XML_BC_FROM_DOF[(dof, var)]
    seq_id = get_or_create_seq_id(model.named["sequences"], seq)
    e_sc = ET.SubElement(e_bc, "scale", lc=str(seq_id + 1), type="map")
    # Other subelements
    ET.SubElement(e_bc, "relative").text = str(int(relative))
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
    e_sc.text = data_name
    return e_bc, e_NodeData
