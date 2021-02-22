from collections import defaultdict

# Same-package modules
from .core import Body, ImplicitBody, Extrapolant, Interpolant
from .febioxml import *

# These parts work the same as in FEBio XML 2.5
from .febioxml_2_5 import meshdata_xml

# Facts about FEBio XML 3.0

BC_TYPE_TAG = {
    "node": {"variable": "prescribe", "fixed": "fix"},
    "body": {"variable": "prescribe", "fixed": "fix"},
}

# XML element parents and names
BODY_COND_PARENT = "Rigid"
MESH_PARENT = "Mesh"
ELEMENTDATA_PARENT = "MeshData"
NODEDATA_PARENT = "MeshData"
ELEMENTSET_PARENT = "Mesh"
STEP_PARENT = "Step"
STEP_NAME = "step"

BC_TYPE_TAG = {
    "node": {"variable": "prescribe", "fixed": "fix"},
    "body": {"variable": "prescribe", "fixed": "fix"},
}

XML_RB_DOF_FROM_DOF = {
    "x1": "Rx",
    "x2": "Ry",
    "x3": "Rz",
    "α1": "Ru",
    "α2": "Rv",
    "α3": "Rw",
}

XML_INTERP_FROM_INTERP = {
    Interpolant.STEP: "STEP",
    Interpolant.LINEAR: "LINEAR",
    Interpolant.SPLINE: "SMOOTH",
}
INTERP_FROM_XML_INTERP = {v: k for k, v in XML_INTERP_FROM_INTERP.items()}

XML_EXTRAP_FROM_EXTRAP = {
    Extrapolant.CONSTANT: "CONSTANT",
    Extrapolant.LINEAR: "EXTRAPOLATE",
    Extrapolant.REPEAT: "REPEAT",
    Extrapolant.REPEAT_CONTINUOUS: "REPEAT OFFSET",
}
EXTRAP_FROM_XML_EXTRAP = {v: k for k, v in XML_EXTRAP_FROM_EXTRAP.items()}


# Functions for reading FEBio XML 3.0


# Functions for writing FEBio XML 3.0


def body_constraints_xml(
    body, constraints: dict, material_registry, implicit_rb_mats, sequence_registry
):
    """Return <rigid_constraint> element(s) for body's constraints.

    The constrained variable can be displacement or rotation.

    """
    elems = []
    mat_id = body_mat_id(body, material_registry, implicit_rb_mats)
    # Can't put fixed and variable constraints in the same
    # <rigid_constraint> element, so first we have to group the
    # constraints by kind
    fixed_constraints = []
    variable_constraints = []
    for dof, bc in constraints.items():
        if bc["sequence"] == "fixed":
            fixed_constraints.append((dof, bc))
        else:  # bc['sequence'] is Sequence
            variable_constraints.append((dof, bc))
    # Create <rigid_constraint> element for fixed constraints
    e_rb_fixed = ET.Element("rigid_constraint")
    e_rb_fixed.attrib["type"] = BC_TYPE_TAG["body"]["fixed"]
    ET.SubElement(e_rb_fixed, "rb").text = str(mat_id + 1)
    ET.SubElement(e_rb_fixed, "dofs").text = ",".join(
        XML_RB_DOF_FROM_DOF[dof] for dof, _ in fixed_constraints
    )
    elems.append(e_rb_fixed)
    # Create <rigid_constraint> element for variable constraints.  I
    # think you must use a separate element for each degree of freedom
    # (x, y, z, Rx, Ry, Rz).
    for dof, bc in variable_constraints:
        if bc["variable"] == "force":
            # TODO: Reverse engineer force constraints on rigid bodies
            # in FEBio XML 3.0
            raise NotImplementedError
        e_rb = ET.Element("rigid_constraint")
        e_rb.attrib["type"] = BC_TYPE_TAG["body"]["variable"]
        ET.SubElement(e_rb, "rb").text = str(mat_id + 1)
        ET.SubElement(e_rb, "dof").text = XML_RB_DOF_FROM_DOF[dof]
        seq = bc["sequence"]
        seq_id = get_or_create_seq_id(sequence_registry, seq)
        e_value = ET.SubElement(e_rb, "value")
        e_value.attrib["lc"] = str(seq_id + 1)
        e_value.text = str(bc["scale"])
        ET.SubElement(e_rb, "relative").text = bool_to_text(bc["relative"])
        elems.append(e_rb)
    return elems


def mesh_xml(model, domains, material_registry):
    """Create <Mesh> and <MeshDomain> XML elements.

    <Geometry> became <Mesh> in FEBio XML 3.0

    """
    e_geometry = ET.Element(MESH_PARENT)
    e_meshdomains = ET.Element("MeshDomains")
    # Write <nodes>
    e_nodes = ET.SubElement(e_geometry, "Nodes")
    for i, x in enumerate(model.mesh.nodes):
        feb_nid = i + 1  # 1-indexed
        e = ET.SubElement(e_nodes, "node", id="{}".format(feb_nid))
        e.text = vec_to_text(x)
        e_nodes.append(e)
    # Write <Elements> and <SolidDomain> for each domain
    for i, domain in enumerate(domains):
        # <Elements>
        e_elements = ET.SubElement(e_geometry, "Elements", name=domain["name"])
        e_elements.attrib["type"] = domain["element_type"].feb_name
        for i, e in domain["elements"]:
            e_element = ET.SubElement(e_elements, "elem")
            e_element.attrib["id"] = str(i + 1)
            e_element.text = ", ".join(str(i + 1) for i in e.ids)
        # <SolidDomain>
        mat_name = material_registry.names(domain["material"], "canonical")[0]
        e_soliddomain = ET.SubElement(
            e_meshdomains, "SolidDomain", name=domain["name"], mat=mat_name
        )
    return e_geometry, e_meshdomains


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


def node_fix_disp_xml(fixed_conditions, nodeset_registry):
    """Return XML elements for node fixed displacement conditions.

    fixed_conditions := The data structure in model.fixed["node"]

    This function may create and add new nodesets to the nodeset name
    registry.  If generating a full XML tree, be sure to write these new
    nodesets to the tree.

    """
    # Tag hierarchy: <Boundary><bc type="fix" node_set="set_name">
    e_bcs = []
    # FEBio XML 3.0 stores the nodal BCs by node set, so we need to do
    # some collation first.
    by_nodeset = defaultdict(list)
    for dofvar, nodeset in fixed_conditions.items():
        if not nodeset:
            continue
        nodeset = NodeSet(nodeset)
        by_nodeset[nodeset].append(dofvar)
    for nodeset, dofvar_pairs in by_nodeset.items():
        # Get or create a name for the node set
        base = f"fixed_nodes_{','.join(dof for dof, var in dofvar_pairs)}_auto"
        nodeset_name = nodeset_registry.get_or_create_name(base, nodeset)
        # Create element
        e_bc = ET.Element(
            "bc", type=BC_TYPE_TAG["node"]["fixed"], node_set=nodeset_name
        )
        txt = ",".join(XML_BC_FROM_DOF[t] for t in dofvar_pairs)
        e_dofs = ET.SubElement(e_bc, "dofs").text = txt
        e_bcs.append(e_bc)
    return e_bcs


def node_var_disp_xml(
    model, xmlroot, nodes, scales, seq, dof, var, relative, step_name
):
    """Return XML elements for nodes variable displacement

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
    nm_base = "nodal_bc_" f"step={step_name}_var={var[0]}_seq={seq_id}_autogen"
    nodeset = NodeSet(nodes)
    nodeset_name = model.named["node sets"].get_or_create_name(nm_base, nodeset)
    e_bc.attrib["node_set"] = nodeset_name
    # Generate a non-duplicate name for the Geometry/MeshData/NodeData
    # element, which will contain the node-specific scaling factors.
    stem = "nodal_bc_" f"step={step_name}_{dof}_seq{seq_id}_autogen"
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


def sequence_xml(sequence: Sequence, sequence_id: int, t0=0.0):
    """Return a <load_controller> XML element for a sequence.

    sequence := Sequence object.

    sequence_id := Integer ID (origin = 0) to use for the sequence's XML
    element "id" attribute.  The ID will be incremented by 1 to account
    for FEBio XML's use of 1-referenced IDs.

    t0 := Time offset to apply to the sequence's time points before
    writing them to XML.  The intended use for this is to translate from
    step-local to global simulation time.

    """
    e_loadcurve = ET.Element(
        "load_controller", id=str(sequence_id + 1), type="loadcurve"
    )
    ET.SubElement(e_loadcurve, "interpolate").text = XML_INTERP_FROM_INTERP[
        sequence.interpolant
    ]
    ET.SubElement(e_loadcurve, "extend").text = XML_EXTRAP_FROM_EXTRAP[
        sequence.extrapolant
    ]
    e_points = ET.SubElement(e_loadcurve, "points")
    for pt in sequence.points:
        ET.SubElement(e_points, "point").text = f"{pt[0] + t0}, {pt[1]}"
    return e_loadcurve


def surface_pair_xml(faceset_registry, primary, secondary, name):
    """Return SurfacePair XML element."""
    e_surfpair = ET.Element("SurfacePair", name=name)
    ET.SubElement(e_surfpair, "primary").text = faceset_registry.names(primary)[0]
    ET.SubElement(e_surfpair, "secondary").text = faceset_registry.names(secondary)[0]
    return e_surfpair


def step_xml_factory():
    """Create empty <step> XML elements"""
    i = 1
    while True:
        e = ET.Element(STEP_NAME, id=str(i), name=f"Step{i}")
        yield e
