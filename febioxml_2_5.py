# Base packages
from collections import defaultdict
from typing import Dict

# System packages
from lxml import etree as ET

# Same-package modules
from .core import (
    ZeroIdxID,
    OneIdxID,
    Sequence,
    Interpolant,
    Extrapolant,
    Body,
    ImplicitBody,
)
from .febioxml import *


# Facts about FEBio XML 2.5

# XML element parents and names
BODY_COND_PARENT = "Boundary"
MESH_PARENT = "Geometry"
ELEMENTDATA_PARENT = "MeshData"
NODEDATA_PARENT = "MeshData"
ELEMENTSET_PARENT = "Geometry"
STEP_PARENT = "."
STEP_NAME = "Step"

BC_TYPE_TAG = {
    "node": {"variable": "prescribe", "fixed": "fix"},
    "body": {"variable": "prescribed", "fixed": "fixed"},
}

# Map of Ticker fields → elements relative to <Step>
TICKER_PARAMS = {
    "n": ReqParameter("Control/time_steps"),
    "dtnom": ReqParameter("Control/step_size"),
    "dtmin": ReqParameter("Control/time_stepper/dtmin"),
    "dtmax": ReqParameter("Control/time_stepper/dtmax"),
}
# Map of Controller fields → elements relative to <Step>
CONTROLLER_PARAMS = {
    "max_retries": OptParameter("Control/time_stepper/max_retries", 5),
    "opt_iter": OptParameter("Control/time_stepper/opt_iter", 10),
    "save_iters": OptParameter("Control/plot_level", "PLOT_MAJOR_ITRS"),
}
# Map of Solver fields → elements relative to <Step>
SOLVER_PARAMS = {
    "dtol": OptParameter("Control/dtol", 0.001),
    "etol": OptParameter("Control/etol", 0.01),
    "rtol": OptParameter("Control/rtol", 0),
    "lstol": OptParameter("Control/lstol", 0.9),
    "ptol": OptParameter("Control/ptol", 0.01),
    "min_residual": OptParameter("Control/min_residual", 1e-20),
    "update_method": OptParameter("Control/qnmethod", "BFGS"),
    "reform_each_time_step": OptParameter("Control/reform_each_time_step", True),
    "reform_on_diverge": OptParameter("Control/diverge_reform", True),
    "max_refs": OptParameter("Control/max_refs", 15),
    "max_ups": OptParameter("Control/max_ups", 10),
}


# Functions for reading FEBio XML 2.5


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
    not-None except the following:

    (1) "nodal values" will be None if the condition applies the same condition to all nodes.

    (2) "name" is always None because it was introduced in FEBio XML
    3.0.

    (3) "scale" will be None if the condition is heterogeneous, as FEBio
    XML 3.0 does not include a scale in this case.

    """
    step_id = -1  # Curent step ID (0-indexed)
    for e_Step in root.findall(f"{STEP_PARENT}/{STEP_NAME}"):
        step_id += 1
        for e_prescribe in e_Step.findall(
            f"Boundary/{BC_TYPE_TAG['node']['variable']}"
        ):
            # Re-initialize output
            info = {
                "name": None,
                "node set name": None,
                "axis": None,  # x1, fluid, charge, etc.
                "variable": None,  # displacement, force, pressure, etc.
                "sequence ID": None,
                "scale": None,  # For consistency with FEBio XML 3.0
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
                    # Heterogeneous nodal boundary condition
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
                    # Homogeneous nodal boundary condition
                    seq_scale = info["scale"]
                    val_scale = to_number(e_value.text)
                    info["scale"] = seq_scale * val_scale
            e_relative = e_prescribe.find("relative")
            if e_relative is not None:
                info["relative"] = True
            info["step ID"] = step_id
            yield info


def read_domains(root: ET.Element):
    """Return list of domains"""
    domains = []
    e_domains = root.findall(f"{MESH_PARENT}/Elements")
    for e_domain in e_domains:
        name = e_domain.attrib.get("name", None)
        elements = [
            ZeroIdxID(int(e.attrib["id"]) - 1) for e in e_domain.findall("elem")
        ]
        domain = {
            "name": name,
            "material": ("ordinal_id", ZeroIdxID(int(e_domain.attrib["mat"]) - 1)),
            "elements": elements,
        }
        domains.append(domain)
    return domains


def sequences(root: Element) -> Dict[int, Sequence]:
    """Return dictionary of sequence ID → sequence from FEBio XML 2.5"""
    sequences = {}
    for ord_id, e_lc in enumerate(root.findall("LoadData/loadcurve")):
        fake_id = int(e_lc.attrib["id"])
        curve = [read_point(a.text) for a in e_lc.getchildren()]
        # Set extrapolation
        if "extend" in e_lc.attrib:
            extrap = EXTRAP_FROM_XML_EXTRAP[e_lc.attrib["extend"]]
            if extrap == "extrapolate":
                extrap = Extrapolant.LINEAR
        else:
            extrap = Extrapolant.CONSTANT  # FEBio's default
        # Set interpolation
        if "type" in e_lc.attrib:
            interp = INTERP_FROM_XML_INTERP[e_lc.attrib["type"]]
        else:
            interp = Interpolant.LINEAR  # FEBio's default
        # Create and store the Sequence object
        sequences[ord_id] = Sequence(curve, interp=interp, extrap=extrap)
    return sequences


def read_rigid_body_bc(model, e_rigid_body, explicit_bodies, implicit_bodies, step):
    """Read & apply a <rigid_body> element

    model := Model object.

    e_rigid_body := The <rigid_body> XML element.

    explicit_bodies := map of material ID → body.

    implicit_bodies := map of material ID → body.

    step := The step to which the rigid body boundary condition belongs.

    """
    # Each <rigid_body> element defines constraints for one rigid body,
    # identified by its material ID.  Constraints may be fixed
    # (atemporal) or time-varying (temporal).
    #
    # Get the Body object from the material id
    mat_id = int(e_rigid_body.attrib["mat"]) - 1
    if mat_id in explicit_bodies:
        body = explicit_bodies[mat_id]
    else:
        # Assume mat_id refers to an implicit rigid body
        body = implicit_bodies[mat_id]
    # Read the body's constraints
    for e_dof in e_rigid_body.findall("fixed"):
        dof = DOF_NAME_FROM_XML_NODE_BC[e_dof.attrib["bc"]]
        var = VAR_FROM_XML_NODE_BC[e_dof.attrib["bc"]]
        model.fixed["body"][(dof, var)].add(body)
    for e_dof in e_rigid_body.findall("prescribed"):
        dof = DOF_NAME_FROM_XML_NODE_BC[e_dof.attrib["bc"]]
        var = VAR_FROM_XML_NODE_BC[e_dof.attrib["bc"]]
        seq = read_parameter(e_dof, model.named["sequences"])
        if e_dof.get("type", None) == "relative":
            relative = True
        else:
            relative = False
        model.apply_body_bc(body, dof, var, seq, step, relative=relative, step=step)


# Functions for writing XML 2.5


def body_constraints_xml(
    body, constraints: dict, material_registry, implicit_rb_mats, sequence_registry
):
    """Return <rigid_body> element for a body's constraints.

    The constrained variable can be displacement or rotation.

    """
    mat_id = body_mat_id(body, material_registry, implicit_rb_mats)
    e_rb_bc = ET.Element("rigid_body", mat=str(mat_id + 1))
    for dof, bc in constraints.items():
        if bc["sequence"] == "fixed":
            kind = "fixed"
        else:  # bc['sequence'] is Sequence
            kind = "variable"
            seq = bc["sequence"]
            v = bc["scale"]
        # Determine which tag name to use for the specified
        # variable: force or displacement
        if bc["variable"] in ["displacement", "rotation"]:
            tagname = BC_TYPE_TAG["body"][kind]
        elif bc["variable"] == "force":
            tagname = "force"
            if bc["relative"]:
                raise ValueError(
                    f"A relative body boundary condition for {dof} {bc['variable']} was requested, but relative body boundary conditions are supported only for displacement and rotation."
                )
        else:
            raise ValueError(f"Variable {bc['variable']} not supported for BCs.")
        bc_attr = XML_BC_FROM_DOF[(dof, bc["variable"])]
        e_bc = ET.SubElement(e_rb_bc, tagname, bc=bc_attr)
        if kind == "variable":
            seq_id = get_or_create_seq_id(sequence_registry, seq)
            e_bc.attrib["lc"] = str(seq_id + 1)
            if bc["relative"]:
                e_bc.attrib["type"] = "relative"
            e_bc.text = str(v)
    return [e_rb_bc]


def mesh_xml(model, domains, material_registry):
    """Create <Geometry> XML element.

    Returns a tuple because the FEBio XML 3.0 version needs to return
    two XML elements.

    """
    e_geometry = ET.Element(MESH_PARENT)
    # Write <nodes>
    e_nodes = ET.SubElement(e_geometry, "Nodes")
    for i, x in enumerate(model.mesh.nodes):
        feb_nid = i + 1  # 1-indexed
        e = ET.SubElement(e_nodes, "node", id="{}".format(feb_nid))
        e.text = vec_to_text(x)
        e_nodes.append(e)
    # Write <Elements> for each domain
    for i, domain in enumerate(domains):
        if domain["material"] is None:
            raise ValueError("Some elements have no material assigned.")
        e_elements = ET.SubElement(e_geometry, "Elements", name=f"Domain{i+1}")
        e_elements.attrib["type"] = domain["element_type"].feb_name
        mat_id = material_registry.names(domain["material"], "ordinal_id")[0]
        e_elements.attrib["mat"] = str(mat_id + 1)
        for i, e in domain["elements"]:
            e_element = ET.SubElement(e_elements, "elem")
            e_element.attrib["id"] = str(i + 1)
            e_element.text = ", ".join(str(i + 1) for i in e.ids)
    return (e_geometry,)


def meshdata_xml(model):
    """Return <ElementData> and <ElementSet> XML elements

    Currently this function only generates the part of the MeshData
    section that deals with material axis element data.

    """
    e_meshdata = []
    e_elemsets = []
    e_edata_mat_axis = ET.Element(
        "ElementData", var="mat_axis", elem_set="autogen-mat_axis"
    )
    e_elemset_mat_axis = ET.Element("ElementSet", name="autogen-mat_axis")
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
            ET.SubElement(e_elemset_mat_axis, "elem", id=str(i + 1))
    if len(e_edata_mat_axis) != 0:
        e_meshdata.append(e_edata_mat_axis)
        e_elemsets.append(e_elemset_mat_axis)
    return e_meshdata, e_elemsets


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


def node_fix_disp_xml(fixed_conditions, nodeset_registry):
    """Return XML elements for node fixed displacement conditions.

    fixed_conditions := The data structure in model.fixed["node"]

    This function may create and add new nodesets to the nodeset name
    registry.  If generating a full XML tree, be sure to write these new
    nodesets to the tree.

    """
    # Tag hierarchy: <Boundary><fix bc="x" node_set="set_name">
    e_bcs = []
    for (dof, var), nodeset in fixed_conditions.items():
        if not nodeset:
            continue
        nodeset = NodeSet(nodeset)  # make hashable
        base = f"fixed_{dof}_autogen-nodeset"
        name = nodeset_registry.get_or_create_name(base, nodeset)
        # Create the tag
        e_bc = ET.Element(
            BC_TYPE_TAG["node"]["fixed"], bc=XML_BC_FROM_DOF[(dof, var)], node_set=name
        )
        e_bcs.append(e_bc)
    return e_bcs


def node_var_disp_xml(
    model, xmlroot, nodes, scales, seq, dof, var, relative, step_name
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
    nm_base = "nodal_bc_" f"step={step_name}_var={var[0]}_seq={seq_id}_autogen"
    nodeset = NodeSet(nodes)
    nodeset_name = model.named["node sets"].get_or_create_name(nm_base, nodeset)
    e_bc.attrib["node_set"] = nodeset_name
    # Generate a non-duplicate name for the Geometry/MeshData/NodeData
    # element, which will contain the node-specific scaling factors.
    stem = "nodal_bc_" f"step={step_name}_{dof}_seq={seq_id}_autogen"
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


def sequence_xml(sequence: Sequence, sequence_id: int, t0=0.0):
    """Return a <load_curve> XML element for a sequence.

    sequence := Sequence object.

    sequence_id := Integer ID (origin = 0) to use for the sequence's XML
    element "id" attribute.  The ID will be incremented by 1 to account
    for FEBio XML's use of 1-referenced IDs.

    t0 := Time offset to apply to the sequence's time points before
    writing them to XML.  The intended use for this is to translate from global to

    """
    e_loadcurve = ET.Element(
        "loadcurve",
        id=str(sequence_id + 1),
        type=XML_INTERP_FROM_INTERP[sequence.interpolant],
        extend=XML_EXTRAP_FROM_EXTRAP[sequence.extrapolant],
    )
    for pt in sequence.points:
        ET.SubElement(e_loadcurve, "point").text = f"{pt[0] + t0}, {pt[1]}"
    return e_loadcurve


def surface_pair_xml(faceset_registry, primary, secondary, name):
    """Return SurfacePair XML element."""
    e_surfpair = ET.Element("SurfacePair", name=name)
    ET.SubElement(
        e_surfpair,
        "master",
        surface=faceset_registry.names(primary)[0],
    )
    ET.SubElement(
        e_surfpair,
        "slave",
        surface=faceset_registry.names(secondary)[0],
    )
    return e_surfpair
