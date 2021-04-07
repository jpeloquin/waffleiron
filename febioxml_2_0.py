# Base packages
from collections import defaultdict

# System packages
from lxml import etree as ET

# Same-package modules
from .core import ContactConstraint, Interpolant, Extrapolant, Sequence
from .output import material_to_feb
from .febioxml import *
from .febioxml_2_5 import mesh_xml, sequences, read_domains


# Facts about FEBio XML 2.0

BODY_COND_PARENT = "Constraints"
MESH_PARENT = "Geometry"
ELEMENTDATA_PARENT = "Geometry"
NODEDATA_PARENT = "Geometry"
ELEMENTSET_PARENT = "Geometry"
STEP_PARENT = "."
STEP_NAME = "Step"

BC_TYPE_TAG = {
    "node": {"variable": "prescribe", "fixed": "fix"},
    "body": {"variable": "prescribed", "fixed": "fixed"},
}

XML_INTERP_FROM_INTERP = {
    Interpolant.STEP: "step",
    Interpolant.LINEAR: "linear",
    Interpolant.SPLINE: "smooth",
}
INTERP_FROM_XML_INTERP = {v: k for k, v in XML_INTERP_FROM_INTERP.items()}


XML_EXTRAP_FROM_EXTRAP = {
    Extrapolant.CONSTANT: "constant",
    Extrapolant.LINEAR: "extrapolate",
    Extrapolant.REPEAT: "repeat",
    Extrapolant.REPEAT_CONTINUOUS: "repeat offset",
}
EXTRAP_FROM_XML_EXTRAP = {v: k for k, v in XML_EXTRAP_FROM_EXTRAP.items()}

# Map of Ticker fields → elements relative to <Step>
TICKER_PARAMS = {
    "n": ReqParameter("Control/time_steps"),
    "dtnom": ReqParameter("Control/step_size"),
    "dtmin": OptParameter("Control/time_stepper/dtmin", 0),  # undocumented default
    "dtmax": OptParameter("Control/time_stepper/dtmax", 0.05),  # undocumented default
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


# Functions for reading FEBio XML 2.0


def iter_node_conditions(root):
    """Return generator over prescribed nodal condition info.

    Returns dict of property names → values.  All properties are
    guaranteed to be not-None, except "nodal values", which will be None
    if the condition applies the same condition to all nodes.

    All returned IDs are 0-indexed for consistency with febtools.

    """
    step_id = -1  # Curent step ID (0-indexed)
    for e_Step in root.findall(f"{STEP_PARENT}/{STEP_NAME}"):
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
            # Step ID
            info["step ID"] = step_id
            # Read values
            info["dof"] = DOF_NAME_FROM_XML_NODE_BC[e_prescribe.attrib["bc"]]
            info["variable"] = VAR_FROM_XML_NODE_BC[e_prescribe.attrib["bc"]]
            # Node set
            if "set" in e_prescribe.attrib:
                info["node set name"] = e_prescribe.attrib["set"]
            # Scale
            if "scale" in e_prescribe.attrib:
                info["scale"] = to_number(e_prescribe.attrib["scale"])
            # Relative
            if "type" in e_prescribe.attrib:
                if e_prescribe.attrib["type"] == "relative":
                    info["relative"] = True
            else:
                info["relative"] = False
            # Time series
            info["sequence ID"] = to_number(e_prescribe.attrib["lc"]) - 1
            # Node-specific values
            node_elements = e_prescribe.findall("node")
            if len(node_elements) != 0:
                info["nodal values"] = {}
            for e in node_elements:
                id_ = int(e.attrib["id"])
                info["nodal values"][id_] = to_number(e.text)
            yield info


# Functions for writing FEBio XML 2.0


def contact_section(model):
    tag_branch = ET.Element("Contact")
    contact_constraints = [
        constraint
        for constraint in model.constraints
        if type(constraint) is ContactConstraint
    ]
    for contact in contact_constraints:
        tag_contact = ET.SubElement(tag_branch, "contact", type=contact.algorithm)
        # Set compression only or tension–compression
        if contact.algorithm == "sliding-elastic":
            ET.SubElement(tag_contact, "tension").text = str(int(contact.tension))
        else:
            if contact.tension:
                raise ValueError(
                    f"Only the sliding–elastic contact algorithm is known to support tension–compression contact in FEBio."
                )
        # Write penalty-related tags
        ET.SubElement(tag_contact, "auto_penalty").text = (
            "1" if contact.penalty["type"] == "auto" else "0"
        )
        ET.SubElement(tag_contact, "penalty").text = f"{contact.penalty['factor']}"
        # Write algorithm modification tags
        ET.SubElement(tag_contact, "laugon").text = (
            "1" if contact.augmented_lagrange else "0"
        )
        # (two_pass would go here)
        # Write surfaces
        e_master = ET.SubElement(tag_contact, "surface", type="master")
        for f in contact.leader:
            e_master.append(tag_face(f))
        e_follower = ET.SubElement(tag_contact, "surface", type="slave")
        for f in contact.follower:
            e_follower.append(tag_face(f))
    return tag_branch


def meshdata_xml(model):
    """Return <ElementData> and <ElementSet> XML element(s)

    Technically this function should also return <ElementSet> XML
    elements but in FEBio XML 2.0 there is little reason to do this.

    """
    e_elementdata = ET.Element("ElementData")
    e_meshdata = [e_elementdata]
    for i, element in enumerate(model.mesh.elements):
        # Write any defined element data
        if ("thickness" in element.properties) or ("v_fiber" in element.properties):
            e_element = ET.SubElement(e_elementdata, "element", id=str(i + 1))
        if "thickness" in element.properties:
            ET.SubElement(e_element, "thickness").text = ",".join(
                str(t) for t in element.properties["thickness"]
            )
        if "v_fiber" in element.properties:
            ET.SubElement(e_edata, "fiber").text = ",".join(
                str(a) for a in element.properties["v_fiber"]
            )
    e_elemsets = tuple()  # For compatibility with other XML versions
    return e_meshdata, e_elemsets


def node_fix_disp_xml(fixed_conditions, nodeset_registry):
    """Return XML elements for node fixed displacement conditions.

    fixed_conditions := The data structure in model.fixed["node"]

    This function may create and add new nodesets to the nodeset name
    registry.  If generating a full XML tree, be sure to write these new
    nodesets to the tree.

    """
    e_bcs = []
    # Tag hierarchy: <Boundary><fix bc="x"><node id="1"> for each node
    for (dof, var), nodeset in fixed_conditions.items():
        if not nodeset:
            continue
        e_bc = ET.Element(
            e_boundary, BC_TYPE_TAG["node"]["fixed"], bc=XML_BC_FROM_DOF[(dof, var)]
        )
        for i in nodeset:
            ET.SubElement(e_bc, "node", id=str(i + 1))
        e_bcs.append(e_bc)
    return e_bcs


def tag_face(face):
    nm = {3: "tri3", 4: "quad4"}
    tag = ET.Element(nm[len(face)])
    tag.text = ", ".join([f"{i+1}" for i in face])
    return tag


def split_bc_attrib(s):
    """Split boundary condition names.

    In FEBio XML 2.0, each BC is one character.

    """
    return [c for c in s]


def nodal_var_disp_xml(model, nodes, scales, scale, dof, var):
    e_bc = ET.Element(
        "prescribe",
        bc=XML_BC_FROM_DOF[(dof, var)],
    )
    e_bc.attrib["lc"] = str(seq_id + 1)
    # Write nodes as children of <Step><Boundary><prescribe>
    for i, sc in zip(nodes, scales):
        ET.SubElement(e_bc, "node", id=str(i + 1)).text = float_to_text(sc)
    return e_bc, e_nodedata
