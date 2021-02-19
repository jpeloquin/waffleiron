# Base packages
from collections import defaultdict

# System packages
from lxml import etree as ET

# Same-package modules
from .core import ContactConstraint, Interpolant, Extrapolant, Sequence
from .output import material_to_feb
from .control import step_duration
from .febioxml import *


# Facts about FEBio XML 2.0

BODY_COND_PARENT = "Constraints"

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
