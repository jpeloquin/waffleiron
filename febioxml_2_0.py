# Base packages
from collections import defaultdict

# System packages
from lxml import etree as ET

# Same-package modules
from .core import (
    ContactConstraint,
    Interpolant,
    Extrapolant,
    NodeSet,
    Sequence,
    ZeroIdxID,
)
from .output import material_to_feb
from .control import Dynamics, SaveIters, Solver
from .febioxml import *
from .febioxml_2_5 import (
    mesh_xml,
    read_domains,
    read_elementdata_mat_axis,  # not sure this was supported in FEBio XML 2.0
    read_sequences,
    read_body_bcs,
    xml_dynamics,
)


# Facts about FEBio XML 2.0

VERSION = "2.0"

BODY_COND_PARENT = "Constraints"
BODY_COND_NAME = "rigid_body"
# Implicit rigid bodies do not seem to be supported in FEBio XML 2.0.
IMPBODY_PARENT = None
IMPBODY_NAME = None
MESH_TAG = "Geometry"
ELEMENTDATA_PARENT = "Geometry"
NODEDATA_PARENT = "Geometry"
ELEMENTSET_PARENT = "Geometry"
SEQUENCE_PARENT = "LoadData"
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
    "n": ReqParameter("Control/time_steps", int),
    "dtnom": ReqParameter("Control/step_size", to_number),
    "dtmin": OptParameter(
        "Control/time_stepper/dtmin", to_number, 0
    ),  # undocumented default
    "dtmax": OptParameter(
        "Control/time_stepper/dtmax", to_number, 0.05
    ),  # undocumented default
}
# Map of Controller fields → elements relative to <Step>
CONTROLLER_PARAMS = {
    "max_retries": OptParameter("Control/time_stepper/max_retries", int, 5),
    "opt_iter": OptParameter("Control/time_stepper/opt_iter", int, 10),
    "save_iters": OptParameter("Control/plot_level", SaveIters, SaveIters.MAJOR),
}
# Map of Solver fields → elements relative to <Step>
SOLVER_PATH_IN_STEP = "Control"
SOLVER_PARAMS = {
    "dtol": OptParameter("Control/dtol", to_number, 0.001),
    "etol": OptParameter("Control/etol", to_number, 0.01),
    "rtol": OptParameter("Control/rtol", to_number, 0),
    "lstol": OptParameter("Control/lstol", to_number, 0.9),
    "ptol": OptParameter("Control/ptol", to_number, 0.01),
    "min_residual": OptParameter("Control/min_residual", to_number, 1e-20),
    "reform_each_time_step": OptParameter(
        "Control/reform_each_time_step", to_bool, True
    ),
    "reform_on_diverge": OptParameter("Control/diverge_reform", to_bool, True),
    "max_refs": OptParameter("Control/max_refs", int, 15),
    "max_ups": OptParameter("Control/max_ups", int, 10),
}
DEFAULT_UPDATE_METHOD = "BFGS"
QNMETHOD_PATH_IN_STEP = "Control/qnmethod"
QNMETHOD_PARAMS = {
    "max_ups": OptParameter("Control", int, 10),
}


# Functions for reading FEBio XML 2.0


def iter_node_conditions(root):
    """Return generator over prescribed nodal condition info.

    Returns dict of property names → values.  All properties are
    guaranteed to be not-None, except "nodal values", which will be None
    if the condition applies the same condition to all nodes.

    All returned IDs are 0-indexed for consistency with waffleiron.

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


def read_dynamics(e):
    return Dynamics(e.attrib["type"].lower())


def read_fixed_node_bcs(root: etree.Element, model):
    """Return nodesets with fixed degrees of freedom

    :param root: <febio_spec> Element
    :param nodesets: Map of nodeset name → nodeset
    :return: Map of (dof, var) → NodeSet

    In FEBio XML 2.0, the parent XML element is Boundary/fix.  The fixed DoFs are
    stored in an attribute as a concatenate string, like:

    <fix bc="xyz" set="nodeset_name"/>

    """
    bcs = {}
    for e_fix in root.findall(f"Boundary/{BC_TYPE_TAG['node']['fixed']}"):
        fx_kws = [char for char in e_fix.attrib["bc"].strip()]
        for k in fx_kws:
            dof = DOF_NAME_FROM_XML_NODE_BC[k]
            var = VAR_FROM_XML_NODE_BC[k]
            # In FEBio XML 2.0, each node to which the fixed boundary
            # condition is applied is listed under the <fix> tag.
            node_ids = set()
            for e_node in e_fix:
                node_ids.add(int(e_node.attrib["id"]) - 1)
            # Preserve fixed conditions for this DoF and variable for other nodes. There
            # may be multiple <bc> elements with overlapping DoFs and variables but
            # different node lists.  Note also that since nodesets are implemented as
            # immutable frozen sets a name for an equal nodeset will also apply to this
            # nodeset.  Therefore,  if the nodeset is named in an FEBio XML 2.0 file and
            # read into waffleiron, its original name should be used when it is exported to
            # name-oriented formats like FEBio XML 2.5.
            nodeset = NodeSet(model.fixed["node"][(dof, var)]) | node_ids
            bcs[(dof, var)] = nodeset
    return bcs


def read_nodeset(e_nodeset):
    """Return list of node IDs (zero-indexed) in <NodeSet>"""
    items = [
        ZeroIdxID(int(e_item.attrib["id"]) - 1) for e_item in e_nodeset.getchildren()
    ]
    return items


# Functions for writing FEBio XML 2.0


def contact_bare_xml(contact, contact_name=None):
    """Return <contact> element specifying type and surfaces

    In FEBio XML 2.0, the surfaces involved in a contact are written as children of
    the <contact> element.

    """
    e_contact = ET.Element("contact", type=contact.algorithm)
    # Contact name
    if contact_name is not None:
        e_contact.attrib["name"] = str(contact_name)
    # Contact surfaces
    e_leader = ET.SubElement(e_contact, "surface", type="master")
    for f in contact.leader:
        e_leader.append(tag_face(f))
    e_follower = ET.SubElement(e_contact, "surface", type="slave")
    for f in contact.follower:
        e_follower.append(tag_face(f))
    return e_contact


def xml_meshdata(model):
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
            ET.SubElement(e_element, "fiber").text = ",".join(
                str(a) for a in element.properties["v_fiber"]
            )
    e_elemsets = tuple()  # For compatibility with other XML versions
    return e_meshdata, e_elemsets


def read_solver(step_xml):
    """Return Solver instance from <Step> XML"""
    solver_kwargs = read_parameters(step_xml, SOLVER_PARAMS)
    return Solver(**solver_kwargs)


######################################################
# Functions to create XML elements for FEBio XML 2.0 #
######################################################

# Each of these functions should return one or more XML elements.  As much as possible,
# their arguments should be data, not a `Model`, the whole XML tree, or other
# specialized objects.  Even use of name registries should minimized in favor of simple
# dictionaries when possible.


def xml_nodeset(nodes, name):
    """Return XML element for a (named) node set"""
    e = etree.Element("NodeSet", name=name)
    # Sort nodes to be user-friendly (humans often read .feb files)
    for node_id in sorted(nodes):
        etree.SubElement(e, "node", id=str(node_id + 1))
    return e


def xml_node_fixed_bcs(fixed_conditions, nodeset_registry):
    """Return XML elements for node fixed displacement conditions.

    :param fixed_conditions: The data structure in model.fixed["node"]

    This function may create and add new nodesets to the nodeset name
    registry.  If generating a full XML tree, be sure to write these new
    nodesets to the tree.

    """
    e_bcs = []
    # Tag hierarchy: <Boundary><fix bc="x"><node id="1"> for each node
    for (dof, var), nodeset in fixed_conditions.items():
        if not nodeset:
            continue
        e_bc = ET.Element(BC_TYPE_TAG["node"]["fixed"], bc=XML_BC_FROM_DOF[(dof, var)])
        for i in nodeset:
            ET.SubElement(e_bc, "node", id=str(i + 1))
        e_bcs.append(e_bc)
    return e_bcs


def tag_face(face):
    nm = {3: "tri3", 4: "quad4"}
    tag = ET.Element(nm[len(face)])
    tag.text = ", ".join([f"{i+1}" for i in face])
    return tag


def xml_node_var_bc(model, xmlroot, nodes, scales, seq, dof, var, relative, step_name):
    e_bc = ET.Element(
        "prescribe",
        bc=XML_BC_FROM_DOF[(dof, var)],
    )
    seq_id = get_or_create_seq_id(model.named["sequences"], seq)
    e_bc.attrib["lc"] = str(seq_id + 1)
    # Write nodes as children of <Step><Boundary><prescribe>
    for i, sc in zip(nodes, scales):
        ET.SubElement(e_bc, "node", id=str(i + 1)).text = float_to_text(sc)
    return e_bc


def xml_qnmethod(solver):
    """Convert Solver.update_method to XML"""
    conv = {"BFGS": "0", "Broyden": "1", "Newton": "0"}
    # ^ you only actually get Newton iterations if max_ups = 0
    return const_property_to_xml(conv[solver.update_method], "qnmethod")
