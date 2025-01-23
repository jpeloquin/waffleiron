# Base packages
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Public packages
from lxml import etree
from numpy import ndarray

# Same-package modules
from .core import (
    NodeSet,
    ZeroIdxID,
    OneIdxID,
    Sequence,
    Interpolant,
    Extrapolant,
    Body,
    ImplicitBody,
    ScaledSequence,
    ElementSet,
)
from .control import Dynamics, SaveIters, Solver, Physics
from .febioxml import (
    CONTACT_PARAMS,
    ReqParameter,
    OptParameter,
    to_number,
    to_bool,
    DOF_NAME_FROM_XML_NODE_BC,
    VAR_FROM_XML_NODE_BC,
    find_unique_tag,
    EXTRAP_FROM_XML_EXTRAP,
    read_point,
    INTERP_FROM_XML_INTERP,
    read_parameter,
    body_mat_id,
    get_or_create_seq_id,
    XML_BC_FROM_DOF,
    CONTACT_NAME_FROM_CLASS,
    vec_to_text,
    vec_to_text,
    read_parameters,
    float_to_text,
    XML_INTERP_FROM_INTERP,
    XML_EXTRAP_FROM_EXTRAP,
    const_property_to_xml,
    BodyConstraint,
    read_mat_axis_xml,
)

# Facts about FEBio XML 2.5

VERSION = "2.5"

# XML element parents and names
BODY_COND_PARENT = "Boundary"
BODY_COND_NAME = "rigid_body"
IMPBODY_PARENT = "Boundary"
IMPBODY_NAME = "rigid"
MESH_PARENT = "Geometry"
ELEMENTDATA_PARENT = "MeshData"
NODEDATA_PARENT = "MeshData"
ELEMENTSET_PARENT = "Geometry"
SEQUENCE_PARENT = "LoadData"
STEP_PARENT = "."
STEP_NAME = "Step"
SURFACEPAIR_LEADER_NAME = "master"
SURFACEPAIR_FOLLOWER_NAME = "slave"

BC_TYPE_TAG = {
    "node": {"variable": "prescribe", "fixed": "fix"},
    "body": {"variable": "prescribed", "fixed": "fixed"},
}

DYNAMICS_TO_XML = {
    (Physics.SOLID, Dynamics.STATIC): "static",
    (Physics.SOLID, Dynamics.DYNAMIC): "dynamic",
    (Physics.BIPHASIC, Dynamics.STATIC): "steady-state",
    (Physics.BIPHASIC, Dynamics.DYNAMIC): "transient",
    (Physics.MULTIPHASIC, Dynamics.STATIC): "steady-state",
    (Physics.MULTIPHASIC, Dynamics.DYNAMIC): "transient",
}
DYNAMICS_FROM_XML = {txt: dyn for (phys, dyn), txt in DYNAMICS_TO_XML.items()}

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


# Functions for reading FEBio XML 2.5


def elem_var_fiber_xml(e):
    tag = etree.Element("elem")
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


def get_surface_name(surfacepair_subelement):
    """Return surface name for subelement of SurfacePair

    For example, return "surface1" for the element <primary surface="surface1"/>.

    This function exists because the surface name changed to element content in FEBio
    XML 3.0.

    """
    return surfacepair_subelement.attrib["surface"]


def read_domains(root: etree.Element):
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


def read_nodeset(e_nodeset):
    """Return list of node IDs (zero-indexed) in <NodeSet>"""
    items = [
        ZeroIdxID(int(e_item.attrib["id"]) - 1) for e_item in e_nodeset.getchildren()
    ]
    return items


def read_elementdata_mat_axis(
    tree_root, element_sets: Optional[Dict[str, ElementSet]] = None
) -> Dict[str, Tuple[int, ndarray]]:
    """Return a dictionary of all mat_axis data"""
    data = defaultdict(list)
    for e_edata in tree_root.findall(
        f"{ELEMENTDATA_PARENT}/ElementData[@var='mat_axis']"
    ):
        name = e_edata.attrib["elem_set"]
        if element_sets is not None and name not in element_sets:
            raise ValueError(
                f"{e_edata.base}:{e_edata.sourceline} <ElementData> references an element set named '{name}', which is not defined."
            )
        for e in e_edata.findall("elem"):
            data[name].append(read_mat_axis_xml(e))
    return data


def read_sequences(root: etree.Element) -> Dict[int, Sequence]:
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
        sequences[ord_id] = Sequence(
            curve, interp=interp, extrap=extrap, steplocal=False
        )
    return sequences


def read_dynamics(e):
    return DYNAMICS_FROM_XML[e.attrib["type"].lower()]


def read_fixed_node_bcs(root: etree.Element, model):
    """Return nodesets with fixed degrees of freedom

    :param root: <febio_spec> Element
    :param nodesets: Map of nodeset name → nodeset.  All nodesets referenced by the
    fixed BC XML elements must have names stored in this parameter.
    :return: Map of (dof, var) → NodeSet

    In FEBio XML 2.5, the parent XML element is Boundary/fix.  The fixed DoFs are
    stored in an attribute as a comma-separated string, like:

    <fix bc="x,y,z" set="nodeset_name"/>

    """
    bcs = {}
    for e_fix in root.findall(f"Boundary/{BC_TYPE_TAG['node']['fixed']}"):
        fx_kws = [kw.strip() for kw in e_fix.attrib["bc"].split(",")]
        for k in fx_kws:
            dof = DOF_NAME_FROM_XML_NODE_BC[k]
            var = VAR_FROM_XML_NODE_BC[k]
            # In FEBio XML 2.5, the node set to which the fixed boundary condition is
            # applied is referenced by name.  The name must already be present in the
            # model's name registry.
            nodeset = model.named["node sets"].obj(e_fix.attrib["node_set"])
            bcs[(dof, var)] = nodeset
    return bcs


def read_body_bcs(
    root, explicit_bodies, implicit_bodies, sequences
) -> List[BodyConstraint]:
    """Return list of rigid body constraints from FEBio XML 4.0"""
    body_constraints = []
    for e_rbc in root.findall(f"{BODY_COND_PARENT}/{BODY_COND_NAME}"):
        body_constraints += read_body_bc(
            e_rbc, explicit_bodies, implicit_bodies, sequences
        )
    return body_constraints


def read_body_bc(
    e_rigid_bc,
    explicit_bodies: Dict[int, Body],
    implicit_bodies: Dict[int, ImplicitBody],
    sequences: Dict[int, Sequence],
) -> List[BodyConstraint]:
    """Return structured data for <rigid_bc>

    Returns a list because a <rigid_bc> element can store more than one DoF.

    """
    # Each <rigid_body> element defines constraints for one rigid body, identified by
    # its material ID.  Constraints may be fixed (constant) or time-varying ( variable).
    constraints = []

    # Get the Body object from the material id
    mat_id = int(e_rigid_bc.attrib["mat"]) - 1
    if mat_id in explicit_bodies:
        body = explicit_bodies[mat_id]
    else:
        # Assume mat_id refers to an implicit rigid body
        body = implicit_bodies[mat_id]
    # Variable displacement (and rotation)
    for e_dof in e_rigid_bc.findall(BC_TYPE_TAG["body"]["variable"]):
        dof = DOF_NAME_FROM_XML_NODE_BC[e_dof.attrib["bc"]]
        var = VAR_FROM_XML_NODE_BC[e_dof.attrib["bc"]]
        seq = read_parameter(e_dof, sequences)
        if e_dof.get("type", None) == "relative":
            is_relative = True
        else:
            is_relative = False
        constraints.append(BodyConstraint(body, dof, var, False, seq, is_relative))
    # Fixed displacement (and rotation)
    for e_dof in e_rigid_bc.findall(BC_TYPE_TAG["body"]["fixed"]):
        dof = DOF_NAME_FROM_XML_NODE_BC[e_dof.attrib["bc"]]
        var = VAR_FROM_XML_NODE_BC[e_dof.attrib["bc"]]
        constraints.append(BodyConstraint(body, dof, var, True, None, None))
    # TODO: variable force
    return constraints


def read_rigid_interface(e_rigid):
    """Parse a <rigid> element"""
    mat_id = int(e_rigid.attrib["rb"]) - 1
    nodeset_name = e_rigid.attrib["node_set"]
    return nodeset_name, mat_id


def read_solver(step_xml):
    """Return Solver instance from <Step> XML"""
    solver_kwargs = read_parameters(step_xml, SOLVER_PARAMS)
    return Solver(**solver_kwargs)


######################################################
# Functions to create XML elements for FEBio XML 2.5 #
######################################################

# Each of these functions should return one or more XML elements.  As much as possible,
# their arguments should be data, not a `Model`, the whole XML tree, or other
# specialized objects.  Even use of name registries should minimized in favor of simple
# dictionaries when possible.


def xml_body_constraints(
    body, constraints: dict, material_registry, implicit_rb_mats, sequence_registry
):
    """Return <rigid_body> element for a body's constraints.

    The constrained variable can be displacement or rotation.

    """
    mat_id, _ = body_mat_id(body, material_registry, implicit_rb_mats)
    e_rb_bc = etree.Element("rigid_body", mat=str(mat_id + 1))
    for dof, bc in constraints.items():
        if bc["sequence"] == "fixed":
            kind = "fixed"
        elif bc["sequence"]:  # bc['sequence'] is a Sequence
            kind = "variable"
            seq = bc["sequence"]
            v = bc["scale"]
            if isinstance(bc["sequence"], ScaledSequence):
                v = v * bc["sequence"].scale
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
        e_bc = etree.SubElement(e_rb_bc, tagname, bc=bc_attr)
        if kind == "variable":
            seq_id = get_or_create_seq_id(sequence_registry, seq)
            e_bc.attrib["lc"] = str(seq_id + 1)
            if bc["relative"]:
                e_bc.attrib["type"] = "relative"
            e_bc.text = str(v)
    return [e_rb_bc]


def contact_bare_xml(contact, model, named_surface_pairs, contact_name=None):
    """Return <contact> element specifying type and surfaces

    In FEBio XML 2.5, the surfaces involved in a contact are written as a reference
    to a named surface pair.

    """
    contact_tag = CONTACT_NAME_FROM_CLASS[contact.__class__]
    e_contact = etree.Element("contact", type=contact_tag)
    # Contact name
    if contact_name is not None:
        e_contact.attrib["name"] = str(contact_name)
    # Autogenerate names for the "surfaces" (face sets) in the contact.  This doesn't
    # have a direct impact on the <contact> element; the point of doing this is the
    # side effect of naming the face sets involved in the contact.
    surface_name = {"leader": "", "follower": ""}
    for k in surface_name:
        face_set = getattr(contact, k)
        nm = model.named["face sets"].get_or_create_name(
            f"contact_surface_-_{contact_tag}",
            face_set,
        )
        surface_name[k] = nm
    # Contact surface (face set) pair
    nm_surfpair = named_surface_pairs.get_or_create_name(
        f"contact_surfaces_-_{contact_tag}",
        (contact.leader, contact.follower),
    )
    e_contact.attrib["surface_pair"] = nm_surfpair
    return e_contact


def mesh_xml(model, domains, material_registry):
    """Create <Geometry> XML element.

    Returns a tuple because the FEBio XML 3.0 version needs to return
    two XML elements.

    """
    e_geometry = etree.Element(MESH_PARENT)
    # Write <nodes>
    e_nodes = etree.SubElement(e_geometry, "Nodes")
    for i, x in enumerate(model.mesh.nodes):
        feb_nid = i + 1  # 1-indexed
        e = etree.SubElement(e_nodes, "node", id="{}".format(feb_nid))
        e.text = vec_to_text(x)
        e_nodes.append(e)
    # Write <Elements> for each domain
    for i, domain in enumerate(domains):
        if domain["material"] is None:
            raise ValueError("Some elements have no material assigned.")
        e_elements = etree.SubElement(e_geometry, "Elements", name=f"Domain{i + 1}")
        e_elements.attrib["type"] = domain["element_type"].feb_name
        mat_id = material_registry.names(domain["material"], "ordinal_id")[0]
        e_elements.attrib["mat"] = str(mat_id + 1)
        for i, e in domain["elements"]:
            e_element = etree.SubElement(e_elements, "elem")
            e_element.attrib["id"] = str(i + 1)
            e_element.text = ", ".join(str(i + 1) for i in e.ids)
    return (e_geometry,)


def xml_meshdata(model):
    """Return <ElementData> and <ElementSet> XML elements

    Currently this function only generates the part of the MeshData
    section that deals with material axis element data.

    """
    e_meshdata = []
    e_elemsets = []
    e_edata_mat_axis = etree.Element(
        "ElementData", var="mat_axis", elem_set="autogen-mat_axis"
    )
    e_elemset_mat_axis = etree.Element("ElementSet", name="autogen-mat_axis")
    i_elemset = 0
    # ^ index into the extra element set we're forced to construct
    for i, e in enumerate(model.mesh.elements):
        # Write local basis if defined
        if e.basis is not None:
            e_elem = etree.SubElement(e_edata_mat_axis, "elem", lid=str(i_elemset + 1))
            e_elem.append(etree.Comment(f"Element {i + 1}"))
            i_elemset += 1
            etree.SubElement(e_elem, "a").text = vec_to_text(e.basis[:, 0])
            etree.SubElement(e_elem, "d").text = vec_to_text(e.basis[:, 1])
            etree.SubElement(e_elemset_mat_axis, "elem", id=str(i + 1))
    if len(e_edata_mat_axis) != 0:
        e_meshdata.append(e_edata_mat_axis)
        e_elemsets.append(e_elemset_mat_axis)
    return e_meshdata, e_elemsets


def node_data_xml(nodes, data, data_name, nodeset_name):
    """Construct NodeData XML element"""
    e_NodeData = etree.Element("NodeData")
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
        etree.SubElement(
            e_NodeData,
            "node",
            lid=str(lid_from_node_id[i]),
        ).text = float_to_text(v)
    return e_NodeData


def xml_node_fixed_bcs(fixed_conditions, nodeset_registry):
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
        e_bc = etree.Element(
            BC_TYPE_TAG["node"]["fixed"], bc=XML_BC_FROM_DOF[(dof, var)], node_set=name
        )
        e_bcs.append(e_bc)
    return e_bcs


def xml_node_var_bc(model, xmlroot, nodes, scales, seq, dof, var, relative, step_name):
    """Return XML elements for nodal variable displacement

    model := Model object.  Needed for the name registry.

    Returns tuple of (<bc> element, <NodeData> element)

    """
    # Hierarchy: <Boundary><prescribe node_set="set_name">
    e_bc = etree.Element(
        BC_TYPE_TAG["node"]["variable"], bc=XML_BC_FROM_DOF[(dof, var)]
    )
    seq_id = get_or_create_seq_id(model.named["sequences"], seq)
    e_sc = etree.SubElement(e_bc, "scale", lc=str(seq_id + 1))
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
    etree.SubElement(e_bc, "value", node_data=data_name)
    # Other attributes
    etree.SubElement(e_bc, "relative").text = str(int(relative))
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
    e_loadcurve = etree.Element(
        "loadcurve",
        id=str(sequence_id + 1),
        type=XML_INTERP_FROM_INTERP[sequence.interpolant],
        extend=XML_EXTRAP_FROM_EXTRAP[sequence.extrapolant],
    )
    for pt in sequence.points:
        etree.SubElement(e_loadcurve, "point").text = f"{pt[0] + t0}, {pt[1]}"
    return e_loadcurve


def surface_pair_xml(faceset_registry, primary, secondary, name):
    """Return SurfacePair XML element."""
    e_surfpair = etree.Element("SurfacePair", name=name)
    etree.SubElement(
        e_surfpair,
        "master",
        surface=faceset_registry.names(primary)[0],
    )
    etree.SubElement(
        e_surfpair,
        "slave",
        surface=faceset_registry.names(secondary)[0],
    )
    return e_surfpair


def xml_nodeset(nodes, name):
    """Return XML element for a (named) node set"""
    e = etree.Element("NodeSet", name=name)
    # Sort nodes to be user-friendly (humans often read .feb files) and, more
    # importantly, so that local IDs in NodeData elements (FEBio XML 2.5) or mesh_data
    # elements (FEBio XML 3.0) have a stable relationship with actual node IDs.
    for node_id in sorted(nodes):
        etree.SubElement(e, "node", id=str(node_id + 1))
    return e


def xml_rigid_nodeset_bc(name: str, material_name: str = None, material_id: int = None):
    """Return XML element for a rigid node set (implicit rigid body)

    :param name: Name of node set to be treated as rigid.

    :param material_name: Name of rigid material corresponding to this rigid node set.
    Not needed in FEBio XML 2.5; included only for call signature compatibility.

    :param material_id: Ordinal ID (in FEBio XML; 1-indexed) of rigid material
    corresponding to this rigid node set.

    """
    if material_id is None:
        raise ValueError("Must provide material_id.")
    e = etree.Element(IMPBODY_NAME)
    e.attrib["rb"] = str(material_id)
    e.attrib["node_set"] = name
    return e


def xml_dynamics(dynamics: Dynamics, physics):
    """Return <analysis> element"""
    e = etree.Element("analysis")
    e.attrib["type"] = DYNAMICS_TO_XML[(physics, dynamics)]
    return e


def xml_qnmethod(solver):
    """Convert Solver.update_method to XML"""
    conv = {"BFGS": "0", "Broyden": "1", "Newton": "0"}
    # ^ you only actually get Newton iterations if max_ups = 0
    return const_property_to_xml(conv[solver.update_method], "qnmethod")
