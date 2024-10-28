from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from lxml import etree
from numpy import ndarray

from . import ScaledSequence, NodeSet
from .control import Solver, Dynamics
from .core import ZeroIdxID, Body, ImplicitBody, Sequence, ElementSet
from .febioxml import (
    OptParameter,
    to_number,
    to_bool,
    body_mat_id,
    group_constraints_fixed_variable,
    get_or_create_seq_id,
    bool_to_text,
    XML_BC_FROM_DOF,
    bcs_by_nodeset_and_var,
    const_property_to_xml,
    find_unique_tag,
    DOF_NAME_FROM_XML_NODE_BC,
    VAR_FROM_XML_NODE_BC,
    read_parameters,
    bvec_to_text,
    BodyConstraint,
    read_parameter,
    read_mat_axis_xml,
    parse_nodeset_ref,
    read_nodeset_ref,
)
from .febioxml_3_0 import (
    CONTACT_PARAMS,
    TICKER_PARAMS,
    CONTROLLER_PARAMS,
    get_surface_name,
    iter_node_conditions,
    contact_bare_xml,
    surface_pair_xml,
    read_domains,
    read_sequences,
    mesh_xml,
    xml_meshdata,
    physics_compat_by_mat,
    sequence_xml,
    node_data_xml,
    DOF_FROM_XML_RB_DOF,
    VAR_FROM_XML_RB_DOF,
)

# XML element parents and names
BODY_COND_PARENT = "Rigid"
BODY_COND_NAME = "rigid_bc"
IMPBODY_PARENT = "Boundary"
IMPBODY_NAME = "bc[@type='rigid']"
MESH_PARENT = "Mesh"
ELEMENTDATA_PARENT = "MeshData"
NODEDATA_PARENT = "MeshData"
ELEMENTSET_PARENT = "Mesh"
NODESET_PARENT = "Mesh"
SEQUENCE_PARENT = "LoadData"
SURFACEPAIR_LEADER_NAME = "primary"
SURFACEPAIR_FOLLOWER_NAME = "secondary"
STEP_PARENT = "Step"
STEP_NAME = "step"


BC_TYPE_TAG: Dict[str, dict] = {
    "body": {
        ("variable", "displacement"): "rigid_displacement",
        ("fixed", "displacement"): "rigid_fixed",
        ("variable", "force"): "rigid_force",
    },
}

XML_RB_FIX_DOF_FROM_DOF = {
    "x1": "Rx",
    "x2": "Ry",
    "x3": "Rz",
    "α1": "Ru",
    "α2": "Rv",
    "α3": "Rw",
}
XML_RB_VAR_DOF_FROM_DOF = {
    "x1": "x",
    "x2": "y",
    "x3": "z",
    "α1": "Ru",
    "α2": "Rv",
    "α3": "Rw",
}
DOF_FROM_XML_RB_FIX_DOF = {v: k for k, v in XML_RB_FIX_DOF_FROM_DOF.items()}
DOF_FROM_XML_RB_VAR_DOF = {v: k for k, v in XML_RB_VAR_DOF_FROM_DOF.items()}

XML_NODE_DOF = {
    "x1": "x",
    "x2": "y",
    "x3": "z",
}
NODE_DOF_FROM_XML = {v: k for k, v in XML_NODE_DOF.items()}

DYNAMICS_TO_XML = {
    Dynamics.STATIC: "0",
    Dynamics.DYNAMIC: "1",
}

# Map of Solver fields → elements relative to <Step>
SOLVER_PATH_IN_STEP = "Control/solver"
SOLVER_PARAMS = {
    "dtol": OptParameter("Control/solver/dtol", to_number, 0.001),
    "etol": OptParameter("Control/solver/etol", to_number, 0.01),
    "rtol": OptParameter("Control/solver/rtol", to_number, 0),
    "lstol": OptParameter("Control/solver/lstol", to_number, 0.9),
    "ptol": OptParameter("Control/solver/ptol", to_number, 0.01),
    "min_residual": OptParameter("Control/solver/min_residual", to_number, 1e-20),
    "reform_each_time_step": OptParameter(
        "Control/solver/reform_each_time_step", to_bool, True
    ),
    "reform_on_diverge": OptParameter("Control/solver/diverge_reform", to_bool, True),
    "max_refs": OptParameter("Control/solver/max_refs", int, 15),
}
DEFAULT_UPDATE_METHOD = "BFGS"
QNMETHOD_PATH_IN_STEP = "Control/solver/qn_method"
QNMETHOD_PARAMS = {
    "max_ups": OptParameter("Control/solver/qn_method/max_ups", int, 10),
}


#####################################################
# Functions to parse XML elements for FEBio XML 4.0 #
#####################################################

# Each of these functions should convert one or more XML elements to a useful data
# structure or a Waffleiron object.


def read_nodeset(e_nodeset):
    """Return list of node IDs (zero-indexed) in <NodeSet>"""
    return [ZeroIdxID(int(s.strip()) - 1) for s in e_nodeset.text.split(",")]


def read_elementdata_mat_axis(
    tree_root, element_sets: Optional[Dict[str, ElementSet]] = None
) -> Dict[str, Tuple[int, ndarray]]:
    """Return a dictionary of all mat_axis data"""
    data = defaultdict(list)
    for e_edata in tree_root.findall(
        f"{ELEMENTDATA_PARENT}/ElementData[@type='mat_axis']"
    ):
        name = e_edata.attrib["elem_set"]
        if element_sets is not None and name not in element_sets:
            raise ValueError(
                f"{e_edata.base}:{e_edata.sourceline} <ElementData> references an element set named '{name}', which is not defined."
            )
        for e in e_edata.findall("elem"):
            data[name].append(read_mat_axis_xml(e))
    return data


def read_fixed_node_bcs(root: etree.Element, model):
    """Return nodesets with fixed degrees of freedom

    :param root: <febio_spec> Element
    :param nodesets: Map of nodeset name → nodeset.  All nodesets referenced by the
    fixed BC XML elements must have names stored in this parameter.
    :return: Map of (dof, var) → NodeSet

    In FEBio XML 3.0, the parent XML element is Boundary.  The fixed DoFs are stored
    in the text content of a child element <dofs> as a comma-separated string, like:

    <bc type="fix" set="nodeset_name">
      <dofs>x,y,z</dofs>
    </bc>

    """
    bcs = {}
    # Zero node displacement
    for e_fix in root.findall(f"Boundary/bc[@type='zero displacement']"):
        var = "displacement"
        for e_dof in e_fix:
            xml_dof = e_dof.tag.removesuffix("_dof")
            dof = NODE_DOF_FROM_XML[xml_dof]
            # The node set to which the fixed boundary condition is applied is
            # referenced by name.  The name must already be present in the model's
            # name registry.
            bcs[(dof, var)] = read_nodeset_ref(
                e_fix.attrib["node_set"],
                node_sets=model.named["node sets"],
                face_sets=model.named["face sets"],
                element_sets=model.named["element sets"],
            )
    # Zero node fluid pressure
    for e_fix in root.findall(f"Boundary/bc[@type='zero fluid pressure']"):
        dof = "fluid"
        var = "pressure"
        bcs[(dof, var)] = read_nodeset_ref(
            e_fix.attrib["node_set"],
            node_sets=model.named["node sets"],
            face_sets=model.named["face sets"],
            element_sets=model.named["element sets"],
        )
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
    mat_name = find_unique_tag(e_rigid_bc, "rb").text
    if mat_name in explicit_bodies:
        body = explicit_bodies[mat_name]
    else:
        # Assume mat_id refers to an implicit rigid body
        # TODO: Why are these separated anyway?
        body = implicit_bodies[mat_name]
    # Variable displacement (and rotation)
    if e_rigid_bc.attrib["type"] == "prescribe":
        var = "displacement"
        dof = DOF_FROM_XML_RB_DOF[find_unique_tag(e_rigid_bc, "dof").text]
        e_seq = find_unique_tag(e_rigid_bc, "value")
        seq = read_parameter(e_seq, sequences)
        # Is the displacement relative?
        e_relative = find_unique_tag(e_rigid_bc, "relative")
        if e_relative is None:
            is_relative = False
        else:
            is_relative = to_bool(e_relative.text)
        constraints.append(BodyConstraint(body, dof, var, False, seq, is_relative))
    # Fixed displacement (and rotation)
    elif e_rigid_bc.attrib["type"] == "fix":
        xml_dofs = (
            s.strip() for s in find_unique_tag(e_rigid_bc, "dofs").text.split(",")
        )
        for xml_dof in xml_dofs:
            dof = DOF_FROM_XML_RB_DOF[xml_dof]
            var = VAR_FROM_XML_RB_DOF[xml_dof]
            constraints.append(BodyConstraint(body, dof, var, True, None, None))
    # TODO: variable force
    return constraints


def read_rigid_interface(e_bc):
    """Parse a <bc type="rigid"> element"""
    nodeset_name = e_bc.attrib["node_set"]
    e_rb = find_unique_tag(e_bc, "rb")
    mat_name = e_rb.text
    return nodeset_name, mat_name


def read_dynamics(e):
    """Read dynamics from FEBio XML"""
    DYNAMICS_FROM_XML = {
        "0": Dynamics.STATIC,
        "1": Dynamics.DYNAMIC,
        "static": Dynamics.STATIC,
        "quasi-static": Dynamics.STATIC,
        "steady-state": Dynamics.STATIC,
        "dynamic": Dynamics.DYNAMIC,
        "transient": Dynamics.DYNAMIC,
    }
    return DYNAMICS_FROM_XML[e.text.lower()]


def read_solver(step_xml):
    """Return Solver instance from <Step> XML"""
    solver_kwargs = read_parameters(step_xml, SOLVER_PARAMS)
    # Read qn_method, which is now separate
    e_qn = find_unique_tag(step_xml, QNMETHOD_PATH_IN_STEP)
    if len(e_qn) != 0:
        qn_params = read_parameters(e_qn, QNMETHOD_PARAMS)
        solver_kwargs["update_method"] = e_qn.attrib["type"]
    else:
        solver_kwargs["update_method"] = DEFAULT_UPDATE_METHOD
        # Get defaults
        qn_params = read_parameters(etree.Element("dummy"), QNMETHOD_PARAMS)
    solver_kwargs |= qn_params
    return Solver(**solver_kwargs)


##############################################################
# Functions to mutate a Waffleiron model using FEBio XML 4.0 #
##############################################################

# Each of these functions takes a Waffleiron model (or another object) and modifies
# it.  It is better to avoid this in favor of returning data structures, but it is
# nonetheless sometimes necessary.


# TODO: Add apply_body_bc


######################################################
# Functions to create XML elements for FEBio XML 4.0 #
######################################################

# Each of these functions should return one or more XML elements.  As much as possible,
# their arguments should be data, not a `Model`, the whole XML tree, or other
# specialized objects.  Even use of name registries should minimized in favor of simple
# dictionaries when possible.


def xml_meshdata(model):
    """Return <ElementData> and <ElementSet> XML elements

    Currently this function only generates the part of the MeshData section that
    deals with material axis element data.

    """
    e_meshdata = []
    e_elemsets = []
    element_set_name = "autogen-mat_axis"
    e_edata_mat_axis = etree.Element(
        "ElementData", type="mat_axis", elem_set=element_set_name
    )
    element_ids = []
    i_elemset = 0  # index into the extra element set we're forced to construct
    for i, e in enumerate(model.mesh.elements):
        # Write local basis if defined
        if e.basis is not None:
            element_ids.append(i)
            # Add the element to MeshData/ElementData
            e_elem = etree.SubElement(e_edata_mat_axis, "elem", lid=str(i_elemset + 1))
            e_elem.append(etree.Comment(f"Element {i + 1}"))
            i_elemset += 1
            etree.SubElement(e_elem, "a").text = bvec_to_text(e.basis[:, 0])
            etree.SubElement(e_elem, "d").text = bvec_to_text(e.basis[:, 1])
    # Create the named element set
    e_elemset_mat_axis = etree.Element("ElementSet", name=element_set_name)
    e_elemset_mat_axis.text = ", ".join([str(i + 1) for i in element_ids])
    if len(e_edata_mat_axis) != 0:
        e_meshdata.append(e_edata_mat_axis)
        e_elemsets.append(e_elemset_mat_axis)
    return e_meshdata, e_elemsets


def xml_body_fixed_constraints(
    material_name: str,
    fixed_constraints: dict,
):
    e = etree.Element(BODY_COND_NAME)
    e.attrib["type"] = BC_TYPE_TAG["body"][("fixed", "displacement")]
    etree.SubElement(e, "rb").text = material_name
    for dof, _ in fixed_constraints:
        etree.SubElement(e, f"{XML_RB_FIX_DOF_FROM_DOF[dof]}_dof").text = "1"
    return e


def xml_body_constraints(
    body, constraints: dict, material_registry, implicit_rb_mats, sequence_registry
):
    """Return <rigid_bc> element(s) for body's displacement/rotation constraints"""
    elems = []
    _, mat_name = body_mat_id(body, material_registry, implicit_rb_mats)
    # Can't put fixed and variable constraints in the same <rigid_bc> element
    fixed_constraints, variable_constraints = group_constraints_fixed_variable(
        constraints
    )
    # Create <rigid_bc> element for fixed constraints
    elems.append(xml_body_fixed_constraints(mat_name, fixed_constraints))
    # Create <rigid_bc> element for variable constraints.
    for dof, bc in variable_constraints:
        e_rb = etree.Element(BODY_COND_NAME)
        k = ("variable", bc["variable"])
        e_rb.attrib["type"] = BC_TYPE_TAG["body"][k]
        etree.SubElement(e_rb, "rb").text = mat_name
        etree.SubElement(e_rb, "dof").text = XML_RB_VAR_DOF_FROM_DOF[dof]
        seq = bc["sequence"]
        seq_id = get_or_create_seq_id(sequence_registry, seq)
        e_value = etree.SubElement(e_rb, "value")
        e_value.attrib["lc"] = str(seq_id + 1)
        v = bc["scale"]
        if isinstance(bc["sequence"], ScaledSequence):
            v = v * bc["sequence"].scale
        e_value.text = str(v)
        if bc["variable"] == "force":
            etree.SubElement(e_rb, "load_type").text = "0"
            # ^ semantics of this are unclear, but this is what FEBio
            # Studio 1.3 exports
        #
        # FEBio only supports relative constraints for displacement
        if bc["variable"] == "displacement":
            etree.SubElement(e_rb, "relative").text = bool_to_text(bc["relative"])
        elif bc["relative"]:
            # Most likely: bc['variable'] == "force"
            raise ValueError(
                f"FEBio XML does not permit relative {bc['variable']} conditions for bodies."
            )
        elems.append(e_rb)
    return elems


def xml_nodeset(nodes, name):
    """Return XML element for a (named) node set"""
    e = etree.Element("NodeSet", name=name)
    # Sort nodes to be user-friendly (humans often read .feb files) and to have
    # canonical XML output.
    e.text = ", ".join(str(id_ + 1) for id_ in sorted(nodes))
    return e


def xml_node_fixed_bcs(fixed_conditions, nodeset_registry):
    """Return XML elements for node fixed displacement conditions.

    :param fixed_conditions: The data structure in model.fixed["node"]

    This function may create and add new nodesets to the nodeset name registry.  If
    generating a full XML tree, be sure to write these new nodesets to the tree.

    """
    # <Boundary><bc type="inconsistent type" node_set="set_name">

    e_bcs = []
    # FEBio XML 4.0 stores the nodal BCs by node set
    by_nodeset = bcs_by_nodeset_and_var(fixed_conditions)
    for nodeset, dofs_by_var in by_nodeset.items():
        # Get or create a name for the node set
        nodeset_name = "fixed_" + "_".join(
            [f"{'_'.join(dofs)}_{var}" for var, dofs in dofs_by_var.items()]
        )
        # Name may be modified for disambiguation
        nodeset_name = nodeset_registry.get_or_create_name(nodeset_name, nodeset)
        for var, dofs in dofs_by_var.items():
            # FEBio XML 4.0 formats fixed nodal boundary conditions inconsistently.
            # Some variables have child elements listing DoFs; others don't.  It's a
            # mess.  So just handle the cases individually.

            # Zero displacement
            if var == "displacement":
                e_bc = etree.Element(
                    "bc", type="zero displacement", node_set=nodeset_name
                )
                for dof in dofs:
                    e_dof = etree.SubElement(e_bc, f"{XML_NODE_DOF[dof]}_dof")
                    e_dof.text = "1"  # silly
                e_bcs.append(e_bc)

            # Zero fluid pressure
            if var == "pressure" and dofs[0] == "fluid":
                assert len(dofs) == 1
                e_bc = etree.Element(
                    "bc", type="zero fluid pressure", node_set=nodeset_name
                )
                e_bcs.append(e_bc)

            # TODO: Zero concentration
    return e_bcs


def xml_node_var_bc(model, xmlroot, nodes, scales, seq, dof, var, relative, step_name):
    """Return XML elements for nodes variable displacement

    :param model:   Model object.  Needed for the name registry.

    Returns tuple of (<bc> element, <NodeData> element)

    """
    # <Boundary><bc type="special snowflake" node_set="set_name">

    # TODO: Passing in the whole model is excessive

    if var == "displacement":
        e_bc = etree.Element("bc", type="prescribed displacement")
        e_dof = etree.SubElement(e_bc, "dof").text = XML_BC_FROM_DOF[(dof, var)]
    elif (dof, var) == ("fluid", "pressure"):
        e_bc = etree.Element("bc", type="prescribed fluid pressure")
    seq_id = get_or_create_seq_id(model.named["sequences"], seq)
    e_sc = etree.SubElement(e_bc, "value", lc=str(seq_id + 1), type="map")
    # Other subelements
    etree.SubElement(e_bc, "relative").text = str(int(relative))
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


def xml_rigid_nodeset_bc(name: str, material_name: str = None, material_id: int = None):
    """Return XML element for a rigid node set (implicit rigid body)

    :param name: Name of node set to be treated as rigid.

    :param material_name: Name of rigid material corresponding to this rigid node set.

    :param material_id: Ordinal ID (in FEBio XML; 1-indexed) of rigid material
    corresponding to this rigid node set.  Not needed in FEBio XML 4.0; included only
    for call signature compatibility.

    """
    if material_name is None:
        raise ValueError("Must provide material_name.")
    e = etree.Element("bc")
    e.attrib["type"] = "rigid"
    e.attrib["node_set"] = name
    etree.SubElement(e, "rb").text = material_name
    return e


def xml_dynamics(dynamics: Dynamics, model=None):
    """Return <analysis> element"""
    # "static" is valid for the "solid" module, but not the "biphasic" module.
    # "steady-state" is valid for the "biphasic" module, but not the "elastic"
    # module.  Fortunately you can use "0" for (quasi-)static and "1" for dynamic,
    # which works for all modules.
    e = etree.Element("analysis")
    e.text = DYNAMICS_TO_XML[dynamics]
    return e


def xml_qnmethod(solver: Solver):
    """Convert Solver.update_method to XML"""
    e = etree.Element("qn_method")
    # FEBio XML 4.0 represents Newton updates as BFGS with max_ups = 0
    e.attrib["type"] = {"BFGS": "BFGS", "Broyden": "Broyden", "Newton": "BFGS"}[
        solver.update_method
    ]
    e_maxups = const_property_to_xml(solver.max_ups, "max_ups")
    # ^ you only actually get Newton iterations if max_ups = 0
    e.append(e_maxups)
    return e
