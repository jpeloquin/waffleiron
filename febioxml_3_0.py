# Base packages
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from numpy import ndarray

# Same-package modules
from .core import (
    NodeSet,
    ZeroIdxID,
    OneIdxID,
    Body,
    ImplicitBody,
    Extrapolant,
    Interpolant,
    ElementSet,
)
from .control import Dynamics, SaveIters, Solver
from .febioxml import *
from .febioxml_2_5 import (
    DYNAMICS_TO_XML,
    DYNAMICS_FROM_XML,
    read_elementdata_mat_axis,
    read_nodeset,
    read_elementset,
)

# These parts work the same as in FEBio XML 2.5
from .febioxml_2_5 import contact_bare_xml, xml_meshdata, xml_nodeset

# Facts about FEBio XML 3.0

VERSION = "3.0"

# XML element parents and names
BODY_COND_PARENT = "Rigid"
BODY_COND_NAME = "rigid_constraint"
IMPBODY_PARENT = "Boundary"
IMPBODY_NAME = "bc[@type='rigid']"
MESH_TAG = "Mesh"
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
    "node": {"variable": "prescribe", "fixed": "fix"},
    "body": {
        ("variable", "displacement"): "prescribe",
        ("variable", "rotation"): "prescribe",
        ("fixed", "displacement"): "fix",
        ("fixed", "rotation"): "fix",
        ("variable", "force"): "force",
    },
}

XML_RB_DOF_FROM_DOF = {
    "x1": "Rx",
    "x2": "Ry",
    "x3": "Rz",
    "α1": "Ru",
    "α2": "Rv",
    "α3": "Rw",
}
DOF_FROM_XML_RB_DOF = {v: k for k, v in XML_RB_DOF_FROM_DOF.items()}
VAR_FROM_XML_RB_DOF = {
    "Rx": "displacement",
    "Ry": "displacement",
    "Rz": "displacement",
    "Ru": "rotation",
    "Rv": "rotation",
    "Rw": "rotation",
}

# Map of Ticker fields → elements relative to <Step>
TICKER_PARAMS = {
    "n": ReqParameter("Control/time_steps", int),
    "dtnom": ReqParameter("Control/step_size", to_number),
    "dtmin": OptParameter(
        "Control/time_stepper/dtmin", to_number, 0
    ),  # Undocumented default, but zero makes sense as a minimum time
    # step, as no smaller value is allowed.
    "dtmax": OptParameter(
        "Control/time_stepper/dtmax", to_number, 1
    ),  # Undocumented default.  FEBio 3 uses 0 if the value is missing
    # (but doesn't enforce it for the first time step), which is
    # broken in multiple ways.  But if a load curve is provided the
    # scale appears to be ignored.  If a load curve is given without
    # a value (scale) is given, the default should probably be 1.  I
    # think FEBio 2 used 0.05 as the default.
}
# Map of Controller fields → elements relative to <Step>
CONTROLLER_PARAMS = {
    "max_retries": OptParameter("Control/time_stepper/max_retries", int, 5),
    "opt_iter": OptParameter("Control/time_stepper/opt_iter", int, 10),
    "save_iters": OptParameter("Control/plot_level", SaveIters, SaveIters.MAJOR),
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
    "max_ups": OptParameter("Control/solver/max_ups", int, 10),
}
DEFAULT_UPDATE_METHOD = "BFGS"
QNMETHOD_PATH_IN_STEP = "Control/solver/qnmethod"
QNMETHOD_PARAMS = {
    "max_ups": OptParameter("Control/solver/max_ups", int, 10),
}

# Functions for reading FEBio XML 3.0


def iter_node_conditions(root):
    """Return generator over prescribed nodal condition info.

    Returns dict of property names → values.  All properties are
    not-None except the following:

    (1) "nodal values" will be None if the condition applies the same
    condition to all nodes.

    (2) "name" will be None if the boundary condition is not named.

    (3) "scale" will be None if the condition is heterogeneous, as FEBio
    XML 3.0 does not include a scale in this case.

    """
    step_id = -1  # Curent step ID (0-indexed)
    for e_Step in root.findall(f"{STEP_PARENT}/{STEP_NAME}"):
        step_id += 1
        for e_prescribe in e_Step.findall("Boundary/bc[@type='prescribe']"):
            # Re-initialize output
            info = {
                "name": None,
                "node set name": None,
                "axis": None,  # x1, fluid, charge, etc.
                "variable": None,  # displacement, force, pressure, etc.
                "sequence ID": None,
                "scale": None,  # not required in FEBio XML 3.0
                "relative": False,
                "nodal values": None,
                "step ID": None,
            }
            # Read values
            if "name" in e_prescribe.attrib:
                info["name"] = e_prescribe.attrib["name"]
            info["node set name"] = e_prescribe.attrib["node_set"]
            e_dof = find_unique_tag(e_prescribe, "dof")
            info["dof"] = DOF_NAME_FROM_XML_NODE_BC[e_dof.text]
            info["variable"] = VAR_FROM_XML_NODE_BC[e_dof.text]
            e_scale = find_unique_tag(e_prescribe, "scale")
            # ^ <scale> is required because it stores the load curve ID
            info["sequence ID"] = to_number(e_scale.attrib["lc"]) - 1
            if ("type" in e_scale.attrib) and (e_scale.attrib["type"] == "map"):
                # Heterogeneous condition
                e_NodeData = find_unique_tag(
                    root, f"{NODEDATA_PARENT}/NodeData[@name='{e_scale.text}']"
                )
                nm_nodeset = e_NodeData.attrib["node_set"]
                e_NodeSet = find_unique_tag(
                    root, f"{NODESET_PARENT}/NodeSet[@name='{nm_nodeset}']"
                )
                # The semantics of the lid attributes in <NodeData> are
                # unclear.  I am interpreting them as indices into the
                # <NodeSet> list.
                values = {}
                node_ids = [int(e.attrib["id"]) - 1 for e in e_NodeSet.findall("node")]
                info["nodal values"] = {
                    node_ids[int(e.attrib["lid"]) - 1]: to_number(e.text)
                    for e in e_NodeData.findall("node")
                }
            else:
                # Homogeneous condition
                info["scale"] = to_number(e_scale.text)
            e_relative = e_prescribe.find("relative")
            if e_relative is not None:
                info["relative"] = True
            info["step ID"] = step_id
            yield info


def read_domains(root: etree.Element):
    """Return list of domains"""
    element_index_from_id = {
        int(e.attrib["id"]): i
        for i, e in enumerate(root.xpath(f"{MESH_TAG}/Elements/elem"))
    }
    domains = []
    for e in find_unique_tag(root, "MeshDomains"):
        name = e.attrib["name"]
        e_domain = find_unique_tag(root, f"{MESH_TAG}/Elements[@name='{name}']")
        elements = [
            element_index_from_id[int(e.attrib["id"])] for e in e_domain.findall("elem")
        ]
        if e.attrib["mat"] == "":
            material = None
        else:
            material = ("canonical", e.attrib["mat"])
        domain = {
            "name": name,
            "elements": elements,
            "material": material,
        }
        domains.append(domain)
    return domains, element_index_from_id


def get_surface_name(surfacepair_subelement):
    """Return surface name for subelement of SurfacePair

    For example, return "surface1" for the element <primary>surface1</primary>.

    This function exists because the surface name was an attribute in FEBio XML 2.5.

    """
    return surfacepair_subelement.text


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
    for e_fix in root.findall(f"Boundary/bc[@type='{BC_TYPE_TAG['node']['fixed']}']"):
        e_dofs = find_unique_tag(e_fix, "dofs")
        fx_kws = [kw.strip() for kw in e_dofs.text.split(",")]
        for k in fx_kws:
            dof = DOF_NAME_FROM_XML_NODE_BC[k]
            var = VAR_FROM_XML_NODE_BC[k]
            # In FEBio XML 3.0, the node set to which the fixed boundary condition is
            # applied is referenced by name.  The name must already be present in the
            # model's name registry.
            nodeset = model.named["node sets"].obj(e_fix.attrib["node_set"])
            bcs[(dof, var)] = nodeset
    return bcs


def read_body_bcs(
    root, explicit_bodies, implicit_bodies, sequences
) -> List[BodyConstraint]:
    """Return list of rigid body constraints from FEBio XML 3.0"""
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
    mat_id = int(find_unique_tag(e_rigid_bc, "rb").text) - 1
    if mat_id in explicit_bodies:
        body = explicit_bodies[mat_id]
    else:
        # Assume mat_id refers to an implicit rigid body
        body = implicit_bodies[mat_id]
    # Variable displacement, rotation, or force
    if e_rigid_bc.attrib["type"] in ("prescribe", "force"):
        # TODO: Add test for reading variable force BC
        var = {"prescribe": "displacement", "force": "force"}[e_rigid_bc.attrib["type"]]
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
    elif e_rigid_bc.attrib["type"] == "force":
        var = "force"
        dof = DOF_FROM_XML_RB_DOF[find_unique_tag(e_rigid_bc, "dof").text]
        e_seq = find_unique_tag(e_rigid_bc, "value")
        seq = read_parameter(e_seq, sequences)
        # Is the force relative?
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
    # TODO: variable torque (different from variable force?), fixed force, fixed torque
    return constraints


def read_rigid_interface(e_bc):
    """Parse a <bc type="rigid"> element"""
    nodeset_name = e_bc.attrib["node_set"]
    e_rb = find_unique_tag(e_bc, "rb")
    mat_id = int(e_rb.text) - 1
    return nodeset_name, mat_id


def read_sequences(root: etree.Element) -> Dict[int, Sequence]:
    """Return dictionary of sequence ID → sequence from FEBio XML 3.0"""
    sequences = {}
    for ord_id, e_lc in enumerate(root.findall("LoadData/load_controller")):
        if e_lc.attrib["type"] != "loadcurve":
            raise NotImplementedError(
                f"{e_lc.base}:{e_lc.sourceline} <load_controller> element of type '{e_lc.attrib['type']}' is not yet supported."
            )
        fake_id = int(e_lc.attrib["id"])
        e_points = find_unique_tag(e_lc, "points")
        curve = [read_point(e.text) for e in e_points]
        # Set extrapolation
        e_extend = find_unique_tag(e_lc, "extend")
        if e_extend is None:
            # FEBio default
            extrap = Extrapolant.CONSTANT
        else:
            extrap = EXTRAP_FROM_XML_EXTRAP[e_extend.text.lower()]
        # Set interpolation
        e_interp = find_unique_tag(e_lc, "interpolate")
        if e_interp is None:
            # FEBio default
            interp = Interpolant.LINEAR
        else:
            interp = INTERP_FROM_XML_INTERP[e_interp.text.lower()]
        # Create and store the Sequence object
        sequences[ord_id] = Sequence(
            curve, interp=interp, extrap=extrap, steplocal=False
        )
    return sequences


def read_dynamics(e):
    return DYNAMICS_FROM_XML[e.text.lower()]


def read_solver(step_xml):
    """Return Solver instance from <Step> XML"""
    solver_kwargs = read_parameters(step_xml, SOLVER_PARAMS)
    return Solver(**solver_kwargs)


######################################################
# Functions to create XML elements for FEBio XML 3.0 #
######################################################

# Each of these functions should return one or more XML elements.  As much as possible,
# their arguments should be data, not a `Model`, the whole XML tree, or other
# specialized objects.  Even use of name registries should minimized in favor of simple
# dictionaries when possible.


def xml_body_constraints(
    body, constraints: dict, material_registry, implicit_rb_mats, sequence_registry
):
    """Return <rigid_constraint> element(s) for body's constraints.

    The constrained variable can be displacement or rotation.

    """
    elems = []
    mat_id, _ = body_mat_id(body, material_registry, implicit_rb_mats)
    # Can't put fixed and variable constraints in the same
    # <rigid_constraint> element
    fixed_constraints, variable_constraints = group_constraints_fixed_variable(
        constraints
    )
    # Create <rigid_constraint> element for fixed constraints
    if fixed_constraints:
        e_rb_fixed = etree.Element(BODY_COND_NAME)
        e_rb_fixed.attrib["type"] = "fix"
        etree.SubElement(e_rb_fixed, "rb").text = str(mat_id + 1)
        etree.SubElement(e_rb_fixed, "dofs").text = ",".join(
            XML_RB_DOF_FROM_DOF[dof] for dof, _ in fixed_constraints
        )
        elems.append(e_rb_fixed)
    # Create <rigid_constraint> element for variable constraints.  I
    # think you must use a separate element for each degree of freedom
    # (x, y, z, Rx, Ry, Rz).
    for dof, bc in variable_constraints:
        e_rb = etree.Element(BODY_COND_NAME)
        k = ("variable", bc["variable"])
        e_rb.attrib["type"] = BC_TYPE_TAG["body"][k]
        etree.SubElement(e_rb, "rb").text = str(mat_id + 1)
        etree.SubElement(e_rb, "dof").text = XML_RB_DOF_FROM_DOF[dof]
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


def mesh_xml(model, domains, material_registry):
    """Create <Mesh> and <MeshDomain> XML elements.

    <Geometry> became <Mesh> in FEBio XML 3.0

    """
    e_geometry = etree.Element(MESH_TAG)
    e_meshdomains = etree.Element("MeshDomains")
    # Write <nodes>
    e_nodes = etree.SubElement(e_geometry, "Nodes")
    for i, x in enumerate(model.mesh.nodes):
        feb_nid = i + 1  # 1-indexed
        e = etree.SubElement(e_nodes, "node", id="{}".format(feb_nid))
        e.text = vec_to_text(x)
        e_nodes.append(e)
    # Write <Elements> and <SolidDomain> for each domain
    for i, domain in enumerate(domains):
        # <Elements>
        e_elements = etree.SubElement(e_geometry, "Elements", name=domain["name"])
        e_elements.attrib["type"] = domain["element_type"].feb_name
        for i, e in domain["elements"]:
            e_element = etree.SubElement(e_elements, "elem")
            e_element.attrib["id"] = str(i + 1)
            e_element.text = ", ".join(str(i + 1) for i in e.ids)
        # <SolidDomain>
        if domain["material"] is not None:
            mat_name = material_registry.names(domain["material"], "canonical")[0]
            e_soliddomain = etree.SubElement(
                e_meshdomains, "SolidDomain", name=domain["name"], mat=mat_name
            )
    return e_geometry, e_meshdomains


def node_data_xml(nodes, data, data_name, nodeset_name):
    """Construct NodeData XML element"""
    e_NodeData = etree.Element("NodeData")
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
    # <Boundary><bc type="fix" node_set="set_name">

    e_bcs = []
    # FEBio XML 3.0 stores the nodal BCs by node set
    by_nodeset = bcs_by_nodeset_and_var(fixed_conditions)
    for nodeset, dofs_by_var in by_nodeset.items():
        # Get or create a name for the node set
        nodeset_name = "fixed_" + "_".join(
            [f"{'_'.join(dofs)}_{var}" for var, dofs in dofs_by_var.items()]
        )
        # Name may be modified for disambiguation
        nodeset_name = nodeset_registry.get_or_create_name(nodeset_name, nodeset)
        for var, dofs in dofs_by_var.items():
            # Create element
            e_bc = etree.Element(
                "bc", type=BC_TYPE_TAG["node"]["fixed"], node_set=nodeset_name
            )
            txt = ",".join(XML_BC_FROM_DOF[(dof, var)] for dof in dofs)
            etree.SubElement(e_bc, "dofs").text = txt
            e_bcs.append(e_bc)
    return e_bcs


def xml_node_var_bc(model, xmlroot, nodes, scales, seq, dof, var, relative, step_name):
    """Return XML elements for nodes variable displacement

    model := Model object.  Needed for the name registry.

    Returns tuple of (<bc> element, <NodeData> element)

    """
    # Hierarchy: <Boundary><bc type="prescribe" node_set="set_name">
    e_bc = etree.Element("bc", type=BC_TYPE_TAG["node"]["variable"])
    e_dof = etree.SubElement(e_bc, "dof").text = XML_BC_FROM_DOF[(dof, var)]
    seq_id = get_or_create_seq_id(model.named["sequences"], seq)
    e_sc = etree.SubElement(e_bc, "scale", lc=str(seq_id + 1), type="map")
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
    e_loadcurve = etree.Element(
        "load_controller", id=str(sequence_id + 1), type="loadcurve"
    )
    etree.SubElement(e_loadcurve, "interpolate").text = XML_INTERP_FROM_INTERP[
        sequence.interpolant
    ]
    etree.SubElement(e_loadcurve, "extend").text = XML_EXTRAP_FROM_EXTRAP[
        sequence.extrapolant
    ]
    e_points = etree.SubElement(e_loadcurve, "points")
    for pt in sequence.points:
        etree.SubElement(e_points, "point").text = f"{pt[0] + t0}, {pt[1]}"
    return e_loadcurve


def surface_pair_xml(faceset_registry, primary, secondary, name):
    """Return SurfacePair XML element.

    The surfaces (face sets) involved in the surface pair must already have names in
    `faceset_registry`.

    """
    e_surfpair = etree.Element("SurfacePair", name=name)
    etree.SubElement(e_surfpair, "primary").text = faceset_registry.names(primary)[0]
    etree.SubElement(e_surfpair, "secondary").text = faceset_registry.names(secondary)[
        0
    ]
    return e_surfpair


def step_xml_factory():
    """Create empty <step> XML elements"""
    i = 1
    while True:
        e = etree.Element(STEP_NAME, id=str(i), name=f"Step{i}")
        yield e


def xml_rigid_nodeset_bc(name: str, material_name: str = None, material_id: int = None):
    """Return XML element for a rigid node set (implicit rigid body)

    :param name: Name of node set to be treated as rigid.

    :param material_name: Name of rigid material corresponding to this rigid node set.
    Not needed in FEBio XML 3.0; included only for call signature compatibility.

    :param material_id: Ordinal ID (in FEBio XML; 1-indexed) of rigid material
    corresponding to this rigid node set.

    """
    if material_id is None:
        raise ValueError("Must provide material_id.")
    e = etree.Element("bc")
    e.attrib["type"] = "rigid"
    e.attrib["node_set"] = name
    etree.SubElement(e, "rb").text = str(material_id)
    return e


def xml_dynamics(dynamics: Dynamics, physics):
    """Return <analysis> element"""
    e = etree.Element("analysis")
    e.text = DYNAMICS_TO_XML[(physics, dynamics)]
    return e


def xml_qnmethod(solver):
    """Convert Solver.update_method to XML"""
    conv = {"BFGS": "0", "Broyden": "1", "Newton": "0"}
    # ^ you only actually get Newton iterations if max_ups = 0
    return const_property_to_xml(conv[solver.update_method], "qnmethod")
