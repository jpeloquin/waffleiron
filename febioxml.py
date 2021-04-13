from collections import namedtuple
import os
from pathlib import Path

import lxml.etree as ET
from lxml.etree import Element, ElementTree
from .core import (
    Body,
    ImplicitBody,
    ContactConstraint,
    NodeSet,
    FaceSet,
    ElementSet,
    _canonical_face,
    Sequence,
    ScaledSequence,
    Interpolant,
    Extrapolant,
)
from .control import Physics
from .element import Quad4, Tri3, Hex8, Penta6, Element
from . import material
from .math import orthonormal_basis


# Helper classes

OptParameter = namedtuple("OptParameter", ["path", "fun", "default"])
ReqParameter = namedtuple("ReqParameter", ["path", "fun"])


# Functions for traversing a Model in a way that facilitates XML read
# or write.


def domains(model):
    """Return list of domains.

    Here, a domain is defined as the collection of all elements of the
    same type with the same material.

    """
    # TODO: Modify the definition of parts such that 1 part = all
    # *connected* elements with the same material.
    #
    # Assemble elements into blocks with like type and material.
    # Elemsets uses material instances as keys.  Each item is a
    # dictionary using element classes as keys, with items being tuples
    # of (element_id, element).
    by_mat_type = {}
    for i, elem in enumerate(model.mesh.elements):
        subdict = by_mat_type.setdefault(elem.material, {})
        like_elements = subdict.setdefault(elem.__class__, [])
        like_elements.append((i, elem))
    # Convert nested dictionaries to a list
    domains = []
    i = 0
    for mat in by_mat_type:
        for typ in by_mat_type[mat]:
            i += 1
            domains.append(
                {
                    "name": f"Domain{i}",
                    "material": mat,
                    "element_type": typ,
                    "elements": by_mat_type[mat][typ],
                }
            )
    return domains


# Functions for reading FEBio XML


def find_unique_tag(root: Element, path):
    """Find and return a tag or an error if > 1 of same."""
    tags = root.findall(path)
    if len(tags) == 1:
        return tags[0]
    elif len(tags) > 1:
        raise ValueError(
            f"Multiple `{path}` tags in file `{os.path.abspath(root.base)}`"
        )
    else:
        return None


def read_parameter(e, sequence_registry):
    """Read a parameter from an XML element.

    The parameter may be fixed or variable.  If variable, a Sequence or
    ScaledSequence will be returned.

    """
    # Check if this is a time-varying or fixed property
    if "lc" in e.attrib:
        # The property is time-varying
        seq_id = int(e.attrib["lc"]) - 1
        sequence = sequence_registry.obj(seq_id, "ordinal_id")
        if e.text is not None and e.text.strip() != "":
            scale = to_number(e.text)
            return ScaledSequence(sequence, scale)
        else:
            return sequence
    else:
        # The property is fixed
        return to_number(e.text)


def read_parameters(xml, paramdict):
    """Return parameters for a dataclass's fields from an XML element"""
    params = {}
    for k, p in paramdict.items():
        if isinstance(p, ReqParameter):
            e = find_unique_tag(xml, p.path)
            if e is None:
                fullpath = "/".join((xml.getroottree().getpath(e), p.path))
                raise ValueError(
                    f"Required XML element '{fullpath}' was not found in '{xml.base}'."
                )
            params[k] = p.fun(e.text)
        else:  # Optional parameter
            e = xml.findall(p.path)
            if len(e) == 0:
                # Use default
                params[k] = p.default
            elif len(e) > 1:
                parentpath = xml.getroottree().getpath(e.getparent())
                raise ValueError(
                    f"{xml.base}:{xml.sourceline} {parentpath} has {len(e)} {e.tag} elements.  It should have at most one."
                )
            else:  # len(s) == 1
                e = e[0]
                params[k] = p.fun(e.text)
    return params


def read_point(text):
    x, y = text.split(",")
    return to_number(x), to_number(y)


def basis_mat_axis_local(element: Element, local_ids=(1, 2, 4)):
    """Return element basis for FEBio XML <mat_axis type="local"> values.

    element is an Element object.

    mat_axis_local is a tuple of 3 element-local node IDs (1-indexed).
    The default value is (1, 2, 4) to match FEBio.  FEBio /treats/ (0,
    0, 0) as equal to (1, 2, 4), so this function does the same.

    """
    # FEBio special-case
    if local_ids == (0, 0, 0):
        local_ids = (1, 2, 4)
    a = element.nodes[local_ids[1] - 1] - element.nodes[local_ids[0] - 1]
    d = element.nodes[local_ids[2] - 1] - element.nodes[local_ids[0] - 1]
    basis = orthonormal_basis(a, d)
    return basis


def logfile_name(root) -> Path:
    """Return logfile path from FEBio XML"""
    paths = [
        e.attrib["file"] for e in root.findall("Output/logfile") if "file" in e.attrib
    ]
    if len(paths) == 0:
        # The default log file name is based on the FEBio XML file
        # name.   If the XML has not yet been written to disk,
        # there is no valid default.
        if root.base is None:
            raise TypeError(
                f"The FEBio XML tree has no file name associated with it, so the default log file name is undefined.  (The XML tree was probably created without reading from a file.)"
            )
        return Path(root.base).with_suffix(".log")
    elif len(paths) == 1:
        return paths[0]
    else:  # len(paths) > 1
        raise ValueError(
            f"{root} (root.base) has more than one `<logfile>` element with a `file` attribute, making the logfile name ambiguous."
        )


def normalize_xml(root):
    """Convert some items in FEBio XML to 'normal' representation.

    FEBio XML allows some items to be specified several ways.  To reduce
    the complexity of the code that converts FEBio XML to a febtools
    Model, this function should be used ahead of time to normalize the
    representation of said items.

    Specific normalizations:

    - When a bare <Control> element exists, wrap it in a <Step> element.

    - When a bare <Boundary> element exists, wrap it in a <Step> element.

    - [TODO] Convert <mat_axis type="local">0,0,0</mat_axis> to the
      default value of 1,2,4.

    This function also does some validation.

    """
    # Validation: At most one of <Control> or <Step> should exist
    if root.find("Control") is not None and root.find("Step") is not None:
        msg = f"{root.base} has both a <Control> and <Step> section. The FEBio documentation does not specify how these sections are supposed to interact, so normalization is aborted."
        raise ValueError(msg)
    #
    # Normalization: When a bare <Control> element exists, wrap it in a
    # <Step> element.
    if root.find("Control") is not None:
        e_Control = root.find("Control")
        # From validation above, we know that no <Step> element exists,
        # so we need to create one.
        e_Step = ET.Element("Step")
        e_Control.getparent().remove(e_Control)
        root.insert(1, e_Step)
        e_Step.append(e_Control)
    #
    # Normalization: When a bare <Boundary> element exists, wrap any
    # <Boundary>/<prescribe> elements in the first <Step> element.
    e_rBoundary = root.find("Boundary")
    if e_rBoundary is not None:
        e_Step = root.find("Step")
        if e_Step is None:
            e_Step = ET.Element("Step")
            root.insert(1, e_Step)
        es_prescribe = e_rBoundary.findall("prescribe")
        # Do we need to create a Step/Boundary element?
        e_sBoundary = e_Step.find("Boundary")
        if len(es_prescribe) != 0 and e_sBoundary is None:
            e_sBoundary = ET.SubElement(e_Step, "Boundary")
        # Move the <prescribe> elements
        for e_prescribe in es_prescribe:
            e_rBoundary.remove(e_prescribe)
            e_sBoundary.append(e_prescribe)
        # Delete the <Boundary> element if it is now empty
        e_rBoundary = root.find("Boundary")
        if len(e_rBoundary) == 0:
            root.remove(e_rBoundary)
    return root


# Functions for writing FEBio XML


def body_mat_id(body, material_registry, implicit_rb_mats):
    """Return a material ID to define a rigid body in XML."""
    # Create or find the associated materials
    if isinstance(body, Body):
        # If an explicit body, its elements define its
        # materials.  We assume that the body is homogenous.
        mat = body.elements[0].material
        mat_id = material_registry.names(mat, nametype="ordinal_id")[0]
    elif isinstance(body, ImplicitBody):
        mat = implicit_rb_mat[body]
        mat_id = material_registry.name(mat, nametype="ordinal_id")
    else:
        msg = (
            f"body {k} does not have a supported type.  "
            + "Supported body types are Body and ImplicitBody."
        )
        raise ValueError(msg)
    return mat_id


def get_or_create_xml(root, path):
    """Return XML element at path, creating it if needed"""
    parts = [p for p in path.split("/") if p != ""]
    parent = root
    for part in parts:
        current = find_unique_tag(parent, part)
        if current is None:
            current = ET.SubElement(parent, part)
        parent = current
    return current


def get_or_create_parent(root, path):
    """Return second-to-last XML element from path, creating it if needed"""
    path = "/".join(path.split("/")[:-1])
    return get_or_create_xml(root, path)


def get_or_create_item_id(registry, item):
    """Get or create ID for an item.

    Getting or creating an ID for an item is complicated because item
    IDs must start at 0 and be sequential and contiguous.

    """
    item_ids = registry.namespace("ordinal_id")
    if len(item_ids) == 0:
        # Handle the trivial case of no pre-existing items
        item_id = 0
        # Create the ID
        registry.add(item_id, item, "ordinal_id")
    else:
        # At least one item already exists.  Make sure the ID
        # constraints have not been violated
        assert min(item_ids) == 0
        assert max(item_ids) == len(item_ids) - 1
        # Check for an existing ID
        try:
            item_id = registry.names(item, "ordinal_id")[0]
        except KeyError:
            # Create an ID because the item doesn't have one
            item_ids = registry.namespace("ordinal_id")
            item_id = max(item_ids) + 1
            registry.add(item_id, item, "ordinal_id")
    return item_id


def get_or_create_seq_id(registry, sequence):
    """Return ID for a Sequence, creating it if needed.

    The returned ID refers to the underlying Sequence object, never to a
    ScaledSequence.

    """
    if type(sequence) is ScaledSequence:
        sequence = sequence.sequence
    return get_or_create_item_id(registry, sequence)


def text_to_bool(s):
    """Convert string to boolean"""
    if not s in ("0", "1"):
        raise ValueError(
            f"Cannot convert '{s}' to boolean.  FEBio boolean flags should be '0' or '1'."
        )
    return s == "1"


def to_number(s):
    """Convert numeric string to int or float as appropriate."""
    try:
        return int(s)
    except ValueError:
        return float(s)


def maybe_to_number(s):
    """Convert string to number if possible, otherwise return string."""
    try:
        return to_number(s)
    except ValueError:
        return s


def to_text(v):
    """Serialize value to text by type"""
    if isinstance(v, str):
        return v
    elif isinstance(v, bool):
        return bool_to_text(v)
    else:
        return num_to_text(v)


def bool_to_text(v):
    return "1" if v else "0"


def int_to_text(v):
    return str(v)


def num_to_text(v):
    """Serialize numeric value to text by type"""
    if isinstance(v, int):
        return int_to_text(v)
    elif isinstance(v, float):
        return float_to_text(v)
    else:
        raise ValueError(
            f"Provided numeric value has type '{type(v).__name__}', which is not supported for conversion to XML."
        )


def vec_to_text(v):
    return ", ".join(f"{a:.7e}" for a in v)


def bvec_to_text(v):
    return ", ".join(float_to_text(a) for a in v)


def float_to_text(a):
    return f"{a:.7g}"


def property_to_xml(value, tag, seq_registry):
    """Convert a constant or variable property to FEBio XML"""
    if isinstance(value, Sequence):
        # Time-varying property, not scaled
        e = ET.Element(tag)
        seq_id = get_or_create_item_id(seq_registry, value)
        e.attrib["lc"] = str(seq_id + 1)
        e.text = "1"  # scale factor
    elif isinstance(value, ScaledSequence):
        # Time-varying property, scaled
        e = ET.Element(tag)
        seq_id = get_or_create_item_id(seq_registry, value.sequence)
        e.attrib["lc"] = str(seq_id + 1)
        e.text = num_to_text(value.scale)
    else:
        # Constant property
        e = const_property_to_xml(value, tag)
    return e


def const_property_to_xml(value, tag):
    """Convert a constant property to FEBio XML"""
    e = ET.Element(tag)
    e.text = to_text(value)
    return e


def update_method_to_xml(value, tag):
    """Convert Solver.update_method to XML"""
    conv = {"BFGS": "0", "Broyden": "1"}
    return const_property_to_xml(conv[value], tag)


# Facts about FEBio XML.  Define this after the function definitions so
# that the functions can be referenced here.

SUPPORTED_FEBIO_XML_VERS = ("2.0", "2.5", "3.0")

SEQUENCE_PARENT = "LoadData"

# Map "bc" attribute value from <prescribe>, <prescribed>, <fix>, or
# <fixed> element to a variable name.  This list is valid for both node
# and rigid body conditions.  FEBio handles force conditions in other
# XML elements: for rigid bodies, <force>, and for nodes, <nodal_load>.
VAR_FROM_XML_NODE_BC = {
    "x": "displacement",
    "y": "displacement",
    "z": "displacement",
    "Rx": "rotation",
    "Ry": "rotation",
    "Rz": "rotation",
    "p": "pressure",
}
# Map "bc" attribute value from <prescribe>, <prescribed>,
# <fix>, or <fixed> element to a degree of freedom.
DOF_NAME_FROM_XML_NODE_BC = {
    "x": "x1",
    "y": "x2",
    "z": "x3",
    "Rx": "α1",
    "Ry": "α2",
    "Rz": "α3",
    "p": "fluid",
}

XML_BC_FROM_DOF = {
    (dof, VAR_FROM_XML_NODE_BC[tag]): tag
    for tag, dof in DOF_NAME_FROM_XML_NODE_BC.items()
}
XML_BC_FROM_DOF.update(
    {("x1", "force"): "x", ("x2", "force"): "y", ("x3", "force"): "z"}
)

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

elem_cls_from_feb = {"quad4": Quad4, "tri3": Tri3, "hex8": Hex8, "penta6": Penta6}

solid_class_from_name = {
    "isotropic elastic": material.IsotropicElastic,
    "Holmes-Mow": material.HolmesMow,
    "fiber-exp-pow": material.ExponentialFiber,
    "fiber-pow-linear": material.PowerLinearFiber,
    "neo-Hookean": material.NeoHookean,
    "solid mixture": material.SolidMixture,
    "rigid body": material.RigidBody,
    "biphasic": material.PoroelasticSolid,
    "Donnan equilibrium": material.DonnanSwelling,
    "multigeneration": material.Multigeneration,
    "orthotropic elastic": material.OrthotropicElastic,
}
solid_name_from_class = {v: k for k, v in solid_class_from_name.items()}

perm_class_from_name = {
    "perm-Holmes-Mow": material.IsotropicHolmesMowPermeability,
    "perm-const-iso": material.IsotropicConstantPermeability,
}
perm_name_from_class = {v: k for k, v in perm_class_from_name.items()}

# TODO: Redesign the compatibility system so that compatibility can be
# derived from the material's type.
physics_compat_by_mat = {
    material.PoroelasticSolid: {Physics.BIPHASIC},
    material.RigidBody: {Physics.SOLID, Physics.BIPHASIC},
    material.OrthotropicElastic: {Physics.SOLID, Physics.BIPHASIC},
    material.IsotropicElastic: {Physics.SOLID, Physics.BIPHASIC},
    material.SolidMixture: {Physics.SOLID, Physics.BIPHASIC},
    material.PowerLinearFiber: {Physics.SOLID, Physics.BIPHASIC},
    material.ExponentialFiber: {Physics.SOLID, Physics.BIPHASIC},
    material.HolmesMow: {Physics.SOLID, Physics.BIPHASIC},
    material.NeoHookean: {Physics.SOLID, Physics.BIPHASIC},
}

# Map of ContactConstraint fields → elements relative to <contact>
CONTACT_PARAMS = {
    "tension": OptParameter("tension", text_to_bool, False),
    "penalty_factor": OptParameter("penalty", to_number, 1),
    "auto_adjust_penalty": OptParameter("auto_penalty", text_to_bool, False),
    "symmetric_stiffness": OptParameter("symmetric_stiffness", text_to_bool, False),
    "use_augmented_lagrange": OptParameter("laugon", text_to_bool, False),
    "augmented_lagrange_rtol": OptParameter("tolerance", to_number, 1.0),
    "augmented_lagrange_gapnorm_atol": OptParameter("gaptol", maybe_to_number, 0.0),
    "augmented_lagrange_minaug": OptParameter("minaug", int, 0),
    "augmented_lagrange_maxaug": OptParameter("maxaug", int, 10),
    "search_scale": OptParameter("search_radius", to_number, 1.0),
    "projection_tol": OptParameter("search_tol", to_number, 0.01),
}
