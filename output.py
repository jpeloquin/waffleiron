# Base packages
from collections import defaultdict
from copy import copy, deepcopy
from math import degrees
from datetime import datetime

# Public packages
import numpy as np
from lxml import etree as ET

# Within-module packages
import febtools as feb
from .core import (
    Body,
    ImplicitBody,
    ContactConstraint,
    NameRegistry,
    NodeSet,
    Sequence,
    ScaledSequence,
    RigidInterface,
)
from .control import step_duration
from . import material as matlib
from .math import sph_from_vec
from . import febioxml
from . import febioxml_2_0
from . import febioxml_2_5
from . import febioxml_3_0
from .febioxml import (
    bool_to_text,
    float_to_text,
    vec_to_text,
    bvec_to_text,
    control_tagnames_to_febio,
    control_values_to_febio,
    TAG_FROM_BC,
    DOF_NAME_FROM_XML_NODE_BC,
    XML_BC_FROM_DOF,
    VAR_FROM_XML_NODE_BC,
)

# ^ The intent here is to eventually be able to switch between FEBio XML
# formats by exchanging this import statement for a different version.
# Common functionality can be shared between febioxml_*_*.py files via
# imports.


def default_febio_config():
    """Return default FEBio settings"""
    return {"output variables": ["displacement", "stress"]}


def _get_or_create_item_id(registry, item):
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


def _get_or_create_seq_id(registry, sequence):
    """Return ID for a Sequence, creating it if needed.

    The returned ID refers to the underlying Sequence object, never to a
    ScaledSequence.

    """
    if type(sequence) is ScaledSequence:
        sequence = sequence.sequence
    return _get_or_create_item_id(registry, sequence)


def _fixup_ordinal_ids(registry):
    """Regenerate ordinal IDs to satisfy invariants"""
    # # If no ordinal IDs, nothing to fix
    # if not "ordinal_id" in registry.nametypes():
    #     return
    # If ordinal IDs exist, make sure each is unique
    items = [item for i, item in sorted(registry.pairs("ordinal_id"))]
    for i, item in enumerate(items):
        registry.add(i, item, nametype="ordinal_id")
    if len(registry.namespace("ordinal_id")) > 0:
        assert min(registry.namespace("ordinal_id")) == 0
        assert (
            max(registry.namespace("ordinal_id"))
            == len(registry.namespace("ordinal_id")) - 1
        )


def _property_to_feb(p, tag, model):
    """Convert a fixed or variable property to FEBio XML."""
    e = ET.Element(tag)
    if isinstance(p, Sequence):
        seq_id = _get_or_create_item_id(model.named["sequences"], p)
        e.attrib["lc"] = str(seq_id + 1)
        e.text = "1"  # basic Sequences have no scale
    elif isinstance(p, ScaledSequence):
        # Time-varying property, scaled
        seq_id = _get_or_create_item_id(model.named["sequences"], p.sequence)
        e.attrib["lc"] = str(seq_id + 1)
        e.text = float_to_text(p.scale)
    else:
        # Fixed property
        e.text = float_to_text(p)
    return e


def exponential_fiber_to_feb(mat, model):
    """Convert ExponentialFiber material instance to FEBio XML."""
    e = ET.Element("material", type="fiber-exp-pow")
    e.append(_property_to_feb(mat.α, "alpha", model))
    e.append(_property_to_feb(mat.β, "beta", model))
    e.append(_property_to_feb(mat.ξ, "ksi", model))
    return e


def power_linear_fiber_to_feb(mat, model):
    """Convert PowerLinearFiber material instance to FEBio XML."""
    e = ET.Element("material", type="fiber-pow-linear")
    e.append(_property_to_feb(mat.E, "E", model))
    e.append(_property_to_feb(mat.β, "beta", model))
    e.append(_property_to_feb(mat.λ0, "lam0", model))
    return e


def holmesmow_to_feb(mat, model):
    """Convert HolmesMow material instance to FEBio XML."""
    e = ET.Element("material", type="Holmes-Mow")
    e.append(_property_to_feb(mat.E, "E", model))
    e.append(_property_to_feb(mat.ν, "v", model))
    e.append(_property_to_feb(mat.β, "beta", model))
    return e


def isotropicelastic_to_feb(mat, model):
    """Convert IsotropicElastic material instance to FEBio XML."""
    e = ET.Element("material", type="isotropic elastic")
    E, ν = feb.material.from_Lamé(mat.y, mat.mu)
    e.append(_property_to_feb(E, "E", model))
    e.append(_property_to_feb(ν, "v", model))
    return e


def orthotropic_elastic_to_feb(mat, model):
    """Convert OrthotropicElastic material instance to FEBio XML."""
    e = ET.Element("material", type="orthotropic elastic")
    # Material properties
    e.append(_property_to_feb(mat.E1, "E1", model))
    e.append(_property_to_feb(mat.E2, "E2", model))
    e.append(_property_to_feb(mat.E3, "E3", model))
    e.append(_property_to_feb(mat.G12, "G12", model))
    e.append(_property_to_feb(mat.G23, "G23", model))
    e.append(_property_to_feb(mat.G31, "G31", model))
    e.append(_property_to_feb(mat.v12, "v12", model))
    e.append(_property_to_feb(mat.v23, "v23", model))
    e.append(_property_to_feb(mat.v31, "v31", model))
    return e


def neo_hookean_to_feb(mat, model):
    """Convert NeoHookean material instance to FEBio XML."""
    e = ET.Element("material", type="neo-Hookean")
    E, ν = feb.material.from_Lamé(mat.y, mat.mu)
    e.append(_property_to_feb(E, "E", model))
    e.append(_property_to_feb(ν, "v", model))
    return e


def iso_const_perm_to_feb(mat, model):
    """Convert IsotropicConstantPermeability instance to FEBio XML"""
    e = ET.Element("permeability", type="perm-const-iso")
    e.append(_property_to_feb(mat.k, "perm", model))
    return e


def iso_holmes_mow_perm_to_feb(mat, model):
    """Convert IsotropicHolmesMowPermeability instance to FEBio XML"""
    e = ET.Element("permeability", type="perm-Holmes-Mow")
    e.append(_property_to_feb(mat.k0, "perm", model))
    e.append(_property_to_feb(mat.M, "M", model))
    e.append(_property_to_feb(mat.α, "alpha", model))
    return e


def poroelastic_to_feb(mat, model):
    """Convert Poroelastic material instance to FEBio XML"""
    e = ET.Element("material", type="biphasic")
    e.append(_property_to_feb(mat.solid_fraction, "phi0", model))
    # Add solid material
    e_solid = material_to_feb(mat.solid_material, model)
    e_solid.tag = "solid"
    e.append(e_solid)
    # Add permeability
    typ = febioxml.perm_name_from_class[type(mat.permeability)]
    f = {
        feb.material.IsotropicConstantPermeability: iso_const_perm_to_feb,
        feb.material.IsotropicHolmesMowPermeability: iso_holmes_mow_perm_to_feb,
    }
    e_permeability = f[type(mat.permeability)](mat.permeability, model)
    e.append(e_permeability)
    return e


def solidmixture_to_feb(mat, model):
    """Convert SolidMixture material instance to FEBio XML."""
    e = ET.Element("material", type="solid mixture")
    for submat in mat.materials:
        m = material_to_feb(submat, model)
        m.tag = "solid"
        e.append(m)
    return e


def multigeneration_to_feb(mat, model):
    """Convert Multigeneration material instance to FEBio XML."""
    e = ET.Element("material", type="multigeneration")
    i = 1
    for t, submat in zip(mat.generation_times, mat.materials):
        e_generation = ET.SubElement(e, "generation")
        e_generation.attrib["id"] = str(i)
        i += 1
        ET.SubElement(e_generation, "start_time").text = str(t)
        e_submat = material_to_feb(submat, model)
        e_submat.tag = "solid"
        e_generation.append(e_submat)
    return e


def rigid_body_to_feb(mat, model):
    """Convert SolidMixture material instance to FEBio XML."""
    e = ET.Element("material", type="rigid body")
    if mat.density is None:
        density = 1
    else:
        density = mat.density
    e.append(_property_to_feb(density, "density", model))
    return e


def donnan_to_feb(mat, model):
    """Convert DonnanSwelling material instance to FEBio XML."""
    e = ET.Element("material", type="Donnan equilibrium")
    e.append(_property_to_feb(mat.phi0_w, "phiw0", model))
    e.append(_property_to_feb(mat.fcd0, "cF0", model))
    e.append(_property_to_feb(mat.ext_osm, "bosm", model))
    e.append(_property_to_feb(mat.osm_coef, "Phi", model))
    return e


def material_to_feb(mat, model):
    """Convert a material instance to FEBio XML."""
    if isinstance(mat, feb.material.OrientedMaterial):
        orientation = mat.orientation
        mat = mat.material
    else:
        orientation = None
    if mat is None:
        e = ET.Element("material", type="unknown")
    else:
        f = {
            feb.material.ExponentialFiber: exponential_fiber_to_feb,
            feb.material.PowerLinearFiber: power_linear_fiber_to_feb,
            feb.material.HolmesMow: holmesmow_to_feb,
            feb.material.IsotropicElastic: isotropicelastic_to_feb,
            feb.material.NeoHookean: neo_hookean_to_feb,
            feb.material.OrthotropicElastic: orthotropic_elastic_to_feb,
            feb.material.PoroelasticSolid: poroelastic_to_feb,
            feb.material.SolidMixture: solidmixture_to_feb,
            feb.material.RigidBody: rigid_body_to_feb,
            feb.material.DonnanSwelling: donnan_to_feb,
            feb.material.Multigeneration: multigeneration_to_feb,
        }
        try:
            e = f[type(mat)](mat, model)
        except ValueError:
            msg = "{} not implemented for conversion to FEBio XML."
            raise
    # Add material coordinate system if it is defined for this material.
    # Any mixture material /should/ call `material_to_feb` (this
    # function) for each sub-material, so we shouldn't need to handle
    # material coordinate systems anywhwere else.
    if orientation is not None:
        if np.array(orientation).ndim == 2:
            # material axes orientation
            e_mat_axis = ET.Element("mat_axis", type="vector")
            ET.SubElement(e_mat_axis, "a").text = febioxml.bvec_to_text(
                orientation[:, 0]
            )
            ET.SubElement(e_mat_axis, "d").text = febioxml.bvec_to_text(
                orientation[:, 1]
            )
            e.insert(0, e_mat_axis)
            e.append(e_mat_axis)
        elif np.array(orientation).ndim == 1:
            # vector orientation
            e_vector = ET.Element("fiber", type="vector")
            e_vector.text = bvec_to_text(orientation)
            e.append(e_vector)
        else:
            raise ValueError(
                f"Rank {orientation.ndim} material orientation not supported.  Provided orientation was {orientation}."
            )
    return e


def add_nodeset(xml_root, name, nodes):
    """Add a named node set to FEBio XML."""
    e_geometry = xml_root.find("./Geometry")
    for existing in xml_root.xpath(f"Geometry/NodeSet[@name='{name}']"):
        existing.getparent().remove(existing)
    e_nodeset = ET.SubElement(e_geometry, "NodeSet", name=name)
    # Sort nodes to be user-friendly (humans often read .feb files) and,
    # more importantly, so that local IDs in NodeData elements (FEBio
    # XML 2.5) or mesh_data elements (FEBio XML 3.0) have a stable
    # relationship with actual node IDs.
    for node_id in sorted(nodes):
        ET.SubElement(e_nodeset, "node", id=str(node_id + 1))


def add_sequence(xml_root, model, sequence, t0=0):
    """Add a sequence (load curve) to a FEBio XML tree.

    So we need to sort them and ensure that the IDs are contiguous.

    xml_root !mutates! := Root object of XML tree.

    sequence := Sequence object.

    sequence_id := Integer ID (0-referenced) to use for the sequence
    element's "id" attribute.  The ID will be incremented by 1 to
    account for FEBio XML's use of 1-referenced IDs.

    """
    seq_id = model.named["sequences"].names(sequence, nametype="ordinal_id")[0]
    e_loaddata = xml_root.find("./LoadData")
    e_loadcurve = ET.SubElement(
        e_loaddata,
        "loadcurve",
        id=str(seq_id + 1),
        type=sequence.interpolant,
        extrap=sequence.extrapolant,
    )
    for pt in sequence.points:
        ET.SubElement(e_loadcurve, "point").text = f"{pt[0] + t0}, {pt[1]}"


def sequence_time_offsets(model):
    """Return map: sequence → global start time.

    In `febtools`, each step has its own running time (step-local time),
    and step-related time sequences are in step-local time.  But in
    FEBio XML, all time sequences are written in global time.  This
    function calculates and returns the time offsets that must be added
    to each sequence to convert said sequence from local time to global
    time.

    """
    cumulative_time = 0.0
    seq_t0 = defaultdict(lambda: 0)  # dict: sequence → time offset
    for step in model.steps:
        # Gather must point curves
        dtmax = step["control"]["time stepper"]["dtmax"]
        if isinstance(dtmax, Sequence):
            dtmax.points = [(cumulative_time + t, v) for t, v in dtmax.points]
        # Gather variable boundary condition / constraint curves
        curves_to_adjust = set([])
        if "bc" in step:
            for i, ax_bc in step["bc"]["node"].items():
                for ax, d in ax_bc.items():
                    if isinstance(d["sequence"], Sequence):
                        curves_to_adjust.add(d["sequence"])
                    elif isinstance(d["sequence"], ScaledSequence):
                        curves_to_adjust.add(d["sequence"].sequence)
        # Gather the body constraint curves
        if "bc" in step:
            for body, body_constraints in step["bc"]["body"].items():
                for ax, params in body_constraints.items():
                    # params = {'variable': variable <string>,
                    #           'sequence': Sequence object or 'fixed',
                    #           'scale': scale <numeric>
                    if isinstance(params["sequence"], Sequence):
                        curves_to_adjust.add(params["sequence"])
                    elif isinstance(params["sequence"], ScaledSequence):
                        curves_to_adjust.add(params["sequence"].sequence)
                    # TODO: Add test to exercise this code
        # Adjust the curves
        for curve in curves_to_adjust:
            seq_t0[curve] = cumulative_time
        # Tally running time
        duration = step_duration(step)
        cumulative_time += duration
    return seq_t0


def choose_module(materials):
    """Determine which module should be used to run the model.

    Currently only chooses between solid and biphasic.

    """
    module = "solid"
    for m in materials:
        if isinstance(m, feb.material.PoroelasticSolid):
            module = "biphasic"
    return module


def contact_section(contacts, model, named_surface_pairs, named_contacts):
    tag_contact_section = ET.Element("Contact")
    tags_surfpair = []
    for contact in contacts:
        tag_contact = ET.SubElement(
            tag_contact_section, "contact", type=contact.algorithm
        )
        # Name the contact to match its surface pair so someone reading
        # the XML can match them more easily
        tag_contact.attrib["name"] = named_contacts.get_or_create_name(
            f"contact_-_{contact.algorithm}", contact
        )
        # Autogenerate names for the face sets in the contact
        surface_name = {"leader": "", "follower": ""}
        for k, face_set in zip(
            ("leader", "follower"), (contact.leader, contact.follower)
        ):
            nm = model.named["face sets"].get_or_create_name(
                f"contact_surface_-_{contact.algorithm}",
                face_set,
            )
            surface_name[k] = nm
        # Autogenerate and add the surface pair
        name_surfpair = named_surface_pairs.get_or_create_name(
            f"contact_surfaces_-_{contact.algorithm}",
            (contact.leader, contact.follower),
        )
        tag_contact.attrib["surface_pair"] = name_surfpair
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
        ET.SubElement(tag_contact, "laugon").text = bool_to_text(
            contact.augmented_lagrange
        )
        ET.SubElement(tag_contact, "symmetric_stiffness").text = bool_to_text(
            contact.symmetric_stiffness
        )
        e_two_pass = ET.SubElement(tag_contact, "two_pass")
        if contact.passes == 2:
            e_two_pass.text = "1"
        elif contact.passes == 1:
            e_two_pass.text = "0"
        else:
            raise ValueError(
                f"{contact.passes} passes requested in a contact constraint; FEBio supports either 0 or 1."
            )
        # Write other parameters.  The FEBio manual is a bit spotty, so
        # extra contact parameters are stuffed in a dictionary.  Write
        # them out as-is, only casting them to strings.
        for k, v in contact.other_params.items():
            ET.SubElement(tag_contact, k).text = f"{v}"
    return tag_contact_section


def control_parameter_to_feb(parameter, value):
    """Return FEBio XML element for a control parameter."""
    nm_feb = control_tagnames_to_febio[parameter]
    if parameter in control_values_to_febio:
        val_feb = control_values_to_febio[parameter][value]
    else:
        if isinstance(value, bool):
            val_feb = bool_to_text(value)
        else:
            val_feb = str(value)
    e = ET.Element(nm_feb)
    if nm_feb == "analysis":
        # For some reason, <analysis> stores its value as an attribute,
        # whereas every other XML element stores its value as its value.
        e.attrib["type"] = val_feb
    else:
        e.text = val_feb
    return e


def body_constraints_to_feb(
    body, constraints, material_registry, implicit_rb_mat, sequence_registry
):
    """Return <rigid_body> element for a body's constraints dictionary.

    These <rigid_body> elements are used as children of a <Boundary>
    element to specify (in FEBio terms) rigid body boundary conditions.

    """
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
    # Create the XML tags for the rigid body BC
    e_body = ET.Element("rigid_body", mat=str(mat_id + 1))
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
            tagname = TAG_FROM_BC["body"][kind]
        elif bc["variable"] == "force":
            tagname = "force"
            if bc["relative"]:
                raise ValueError(
                    f"A relative body boundary condition for {dof} {bc['variable']} was requested, but relative body boundary conditions are supported only for displacement."
                )
        else:
            raise ValueError(f"Variable {bc['variable']} not supported for BCs.")
        bc_attr = XML_BC_FROM_DOF[(dof, bc["variable"])]
        e_bc = ET.SubElement(e_body, tagname, bc=bc_attr)
        if kind == "variable":
            seq_id = _get_or_create_seq_id(sequence_registry, seq)
            e_bc.attrib["lc"] = str(seq_id + 1)
            if bc["relative"]:
                e_bc.attrib["type"] = "relative"
            e_bc.text = str(v)
    return e_body


def xml(model, version="2.5"):
    """Convert a model to an FEBio XML tree.

    Creating an FEBio XML tree from a model is useful because it allows
    XML-editing trickery, if necessary, prior to writing the XML to an
    on-disk .feb file.

    """
    # Create dictionaries to keep track of named items
    named_surface_pairs = NameRegistry()
    named_contacts = NameRegistry()
    # Register all materials that are assigned to elements.  We do this
    # early because in FEBio XML the material ids are needed to define
    # the geometry and meshdata sections.  Technically, implicit bodies
    # also have a rigid material, but because the geometry is implicit
    # they will be added later as they are encountered when handling
    # boundary conditions.  (This means that all implicit rigid
    # materials are generated fresh on export and the old ones are
    # discarded, which is not necessarily desirable, and an area for
    # future improvement.)
    materials_used = set(e.material for e in model.mesh.elements)
    # Create a new dictionary of materials → material IDs.  This
    # dictionary will be updated as new materials are autogenerated
    # during model conversion to FEBio XML.  From this point on, it is
    # the canonical source of material IDs for the export process.
    material_registry = copy(model.named["materials"])
    # Workaround for FEBio quirk: Get rid of unused rigid body
    # materials.  FEBio 2.9.0, and possibly other versions, has a zero
    # diagonal error termination if an unreferenced rigid body material
    # is present.
    for mat in set(material_registry.objects()) - materials_used:
        if type(mat) is matlib.RigidBody:
            material_registry.remove_object(mat)
    _fixup_ordinal_ids(material_registry)
    # Ensure each material has an ID
    for mat in materials_used:
        _get_or_create_item_id(material_registry, mat)
    assert materials_used - set(material_registry.objects()) == set()

    root = ET.Element("febio_spec", version="{}".format(version))
    msg = f"Exported to FEBio XML by febtools prerelease at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S%z')}"
    root.append(ET.Comment(msg))

    version_major, version_minor = [int(a) for a in version.split(".")]
    if version_major == 2 and version_minor == 0:
        febioxml = febioxml_2_0
    elif version_major == 2 and version_minor == 5:
        febioxml = febioxml_2_5
    elif version_major == 3 and version_minor == 0:
        febioxml = febioxml_3_0
    else:
        raise NotImplementedError(
            f"Writing FEBio XML {version_major}.{version_minor} is not supported."
        )

    # Set solver module (analysis type)
    module = choose_module([m for m in material_registry.objects()])
    if version_major == 3 or (version_major == 2 and version_minor == 5):
        # In FEBio XML 2.5 and 3.0, <Module> must exist and be first tag
        e_module = ET.SubElement(root, "Module")
        e_module.attrib["type"] = module
    # FEBio XML 2.0 sets <analysis_type> in <Control>

    Material = ET.SubElement(root, "Material")

    parts = febioxml.parts(model)
    Geometry = febioxml.geometry_section(model, parts, material_registry)
    root.append(Geometry)

    e_boundary = ET.SubElement(root, "Boundary")
    if version_major == 2 and version_minor >= 5:
        contact_constraints = [
            c for c in model.constraints if isinstance(c, ContactConstraint)
        ]
        tag_contact = contact_section(
            contact_constraints, model, named_surface_pairs, named_contacts
        )
        root.append(tag_contact)
    else:
        tag_contact = febioxml_2_0.contact_section(model)
        root.append(tag_contact)
    e_constraints = ET.SubElement(root, "Constraints")
    e_loaddata = ET.SubElement(root, "LoadData")
    Output = ET.SubElement(root, "Output")

    # Typical MKS constants
    e_Constants = ET.Element("Constants")
    if "R" in model.constants:
        ET.SubElement(e_Constants, "R").text = str(model.constants["R"])
    if "temperature" in model.environment:
        ET.SubElement(e_Constants, "T").text = str(model.environment["temperature"])
    if "F" in model.constants:
        ET.SubElement(e_Constants, "Fc").text = str(model.constants["F"])
    # Add Globals/Constants if any defined; FEBio can't cope with an empty
    # Globals element.
    if len(e_Constants.getchildren()) > 0:
        e_Globals = ET.Element("Globals")
        e_Globals.append(e_Constants)
        root.insert(root.index(e_module) + 1, e_Globals)
        # ^ FEBio requires that first element must be <Module>

    # Materials section
    #
    # Make material tags for all materials assigned to elements.  Sort
    # by id to get around FEBio bug (FEBio ignores the ID attribute and
    # just uses tag order).
    materials = sorted(material_registry.pairs("ordinal_id"))
    for mat_id, mat in materials:
        tag = material_to_feb(mat, model)
        tag.attrib["id"] = str(mat_id + 1)
        # Name the material.  FEBio XML 3.0 requires each material to
        # have a name; in prior FEBio XML versions this is optional.
        name = material_registry.get_or_create_name("material", mat)
        tag.attrib["name"] = name
        Material.append(tag)
    # Assemble a list of all implicit rigid bodies used in the model.
    # There is currently no list of rigid bodies in the model or mesh
    # objects, so we have to search for them.  Rigid bodies may be
    # referenced in fixed constraints (model.fixed['body'][k] where k ∈
    # 'x1', 'x2', 'x3', 'α1', 'α2', 'α3') or in
    # model.steps[i]['bc']['body'] for each step i.
    implicit_bodies_to_process = set()  # memo
    # Search fixed constraints for rigid bodies
    for k in model.fixed["body"]:
        for body in model.fixed["body"][k]:
            if isinstance(body, ImplicitBody):
                implicit_bodies_to_process.add(body)
    # Search steps' constraints for rigid bodies
    for step in model.steps:
        if "bc" in step:
            for body in step["bc"]["body"]:
                if isinstance(body, ImplicitBody):
                    implicit_bodies_to_process.add(body)
    # Create FEBio rigid materials for all implicit rigid bodies and add
    # their rigid interfaces with the mesh.  That the implicit material
    # is rigid is an assumption, but an implicit deformable material in
    # FEA wouldn't make any sense.
    implicit_rigid_material_by_body = {}
    for i, implicit_body in enumerate(implicit_bodies_to_process):
        body_name = f"implicit_rigid_body_{i+1}"
        # Create the implicit body's FEBio rigid material
        mat = feb.material.RigidBody()
        tag = material_to_feb(mat, model)
        # TODO: Support comments in reader
        # tag.append(ET.Comment("Implicit rigid body"))
        mat_id = len(Material)
        mat_name = body_name + "_psuedo-material"
        tag.attrib["id"] = str(mat_id + 1)
        tag.attrib["name"] = mat_name
        Material.append(tag)
        # Update material registries
        material_registry.add(mat_name, mat)
        material_registry.add(mat_id, mat, nametype="ordinal_id")
        implicit_rigid_material_by_body[implicit_body] = mat
        #
        # Add the implicit body's rigid interface with the mesh.
        # Assumes interface is a node set.
        if version == "2.0":
            # FEBio XML 2.0 puts rigid bodies under §Constraints
            e_interface = ET.SubElement(e_contact, "contact", type="rigid")
            for i in implicit_body.interface:
                ET.SubElement(e_interface, "node", id=str(i + 1), rb=str(mat_id + 1))
        elif version_major == 2 and version_minor >= 5:
            # FEBio XML 2.5 puts rigid bodies under §Boundary
            try:
                name = model.named["node sets"].name(implicit_body.interface)
            except KeyError:
                name_base = f"{body_name}_interface"
                nodeset = implicit_body.interface
                name = model.named["node sets"].get_or_create_name(name_base, nodeset)
            add_nodeset(root, name, implicit_body.interface)
            ET.SubElement(e_boundary, "rigid", rb=str(mat_id + 1), node_set=name)

    # Write global <Boundary> element
    #
    # Write interfaces.  Currently just rigid interfaces.
    for interface in model.constraints:
        if type(interface) is not RigidInterface:
            continue
        rigid_body_id = material_registry.names(
            interface.rigid_body, nametype="ordinal_id"
        )[0]
        node_set_name = model.named["node sets"].name(interface.node_set)
        add_nodeset(root, node_set_name, interface.node_set)
        ET.SubElement(
            e_Boundary, "rigid", rb=str(rigid_body_id + 1), node_set=node_set_name
        )
    #
    # Write fixed nodal constraints to global <Boundary>
    for (dof, var), nodeset in model.fixed["node"].items():
        if nodeset:
            if version == "2.0":
                # Tag heirarchy: <Boundary><fix bc="x"><node id="1"> for each node
                e_fixed_nodeset = ET.SubElement(
                    e_boundary, "fix", bc=XML_BC_FROM_DOF[(dof, var)]
                )
                for i in nodeset:
                    ET.SubElement(e_fixed_nodeset, "node", id=str(i + 1))
            elif version_major == 2 and version_minor >= 5:
                # Tag heirarchy: <Boundary><fix bc="x" node_set="set_name">
                name_base = f"fixed_{dof}_autogen-nodeset"
                nodeset = NodeSet(nodeset)  # make hashable
                name = model.named["node sets"].get_or_create_name(name_base, nodeset)
                add_nodeset(root, name, nodeset)
                # Create the tag
                ET.SubElement(
                    e_boundary, "fix", bc=XML_BC_FROM_DOF[(dof, var)], node_set=name
                )
    #
    # Write fixed body constraints to global <Boundary>
    body_bcs = {}
    # Choose where to put rigid body constraints depending on FEBio XML
    # version.
    if version == "2.0":
        e_bc_body_parent = e_constraints
    elif version_major == 2 and version_minor == 5:
        e_bc_body_parent = e_boundary
    elif version_major == 3:
        e_bc_body_parent = e_boundary
    # Collect rigid body boundary conditions in a more convenient
    # hierarchy
    for dof, bodies in model.fixed["body"].items():
        for body in bodies:
            body_bcs.setdefault(body, set()).add(dof)
    # Create the tags specifying fixed constraints for the rigid bodies
    for body, axes in body_bcs.items():
        e_body = ET.SubElement(e_bc_body_parent, "rigid_body")
        # Assign the body's material
        if isinstance(body, Body):
            body_material = body.elements[0].material
            # TODO: ensure that body is all one material
        elif isinstance(body, ImplicitBody):
            body_material = implicit_rigid_material_by_body[body]
        mat_id = material_registry.names(body_material, nametype="ordinal_id")[0]
        e_body.attrib["mat"] = str(mat_id + 1)
        # Write tags for each of the fixed degrees of freedom.  Use
        # sorted order to be deterministic & human-friendly.
        for dof, var in sorted(axes):
            ET.SubElement(e_body, "fixed", bc=XML_BC_FROM_DOF[(dof, var)])
    #
    # TODO: Write time-varying nodal constraints to global <Boundary>
    #
    # Write time-varying rigid body constraints to global <Boundary>
    if version == "2.0":
        e_bc_body_parent = e_constraints
    elif version_major == 2 and version_minor >= 5:
        e_bc_body_parent = e_boundary
    for body, constraints in model.varying["body"].items():
        e_rb_new = body_constraints_to_feb(
            body,
            constraints,
            material_registry,
            implicit_rigid_material_by_body,
            model.named["sequences"],
        )
        mat_id = e_rb_new.attrib["mat"]
        e_rb_existing = e_bc_body_parent.find(f'rigid_body[@mat="{mat_id}"]')
        if e_rb_existing is None:
            e_bc_body_parent.append(e_rb_new)
        else:
            for e in e_rb_new:
                e_rb_existing.append(e)

    # Output section
    plotfile = ET.SubElement(Output, "plotfile", type="febio")
    if not model.output["variables"]:  # empty list
        output_vars = ["displacement", "stress", "relative volume"]
        if module == "biphasic":
            output_vars += ["effective fluid pressure", "fluid pressure", "fluid flux"]
        rigid_bodies_present = any(
            isinstance(m, feb.material.RigidBody) for m in material_registry.objects()
        )
        if rigid_bodies_present:
            output_vars += ["reaction forces"]
    else:
        output_vars = model.output["variables"]
    for var in output_vars:
        ET.SubElement(plotfile, "var", type=var)

    # Write MeshData.  Have to do this before handling boundary
    # conditions because some boundary conditions have part of their
    # values stored in MeshData.
    e_MeshData, e_ElementSet = febioxml.meshdata_section(model)
    root.insert(root.index(Geometry) + 1, e_MeshData)
    if len(e_ElementSet) != 0:
        Geometry.append(e_ElementSet)

    # Step section(s)
    cumulative_time = 0.0
    visited_implicit_bodies = set()
    for istep, step in enumerate(model.steps):
        step_name = (
            step["name"] if step["name"] is not None else "Step{}".format(istep + 1)
        )
        e_step = ET.SubElement(root, "Step", name=step_name)
        if version == "2.0":
            ET.SubElement(e_step, "Module", type=step["module"])
        # Warn if there's an incompatibility between requested materials
        # and modules.
        for mat in material_registry.objects():
            # Extract delegate material object from OrientatedMaterial
            # so that we can check it.  TODO: This is a hack; find a
            # cleaner solution that doesn't require special-casing the
            # module compatibility check.
            if isinstance(mat, matlib.OrientedMaterial):
                checked_mat = mat.material
            else:
                checked_mat = mat
            if ("module" in step) and (
                (not type(checked_mat) in febioxml.module_compat_by_mat)
                or (
                    step["module"]
                    not in febioxml.module_compat_by_mat[type(checked_mat)]
                )
            ):
                raise ValueError(
                    f"Material `{type(mat)}` is not listed as compatible with Module {step['module']}"
                )
        e_Control = ET.Element("Control")
        if version == "2.0":
            ET.SubElement(e_Control, "analysis", type=module)
        # Write all the single-element Control parameters (i.e,
        # everything but <time_stepper>
        for parameter in step["control"]:
            if parameter == "time stepper":
                continue
            e_param = control_parameter_to_feb(parameter, step["control"][parameter])
            e_Control.append(e_param)
        # Write <time_stepper> and all its children
        e_ts = ET.SubElement(e_Control, "time_stepper")
        ET.SubElement(e_ts, "dtmin").text = str(
            step["control"]["time stepper"]["dtmin"]
        )
        ET.SubElement(e_ts, "max_retries").text = str(
            step["control"]["time stepper"]["max retries"]
        )
        ET.SubElement(e_ts, "opt_iter").text = str(
            step["control"]["time stepper"]["opt iter"]
        )
        # dtmax may have an associated sequence
        dtmax = step["control"]["time stepper"]["dtmax"]
        e_dtmax = _property_to_feb(dtmax, "dtmax", model)
        e_ts.append(e_dtmax)

        # Boundary conditions
        #
        # FEBio XML spreads the boundary conditions (constraints) out in
        # amongst many tags, in a rather disorganized fashion.
        #
        # For nodal contraints, there is one parent tag per kind + dof
        # + sequence, and one child tag per node + value.  The parent
        # tag may be named 'prescribe' or 'fix'.
        #
        # For body constraints, there is one parent tag per body, and
        # one child tag per kind + dof + sequence + value.  The parent
        # tag may be named 'prescribed' or 'fixed'.  (Note the
        # inconsistent tense compared to nodal constraints.)
        #
        # FEBio does handle empty tags appropriately, which helps.
        e_Boundary = ET.Element("Boundary")
        if version == "2.0":
            e_bc_body_parent = ET.SubElement(e_step, "Constraints")
        elif version_major == 2 and version_minor >= 5:
            e_bc_body_parent = e_Boundary
        #
        # Collect nodal BCs in a more convenient heirarchy for writing
        # FEBio XML.  FEBio XML only supports nodal boundary conditions
        # if the node list shares the same boundary condition kind
        # ("fixed" or "variable"), dof, and sequence, so we sort the
        # nodal boundary conditions into one collection for each
        # distinct combination of these attributes.  The resulting
        # dictionary looks like:
        # node_memo['fixed'|'variable'][dof][sequence] = (node_ids, scales, relative)
        #
        # TODO: Need to split by `relative`, since each <prescribe>
        # cannot mix relative and non-relative boundary conditions.
        node_memo = defaultdict(dict)
        if ("bc" in step) and ("node" in step["bc"]):
            for node_id in step["bc"]["node"]:
                for dof in step["bc"]["node"][node_id]:
                    bc = step["bc"]["node"][node_id][dof]
                    if bc["sequence"] == "fixed":
                        kind = "fixed"
                    else:  # bc['sequence'] is Sequence
                        kind = "variable"
                    node_memo[kind].setdefault((dof, bc["variable"]), {}).setdefault(
                        bc["sequence"], []
                    ).append((node_id, bc["scale"], bc["relative"]))
        # TODO: support kind == 'fixed'.  (Does that make sense for a step?)
        for kind in node_memo:  # 'variable' or 'fixed'
            for dof_var in node_memo[kind]:  # ("x1", "displacement"), etc.
                for seq in node_memo[kind][dof_var]:
                    # `seq` can be a Sequence or ScaledSequence
                    bc = node_memo[kind][dof_var][seq]
                    dof, var = dof_var
                    if kind == "variable":
                        # Get ID for Sequence (recall that a
                        # ScaledSequence has no ID; only its underlying
                        # Sequence gets an ID).  TODO: Move into
                        # node_var_disp_xml once _get_or_create_seq_id
                        # can be refactored.
                        seq_id = _get_or_create_seq_id(model.named["sequences"], seq)
                        node_ids, scales, rel = zip(*bc)
                        e_bc, e_nodedata = febioxml.node_var_disp_xml(
                            model,
                            root,
                            node_ids,
                            scales,
                            seq_id,
                            dof,
                            var,
                            rel[0],
                            istep,
                        )
                        e_Boundary.append(e_bc)
                        e_MeshData.append(e_nodedata)
                    elif kind == "fixed":
                        raise NotImplementedError

        # Temporal (step-specific) contacts
        contacts = [
            c for c in step["bc"]["contact"] if isinstance(c, ContactConstraint)
        ]
        e_Contact = contact_section(
            contacts, model, named_surface_pairs, named_contacts
        )

        # Add <Boundary>, <Contact>, and <Control> elements to <Step>, in that order
        e_step.append(e_Boundary)
        e_step.append(e_Contact)
        e_step.append(e_Control)

        if ("bc" in step) and ("body" in step["bc"]):
            for body, constraints in step["bc"]["body"].items():
                e_rigid_body = body_constraints_to_feb(
                    body,
                    constraints,
                    material_registry,
                    implicit_rigid_material_by_body,
                    model.named["sequences"],
                )
                e_bc_body_parent.append(e_rigid_body)

    # Write XML elements for sequences (load curves) that are in the
    # model's named entity registry. Sequences can be referenced in a
    # lot of places, including boundary conditions, time stepper curves,
    # and any material parameter.  Therefore, as opposed to trying to
    # find them all here at the time of writing, we require that
    # whenenever an XML element that references a sequence is added to
    # the XML tree elsewhere, said sequence is also added to the model's
    # named entity registry (which has to be done anyway becuase FEBio
    # XML references sequences by ID).  Here, we loop over the sequence
    # collected in the named sequence registry and write only those
    # sequences to the XML tree.
    #
    # FEBio ignores the ID attribute; the real ID of a load curve is its
    # ordinal position in the list of <loadcurve> elements.  So we need
    # to sort them and ensure that the IDs are contiguous.
    seq_ids = sorted(model.named["sequences"].namespace("ordinal_id"))
    #
    # Get local → global time offsets to adjust sequences used as
    # boundary conditions or in time stepper.
    seq_t0 = sequence_time_offsets(model)
    if len(seq_ids) > 0:
        assert min(seq_ids) == 0
        assert max(seq_ids) == len(seq_ids) - 1
    for seq_id in seq_ids:
        seq = model.named["sequences"].obj(seq_id, nametype="ordinal_id")
        add_sequence(root, model, seq, seq_t0[seq])

    # Write named geometric entities & sets.  It is better to delay
    # writing named entities & sets until now so we don't accidentally
    # write the same set twice.
    #
    # Write any named node sets that were not already written.
    for nm, node_set in model.named["node sets"].pairs():
        e_nodeset = root.find(f"Geometry/NodeSet[@name='{nm}']")
        if e_nodeset is None:
            add_nodeset(root, nm, node_set)
    # Write *all* named face sets ("surfaces")
    for nm, face_set in model.named["face sets"].pairs():
        e_surface = ET.SubElement(Geometry, "Surface", name=nm)
        for face in face_set:
            e_surface.append(tag_face(face))
    # Write *all* named surface pairs
    for nm, (primary, secondary) in named_surface_pairs.pairs():
        e_surfpair = febioxml.surface_pair_xml(
            model.named["face sets"], primary, secondary, nm
        )
        Geometry.append(e_surfpair)
    # TODO: Handle element sets too.

    tree = ET.ElementTree(root)
    return tree


def tag_face(face):
    nm = {3: "tri3", 4: "quad4"}
    tag = ET.Element(nm[len(face)])
    tag.text = ", ".join([f"{i+1}" for i in face])
    return tag


def write_xml(tree, f):
    """Write an XML tree to a .feb file"""
    tree.write(f, pretty_print=True, xml_declaration=True, encoding="utf-8")


def write_feb(model, f, version="2.5"):
    """Write model's FEBio XML representation to a file object.

    Inputs
    ------
    fpath : string
        Path for output file.

    materials : list of Material objects

    """
    tree = xml(model, version=version)
    write_xml(tree, f)
