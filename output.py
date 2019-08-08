# Base packages
from collections import defaultdict
from copy import copy, deepcopy
from datetime import datetime
from math import degrees
# System packages
from lxml import etree as ET
# Within-module packages
import febtools as feb
from .core import Body, ImplicitBody, ContactConstraint, NameRegistry, Sequence, ScaledSequence, RigidInterface
from .control import step_duration
from . import material as material_lib
from . import febioxml_2_5 as febioxml
from . import febioxml_2_0
from .febioxml import vec_to_text, bvec_to_text, control_tagnames_to_febio, control_values_to_febio, TAG_FROM_BC
# ^ The intent here is to eventually be able to switch between FEBio XML
# formats by exchanging this import statement for a different version.
# Common functionality can be shared between febioxml_*_*.py files via
# imports.


def default_febio_config():
    """Return default FEBio settings"""
    return {"output variables": ["displacement",
                                 "stress"]}


def _get_or_create_item_id(registry, item):
    """Get or create ID for an item.

    Getting or creating an ID for an item is complicated because item
    IDs must start at 0 and be sequential and contiguous.

    """
    item_ids = registry.names("ordinal_id")
    if len(item_ids) == 0:
        # Handle the trivial case of no pre-existing itemuences
        item_id = 0
    else:
        # At least one itemuence already exists.  Make sure the ID
        # constraints have not been violated
        assert min(item_ids) == 0
        assert max(item_ids) == len(item_ids) - 1
        # Check for an existing ID
        try:
            item_id = registry.name(item, "ordinal_id")
        except KeyError:
            # Create an ID because the itemuence doesn't have one
            item_ids = registry.names("ordinal_id")
            item_id = max(item_ids) + 1
            registry.add(item_id, item, "ordinal_id")
    return item_id


def _get_or_create_seq_id(registry, sequence):
    """Get or create ID for a sequence."""
    if type(sequence) is ScaledSequence:
        sequence = sequence.sequence
    return _get_or_create_item_id(registry, sequence)


def _fixup_ordinal_ids(registry):
    """Regenerate ordinal IDs to satisfy invariants"""
    items = [item for i, item in sorted(registry.pairs("ordinal_id"))]
    for i, item in enumerate(items):
        registry.add(i, item, nametype="ordinal_id")
    assert min(registry.names("ordinal_id")) == 0
    assert max(registry.names("ordinal_id")) ==\
        len(registry.names("ordinal_id")) - 1


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
        e.text = str(p.scale)
    else:
        # Fixed property
        e.text = str(p)
    return e


def exponentialfiber_to_feb(mat, model):
    """Convert ExponentialFiber material instance to FEBio XML.

    """
    e = ET.Element('material', type='fiber-exp-pow')
    e.append(_property_to_feb(mat.alpha, "alpha", model))
    e.append(_property_to_feb(mat.beta, "beta", model))
    e.append(_property_to_feb(mat.xi, "ksi", model))
    e.append(_property_to_feb(degrees(mat.theta), "theta", model))
    e.append(_property_to_feb(degrees(mat.phi), "phi", model))
    return e


def power_linear_fiber_to_feb(mat, model):
    """Convert PowerLinearFiber material instance to FEBio XML."""
    e = ET.Element('material', type='fiber-pow-linear')
    e.append(_property_to_feb(mat.E, "E", model))
    e.append(_property_to_feb(mat.β, "beta", model))
    e.append(_property_to_feb(mat.λ0, "lam0", model))
    e.append(_property_to_feb(degrees(mat.azimuth), "theta", model))
    e.append(_property_to_feb(degrees(mat.zenith), "phi", model))
    return e


def holmesmow_to_feb(mat, model):
    """Convert HolmesMow material instance to FEBio XML.

    """
    e = ET.Element('material', type='Holmes-Mow')
    E, ν = feb.material.fromlame(mat.y, mat.mu)
    e.append(_property_to_feb(E, "E", model))
    e.append(_property_to_feb(ν, "v", model))
    e.append(_property_to_feb(mat.beta, "beta", model))
    return e


def isotropicelastic_to_feb(mat, model):
    """Convert IsotropicElastic material instance to FEBio XML.

    """
    e = ET.Element('material', type='isotropic elastic')
    E, ν = feb.material.fromlame(mat.y, mat.mu)
    e.append(_property_to_feb(E, "E", model))
    e.append(_property_to_feb(ν, "v", model))
    return e


def linear_orthotropic_elastic_to_feb(mat, model):
    """Convert LinearOrthotropicElastic material instance to FEBio XML.

    """
    e = ET.Element('material', type='orthotropic elastic')
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
    # Symmetry axes
    axes = ET.SubElement(e, 'mat_axis', type='vector')
    ET.SubElement(axes, 'a').text = ','.join([str(a) for a in mat.x1])
    ET.SubElement(axes, 'd').text = ','.join([str(a) for a in mat.x2])
    return e


def neo_hookean_to_feb(mat, model):
    """Convert NeoHookean material instance to FEBio XML.

    """
    e = ET.Element('material', type='neo-Hookean')
    E, ν = feb.material.fromlame(mat.y, mat.mu)
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
    e = ET.Element('material', type='biphasic')
    # Add solid material
    e_solid = material_to_feb(mat.solid_material, model)
    e_solid.tag = 'solid'
    e.append(e_solid)
    # Add permeability
    typ = febioxml.perm_name_from_class[type(mat.permeability)]
    f = {feb.material.IsotropicConstantPermeability: iso_const_perm_to_feb,
         feb.material.IsotropicHolmesMowPermeability: iso_holmes_mow_perm_to_feb}
    e_permeability = f[type(mat.permeability)](mat.permeability, model)
    e.append(e_permeability)
    return e


def solidmixture_to_feb(mat, model):
    """Convert SolidMixture material instance to FEBio XML.

    """
    e = ET.Element('material', type='solid mixture')
    for submat in mat.materials:
        m = material_to_feb(submat, model)
        m.tag = 'solid'
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
    """Convert SolidMixture material instance to FEBio XML.

    """
    e = ET.Element('material', type='rigid body')
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
    """Convert a material instance to FEBio XML.

    """
    if mat is None:
        e = ET.Element('material', type='unknown')
    else:
        f = {feb.material.ExponentialFiber: exponentialfiber_to_feb,
             feb.material.PowerLinearFiber: power_linear_fiber_to_feb,
             feb.material.HolmesMow: holmesmow_to_feb,
             feb.material.IsotropicElastic: isotropicelastic_to_feb,
             feb.material.NeoHookean: neo_hookean_to_feb,
             feb.material.LinearOrthotropicElastic: linear_orthotropic_elastic_to_feb,
             feb.material.PoroelasticSolid: poroelastic_to_feb,
             feb.material.SolidMixture: solidmixture_to_feb,
             feb.material.RigidBody: rigid_body_to_feb,
             feb.material.DonnanSwelling: donnan_to_feb,
             feb.material.Multigeneration: multigeneration_to_feb}
        try:
            e = f[type(mat)](mat, model)
        except ValueError:
            msg = "{} not implemented for conversion to FEBio XML."
            print(msg.format(mat.__class__))
            raise
    # Add material coordinate system if it is defined for this material.
    # Any mixture material /should/ call `material_to_feb` (this
    # function) for each sub-material, so we shouldn't need to handle
    # material coordinate systems anywhwere else.
    if (mat in model.mesh.material_basis and
        model.mesh.material_basis[mat] is not None):
        basis = model.mesh.material_basis[mat]
        e_mat_axis = ET.Element("mat_axis", type="vector")
        ET.SubElement(e_mat_axis, "a").text = febioxml.bvec_to_text(basis[0])
        ET.SubElement(e_mat_axis, "d").text = febioxml.bvec_to_text(basis[1])
        e.insert(0, e_mat_axis)
    return e


def add_nodeset(model, xml_root, name, nodes):
    """Add a named node set to FEBio XML."""
    e_geometry = xml_root.find("./Geometry")
    for existing in xml_root.xpath(f"Geometry/NodeSet[@name='{name}']"):
        existing.getparent().remove(existing)
    e_nodeset = ET.SubElement(e_geometry, "NodeSet", name=name)
    # Sort nodes to be user-friendly (humans often read .feb files)
    for node_id in sorted(nodes):
        ET.SubElement(e_nodeset, 'node', id=str(node_id + 1))


def add_sequence(xml_root, model, sequence):
    """Add a sequence (load curve) to a FEBio XML tree.

    So we need to sort them and ensure that the IDs are contiguous.

    xml_root !mutates! := Root object of XML tree.

    sequence := Sequence object.

    sequence_id := Integer ID (0-referenced) to use for the sequence
    element's "id" attribute.  The ID will be incremented by 1 to
    account for FEBio XML's use of 1-referenced IDs.

    """
    seq_id = model.named["sequences"].name(sequence, nametype="ordinal_id")
    e_loaddata = xml_root.find("./LoadData")
    e_loadcurve = ET.SubElement(e_loaddata, "loadcurve",
                                id=str(seq_id + 1),
                                type=sequence.typ,
                                extend=sequence.extend)
    for pt in sequence.points:
        ET.SubElement(e_loadcurve, "point").text = ", ".join(str(x) for x in pt)


def _get_or_create_name(registry: NameRegistry, base_name, item):
    """Get or create a unique name for an item.

    `registry` !mutates! := NameRegistry of existing names.
    `_get_or_create_name` updates the registry with the new
    autogenerated name.

    `base_name` := string used to initialize autogenerated name.

    Returns the new name, a string, consisting of `base_name` followed
    by "_" followed by a integer to make the name differ from any
    existing names in the registry.

    """
    # Check if there's an existing name
    try:
        name = registry.name(item)
    except KeyError:
        # Find the first integer not already used as a suffix for the name
        i = 0
        while base_name + "_" + str(i) in registry.names():
            i += 1
        # Create a name using the unused integer
        name = base_name + "_" + str(i)
        # Update the dictionary so the new name persists
        registry.add(name, item)
    return name


def choose_module(materials):
    """Determine which module should be used to run the model.

    Currently only chooses between solid and biphasic.

    """
    module = "solid"
    for m in materials:
        if isinstance(m, feb.material.PoroelasticSolid):
            module = "biphasic"
    return module


def add_contact_section(model, xml_root, named_surface_pairs, named_contacts):
    tag_geometry = xml_root.find("./Geometry")
    tag_contact_section = ET.SubElement(xml_root, 'Contact')
    tags_surfpair = []
    contact_constraints = [constraint for constraint in model.constraints
                           if type(constraint) is ContactConstraint]
    for contact in contact_constraints:
        tag_contact = ET.SubElement(tag_contact_section, 'contact', type=contact.algorithm)
        # Name the contact to match its surface pair so someone reading
        # the XML can match them more easily
        tag_contact.attrib["name"] =\
            _get_or_create_name(named_contacts,
                          f"contact_-_{contact.algorithm}", contact)
        # Autogenerate names for the face sets in the contact
        surface_name = {"leader": "",
                        "follower": ""}
        for k, face_set in zip(("leader", "follower"), (contact.leader, contact.follower)):
            nm = _get_or_create_name(model.named_sets["faces"],
                                     f"contact_surface_-_{contact.algorithm}",
                                     face_set)
            surface_name[k] = nm
            tag_surface = ET.SubElement(tag_geometry, "Surface", name=nm)
            for face in face_set:
                tag_surface.append(tag_face(face))
        # Autogenerate and add the surface pair
        name_surfpair = _get_or_create_name(named_surface_pairs,
                                            f"contact_surfaces_-_{contact.algorithm}",
                                            (contact.leader, contact.follower))
        tag_surfpair = ET.SubElement(tag_geometry, "SurfacePair", name=name_surfpair)
        ET.SubElement(tag_surfpair, "master", surface=surface_name["leader"])
        ET.SubElement(tag_surfpair, "slave", surface=surface_name["follower"])
        tag_contact.attrib["surface_pair"] = name_surfpair
        # Set compression only or tension–compression
        if contact.algorithm == "sliding-elastic":
            ET.SubElement(tag_contact, 'tension').text = str(int(contact.tension))
        else:
            if contact.tension:
                raise ValueError(f"Only the sliding–elastic contact algorithm is known to support tension–compression contact in FEBio.")
        # Write penalty-related tags
        ET.SubElement(tag_contact, 'auto_penalty') \
          .text = "1" if contact.penalty['type'] == 'auto' else "0"
        ET.SubElement(tag_contact, 'penalty').text = f"{contact.penalty['factor']}"
        # Write algorithm modification tags
        ET.SubElement(tag_contact, 'laugon').text = "1" if contact.augmented_lagrange else "0"
        # (two_pass would go here)
    return tag_contact_section


def control_parameter_to_feb(parameter, value):
    """Return FEBio XML element for a control parameter."""
    nm_feb = control_tagnames_to_febio[parameter]
    if parameter in control_values_to_febio:
        val_feb = control_values_to_febio[parameter][value]
    else:
        val_feb = str(value)
    e = ET.Element(nm_feb)
    e.text = val_feb
    return e


def body_constraints_to_feb(body, constraints, material_registry,
                            implicit_rb_mat, sequence_registry):
    """Return <rigid_body> element for a body's constraints dictionary.

    These <rigid_body> elements are used as children of a <Boundary>
    element to specify (in FEBio terms) rigid body boundary conditions.

    """
    # Create or find the associated materials
    if isinstance(body, Body):
        # If an explicit body, its elements define its
        # materials.  We assume that the body is homogenous.
        mat = body.elements[0].material
        mat_id = material_registry.name(mat, nametype="ordinal_id")
    elif isinstance(body, ImplicitBody):
        mat = implicit_rb_mat[body]
        mat_id = material_registry.name(mat, nametype="ordinal_id")
    else:
        msg = f"body {k} does not have a supported type.  " + \
            "Supported body types are Body and ImplicitBody."
        raise ValueError(msg)
    # Create the XML tags for the rigid body BC
    e_body = ET.Element('rigid_body', mat=str(mat_id + 1))
    for axis, bc in constraints.items():
        if bc['sequence'] == 'fixed':
            kind = 'fixed'
        else:  # bc['sequence'] is Sequence
            kind = 'variable'
            seq = bc['sequence']
            v = bc['scale']
        # Determine which tag name to use for the specified
        # variable: force or displacement
        if bc['variable'] in ["displacement", "rotation"]:
             tagname = TAG_FROM_BC['body'][kind]
        elif bc['variable'] == 'force':
             tagname = 'force'
        else:
             raise ValueError(f"Variable {bc['variable']} not supported for BCs.")
        e_bc = ET.SubElement(e_body, tagname,
                             bc=febioxml.axis_to_febio[axis])
        if kind == 'variable':
            seq_id = _get_or_create_seq_id(sequence_registry, seq)
            e_bc.attrib['lc'] = str(seq_id + 1)
            e_bc.text = str(v)
    return e_body


def xml(model, version='2.5'):
    """Convert a model to an FEBio XML tree.

    Creating an FEBio XML tree from a model is useful because it allows
    XML-editing trickery, if necessary, prior to writing the XML to an
    on-disk .feb file.

    """
    # Create dictionaries to keep track of named items
    named_surface_pairs = {}
    named_contacts = {}
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
        if type(mat) is material_lib.RigidBody:
            material_registry.remove_object(mat)
    _fixup_ordinal_ids(material_registry)
    # Ensure each material has an ID
    for mat in materials_used:
        _get_or_create_item_id(material_registry, mat)
    assert materials_used - set(material_registry.objects()) == set()

    root = ET.Element('febio_spec', version="{}".format(version))
    msg = f"Exported to FEBio XML by febtools prerelease at {datetime.today().strftime('%Y-%m-%dT%H:%M:%S%z')}"
    root.append(ET.Comment(msg))

    version_major, version_minor = [int(a) for a in version.split(".")]

    # Set solver module (analysis type)
    module = choose_module([m for m in model.named["materials"].objects()])
    if version_major == 2 and version_minor >= 5:
        # In FEBio XML ≥ 2.5, <Module> must exist and be first tag
        e_module = ET.SubElement(root, 'Module')
        e_module.attrib["type"] = module
    # FEBio XML 2.0 sets <analysis_type> in <Control>

    Globals = ET.SubElement(root, 'Globals')
    Material = ET.SubElement(root, 'Material')

    parts = febioxml.parts(model)
    Geometry = febioxml.geometry_section(model, parts, material_registry)
    root.append(Geometry)

    e_boundary = ET.SubElement(root, 'Boundary')
    if version_major == 2 and version_minor >= 5:
        tag_contact = add_contact_section(model, root, named_surface_pairs, named_contacts)
    else:
        tag_contact = febioxml_2_0.contact_section(model)
        root.append(tag_contact)
    # The <Contact> tag must come before the first <Step> tag or FEBio
    # will only apply the specified contact constraints to the last step
    # (this is an FEBio bug).
    e_constraints = ET.SubElement(root, 'Constraints')
    e_loaddata = ET.SubElement(root, 'LoadData')
    Output = ET.SubElement(root, 'Output')

    # Typical MKS constants
    Constants = ET.SubElement(Globals, 'Constants')
    ET.SubElement(Constants, 'R').text = '8.314e-6'
    ET.SubElement(Constants, 'T').text = '294'
    ET.SubElement(Constants, 'Fc').text = '96485e-9'


    # Materials section
    #
    # Make material tags for all materials assigned to elements.  Sort
    # by id to get around FEBio bug (FEBio ignores the ID attribute and
    # just uses tag order).
    materials = sorted(material_registry.pairs("ordinal_id"))
    for mat_id, mat in materials:
        tag = material_to_feb(mat, model)
        try:
            name = material_registry.name(mat)
        except KeyError:
            name = None
        if name is not None:
            tag.attrib["name"] = name
        tag.attrib["id"] = str(mat_id + 1)
        Material.append(tag)
    # Assemble a list of all implicit rigid bodies used in the model.
    # There is currently no list of rigid bodies in the model or mesh
    # objects, so we have to search for them.  Rigid bodies may be
    # referenced in fixed constraints (model.fixed['body'][k] where k ∈
    # 'x1', 'x2', 'x3', 'α1', 'α2', 'α3') or in
    # model.steps[i]['bc']['body'] for each step i.
    implicit_bodies_to_process = set()  # memo
    # Search fixed constraints for rigid bodies
    for k in model.fixed['body']:
        for body in model.fixed['body'][k]:
            if isinstance(body, ImplicitBody):
                implicit_bodies_to_process.add(body)
    # Search steps' constraints for rigid bodies
    for step in model.steps:
        if "bc" in step:
            for body in step['bc']['body']:
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
        tag.attrib['id'] = str(mat_id + 1)
        tag.attrib['name'] = mat_name
        Material.append(tag)
        # Update material registries
        material_registry.add(mat_name, mat)
        material_registry.add(mat_id, mat, nametype="ordinal_id")
        implicit_rigid_material_by_body[implicit_body] = mat
        #
        # Add the implicit body's rigid interface with the mesh.
        # Assumes interface is a node set.
        if version == '2.0':
            # FEBio XML 2.0 puts rigid bodies under §Constraints
            e_interface = ET.SubElement(e_contact, 'contact',
                                        type='rigid')
            for i in implicit_body.interface:
                ET.SubElement(e_interface, 'node', id=str(i + 1),
                              rb=str(mat_id + 1))
        elif version_major == 2 and version_minor >= 5:
            # FEBio XML 2.5 puts rigid bodies under §Boundary
            try:
                name = model.named["node sets"].name(implicit_body.interface)
            except KeyError:
                name_base = f"{body_name}_interface"
                nodeset = implicit_body.interface
                name = _get_or_create_name(model.named["node sets"],
                                           name_base, nodeset)
            add_nodeset(model, root, name, implicit_body.interface)
            ET.SubElement(e_boundary, "rigid", rb=str(mat_id + 1),
                          node_set=name)

    # Write global <Boundary> element
    #
    # Write interfaces.  Currently just rigid interfaces.
    for interface in model.constraints:
        if type(interface) is not RigidInterface:
            continue
        rigid_body_id = material_registry.name(interface.rigid_body,
                                               nametype="ordinal_id")
        node_set_name = model.named["node sets"].name(interface.node_set)
        add_nodeset(model, root, node_set_name, interface.node_set)
        ET.SubElement(e_Boundary, "rigid",
                      rb=str(rigid_body_id + 1),
                      node_set=node_set_name)
    #
    # Write fixed nodal constraints to global <Boundary>
    for axis, nodeset in model.fixed['node'].items():
        if nodeset:
            if version == '2.0':
                # Tag heirarchy: <Boundary><fix bc="x"><node id="1"> for each node
                e_fixed_nodeset = ET.SubElement(e_boundary, 'fix',
                                                bc=febioxml.axis_to_febio[axis])
                for i in nodeset:
                    ET.SubElement(e_fixed_nodeset, 'node', id=str(i + 1))
            elif version_major == 2 and version_minor >= 5:
                # Tag heirarchy: <Boundary><fix bc="x" node_set="set_name">
                name_base = f"fixed_{axis}_autogen-nodeset"
                name = _get_or_create_name(model.named["node sets"],
                                           name_base, nodeset)
                add_nodeset(model, root, name, nodeset)
                # Create the tag
                ET.SubElement(e_boundary, 'fix', bc=febioxml.axis_to_febio[axis],
                              node_set=name)
    #
    # Write fixed body constraints to global <Boundary>
    body_bcs = {}
    # Choose where to put rigid body constraints depending on FEBio XML
    # version.
    if version == '2.0':
        e_bc_body_parent = e_constraints
    elif version_major == 2 and version_minor >= 5:
        e_bc_body_parent = e_boundary
    # Collect rigid body boundary conditions in a more convenient
    # hierarchy
    for axis, bodies in model.fixed['body'].items():
        for body in bodies:
            body_bcs.setdefault(body, set()).add(axis)
    # Create the tags specifying fixed constraints for the rigid bodies
    for body, axes in body_bcs.items():
        e_body = ET.SubElement(e_bc_body_parent, 'rigid_body')
        # Assign the body's material
        if isinstance(body, Body):
            body_material = body.elements[0].material
            # TODO: ensure that body is all one material
        elif isinstance(body, ImplicitBody):
            body_material = implicit_rigid_material_by_body[body]
        mat_id = material_registry.name(body_material, nametype="ordinal_id")
        e_body.attrib['mat'] = str(mat_id + 1)
        # Write tags for each of the fixed degrees of freedom.  Use
        # sorted order to be deterministic & human-friendly.
        for ax in sorted(axes):
            ET.SubElement(e_body, 'fixed', bc=febioxml.axis_to_febio[ax])
    #
    # TODO: Write time-varying nodal constraints to global <Boundary>
    #
    # Write time-varying rigid body constraints to global <Boundary>
    if version == '2.0':
        e_bc_body_parent = e_constraints
    elif version_major == 2 and version_minor >= 5:
        e_bc_body_parent = e_boundary
    for body, constraints in model.varying["body"].items():
        e_rb_new = body_constraints_to_feb(body, constraints,
                                               material_registry,
                                               implicit_rigid_material_by_body,
                                               model.named["sequences"])
        mat_id = e_rb_new.attrib["mat"]
        e_rb_existing = e_bc_body_parent.find(f'rigid_body[@mat="{mat_id}"]')
        if e_rb_existing is None:
            e_bc_body_parent.append(e_rb_new)
        else:
            for e in e_rb_new:
                e_rb_existing.append(e)

    # Adjust sequences used as boundary conditions or in time stepper.
    # Apply offset to load curves so they start at the same time the
    # previous step ends (global time), as required by FEBio.  In
    # `febtools`, each step has its own running time (local time).
    cumulative_time = 0.0
    for step in model.steps:
        # Gather must point curves
        dtmax = step['control']['time stepper']['dtmax']
        if isinstance(dtmax, Sequence):
            dtmax.points = [(cumulative_time + t, v) for t, v in dtmax.points]
        # Gather variable boundary condition / constraint curves
        curves_to_adjust = set([])
        if "bc" in step:
            for i, ax_bc in step['bc']['node'].items():
                for ax, d in ax_bc.items():
                    if d == 'variable':  # varying ("prescribed") BC
                        curves_to_adjust.add(d['sequence'])
        # Gather the body constraint curves
        if "bc" in step:
            for body, body_constraints in step['bc']['body'].items():
                for ax, params in body_constraints.items():
                    # params = {'variable': variable <string>,
                    #           'sequence': Sequence object or 'fixed',
                    #           'scale': scale <numeric>
                    if type(params['sequence']) is Sequence:
                        curves_to_adjust.add(params['sequence'])
                        # TODO: Add test to exercise this code
        # Adjust the curves
        for curve in curves_to_adjust:
            curve.points = [(cumulative_time + t, v)
                            for t, v in curve.points]
        # Tally running time
        duration = step_duration(step)
        cumulative_time += duration

    # Output section
    plotfile = ET.SubElement(Output, 'plotfile', type='febio')
    if model.output["variables"] is None:
        output_vars = ["displacement", "stress"]
        if module == "biphasic":
            output_vars += ["effective fluid pressure",
                            "fluid pressure",
                            "fluid flux"]
    else:
        output_vars = model.output["variables"]
    for var in output_vars:
        ET.SubElement(plotfile, 'var', type=var)

    # Step section(s)
    cumulative_time = 0.0
    visited_implicit_bodies = set()
    for i, step in enumerate(model.steps):
        e_step = ET.SubElement(root, 'Step',
                               name='Step{}'.format(i + 1))
        if version == '2.0':
            ET.SubElement(e_step, 'Module', type=step['module'])
        # Warn if there's an incompatibility between requested materials
        # and modules.
        for mat in material_registry.objects():
            if ("module" in step) and\
               (step['module'] not in febioxml.module_compat_by_mat[type(mat)]):
                raise ValueError(f"Material `{type(mat)}` is not compatible with Module {step['module']}")
        e_con = ET.SubElement(e_step, 'Control')
        if version == '2.0':
            ET.SubElement(e_con, 'analysis', type=module)
        # Write all the single-element Control parameters (i.e,
        # everything but <time_stepper>
        for parameter in step['control']:
            if parameter == "time stepper":
                continue
            e_param = control_parameter_to_feb(parameter, step['control'][parameter])
            e_con.append(e_param)
        # Write <time_stepper> and all its children
        e_ts = ET.SubElement(e_con, 'time_stepper')
        ET.SubElement(e_ts, 'dtmin').text = \
            str(step['control']['time stepper']['dtmin'])
        ET.SubElement(e_ts, 'max_retries').text = \
            str(step['control']['time stepper']['max retries'])
        ET.SubElement(e_ts, 'opt_iter').text = \
            str(step['control']['time stepper']['opt iter'])
        # dtmax may have an associated sequence
        dtmax = step['control']['time stepper']['dtmax']
        e_dtmax = _property_to_feb(dtmax, "dtmax", model)
        e_ts.append(e_dtmax)

        # Boundary conditions
        #
        # FEBio XML spreads the boundary conditions (constraints) out in
        # amongst many tags, in a rather disorganized fashion.
        #
        # For nodal contraints, there is one parent tag per kind + axis
        # + sequence, and one child tag per node + value.  The parent
        # tag may be named 'prescribe' or 'fix'.
        #
        # For body constraints, there is one parent tag per body, and
        # one child tag per kind + axis + sequence + value.  The parent
        # tag may be named 'prescribed' or 'fixed'.  (Note the different
        # tense compared to nodal constraints—so much for consistency.)
        #
        # FEBio does seem to handle empty tags appropriately, which
        # helps.
        e_bc_nodal_parent = ET.SubElement(e_step, 'Boundary')
        if version == '2.0':
            e_bc_body_parent = ET.SubElement(e_step, 'Constraints')
        elif version_major == 2 and version_minor >= 5:
            e_bc_body_parent = e_bc_nodal_parent
        # Collect nodal BCs in a more convenient heirarchy for writing FEBio XML
        node_memo = {}  # node_memo['fixed'|'variable'][axis] =
                        # {'nodes': [], 'scales': []}
        if ("bc" in step) and ("node" in step["bc"]):
            for node_id in step['bc']['node']:
                for ax in step['bc']['node'][node_id]:  # axis
                    bc = step['bc']['node'][node_id][ax]
                    if bc['sequence'] == 'fixed':
                        kind = 'fixed'
                    else:  # bc['sequence'] is Sequence
                        kind = 'variable'
                    node_memo.setdefault(kind, {}).setdefault(ax, {})['kind'] = kind
                    node_memo[kind][ax]['sequence'] = bc['sequence']
                    node_memo[kind][ax].setdefault('nodes', []).append(node_id)
                    node_memo[kind][ax].setdefault('scales', []).append(bc['scale'])
        # TODO: support kind == 'fixed'.  (Does that make sense for a step?)
        for kind in node_memo:
            for axis in node_memo[kind]:
                bc = node_memo[kind][axis]
                e_bc = ET.SubElement(e_bc_nodal_parent,
                                     TAG_FROM_BC['node'][kind],  # 'fix' | 'prescribe'
                                     bc=febioxml.axis_to_febio[axis])
                if version == '2.0':
                    if kind == 'variable':
                        seq_id = _get_or_create_seq_id(model.named["sequences"],
                                                       bc['sequence'])
                        e_bc.attrib['lc'] =  str(seq_id + 1)
                    # Write nodes as children of <Step><Boundary><prescribe>
                    for i, sc in zip(bc['nodes'], bc['scales']):
                        ET.SubElement(e_bc, 'node', id=str(i+1)).text = f"{sc:.7e}"
                elif version_major == 2 and version_minor >= 5:
                    # Use <Step><Boundary><prescribe node_set="set_name">
                    #
                    # Test if the node-specific scaling factors are all
                    # equal; if they are not, the BC cannot be
                    # represented as a single node set in FEBio XML 2.5.
                    sc0 = bc['scales'][0]
                    if all([sc == sc0 for sc in bc['scales']]):
                        # All scaling factors are equal
                        nm_base = f"step{i+1}_{kind}_{axis}_autogen-nodeset"
                        name = _get_or_create_name(model.named["node sets"],
                                                   nm_base, bc['nodes'])
                    else:
                        msg = (f"A nodal boundary condition was defined with "
                               "non-equal node-specific scaling factors and FEBio XML "
                               "{version} was requested, but FEBio XML {version} can "
                               "only support a single scaling factor for the entire "
                               "node set.")
                        # TODO: Add support for node-specific BCs using
                        # MeshData/NodeData.
                        raise ValueError(msg)
                    if kind == 'variable':
                        seq_id = _get_or_create_seq_id(model.named["sequences"],
                                                       bc['sequence'])
                        e_sc = ET.SubElement(e_bc, 'scale',
                                             lc=str(seq_id + 1))
                        e_sc.text = f"{sc0:.7e}"
                        ET.SubElement(e_bc, 'relative').text = "0"
                    e_bc.attrib['node_set'] = name
        if ("bc" in step) and ("body" in step["bc"]):
            for body, constraints in step["bc"]["body"].items():
                e_rigid_body = body_constraints_to_feb(body,
                                                       constraints,
                                                       material_registry,
                                                       implicit_rigid_material_by_body,
                                                       model.named["sequences"])
                e_bc_body_parent.append(e_rigid_body)

    # Write XML elements for sequences (load curves) that are in the
    # model's named entity registry.  Re-use any ordinal ID found in the
    # model's named entity registry.
    #
    # Sequences can be referenced in a lot of places, including boundary
    # conditions, time stepper curves, and any material parameter.
    # Therefore, instead of try to find them all here, we require that
    # any sequence be added to the model's named entity registry when an
    # XML element that references the sequence is added to the XML tree.
    # Consequently, any sequence not in the named entity registry will
    # not be in the XML output.
    #
    # FEBio ignores the ID attribute; the real ID of a load curve is its
    # ordinal position in the list of <loadcurve> elements.  So we need
    # to sort them and ensure that the IDs are contiguous.
    seq_ids = sorted(model.named["sequences"].names("ordinal_id"))
    assert min(seq_ids) == 0
    assert max(seq_ids) == len(seq_ids) - 1
    for seq_id in seq_ids:
        seq = model.named["sequences"].obj(seq_id, nametype="ordinal_id")
        add_sequence(root, model, seq)

    # Write MeshData
    e_MeshData, e_ElementSet = febioxml.meshdata_section(model)
    root.insert(root.index(Geometry) + 1, e_MeshData)
    Geometry.append(e_ElementSet)

    tree = ET.ElementTree(root)
    return tree


def tag_face(face):
    nm = {3: "tri3",
          4: "quad4"}
    tag = ET.Element(nm[len(face)])
    tag.text = ", ".join([f"{i+1}" for i in face])
    return tag

def write_xml(tree, f):
    """Write an XML tree to a .feb file"""
    tree.write(f, pretty_print=True, xml_declaration=True,
               encoding='utf-8')


def write_feb(model, f, version='2.5'):
    """Write model's FEBio XML representation to a file object.

    Inputs
    ------
    fpath : string
        Path for output file.

    materials : list of Material objects

    """
    tree = xml(model, version=version)
    write_xml(tree, f)
