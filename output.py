# Base packages
from collections import defaultdict
from copy import copy
from functools import singledispatch
from pathlib import PurePosixPath

# Public packages
from typing import BinaryIO, Union

from lxml import etree
from lxml.etree import ElementTree
import numpy as np

# Within-module packages
from .core import (
    ImplicitBody,
    ContactConstraint,
    NameRegistry,
    Sequence,
    ScaledSequence,
    RigidInterface,
    Body,
)
from .control import auto_physics, Physics

from . import Model, material as matlib
from . import febioxml
from . import febioxml_2_0
from . import febioxml_2_5
from . import febioxml_3_0
from . import febioxml_4_0
from .febioxml import (
    CONTACT_NAME_FROM_CLASS,
    CONTACT_PARAMS,
    VerbatimXMLMaterial,
    find_unique_tag,
    get_or_create_item_id,
    get_or_create_xml,
    get_or_create_parent,
    to_text,
    num_to_text,
    vec_to_text,
    material_name_from_class,
    property_to_xml,
    list_domains,
    physics_compat_by_mat,
)

# ^ The intent here is to eventually be able to switch between FEBio XML
# formats by exchanging this import statement for a different version.
# Common functionality can be shared between febioxml_*_*.py files via
# imports.


def default_febio_config():
    """Return default FEBio settings"""
    return {"output variables": ["displacement", "stress"]}


def _fixup_ordinal_ids(registry):
    """Regenerate ordinal IDs to satisfy invariants"""
    # If no ordinal IDs, nothing to fix
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


#########################################
# Material → FEBio XML export functions #
#########################################


@singledispatch
def material_to_feb(mat, model) -> ElementTree:
    """Return material instance as FEBio XML

    The model argument is needed so that time-dependent material parameters, if any,
    can be given the right FEBio XML load curve ID.

    TODO: Try to figure out how to make this conversion work without a model object.

    """
    raise NotImplementedError(f"Conversion of {mat} to FEBio XML is not yet supported.")


@material_to_feb.register
def _(mat: VerbatimXMLMaterial, model) -> ElementTree:
    """Convert VerbatimXMLMaterial instance to FEBio XML"""
    return mat.xml


@material_to_feb.register
def _(
    mat: Union[matlib.OrientedMaterial, matlib.DeviatoricFiber], model
) -> ElementTree:
    """Convert an OrientedMaterial material instance to FEBio XML"""
    orientation = mat.orientation
    e = material_to_feb(mat.material, model)
    # Add material coordinate system if it is defined for this material.  Any mixture
    # material /should/ call `material_to_feb` for each sub-material, so we shouldn't
    # need to handle material coordinate systems anywhere else.
    if orientation is None:
        return e
    if np.array(orientation).ndim == 2:
        # material axes orientation
        e_mat_axis = etree.Element("mat_axis", type="vector")
        etree.SubElement(e_mat_axis, "a").text = febioxml.vec_to_text(orientation[:, 0])
        etree.SubElement(e_mat_axis, "d").text = febioxml.vec_to_text(orientation[:, 1])
        e.insert(0, e_mat_axis)
        e.append(e_mat_axis)
    elif np.array(orientation).ndim == 1:
        # vector orientation
        e_vector = etree.Element("fiber", type="vector")
        e_vector.text = vec_to_text(orientation)
        e.append(e_vector)
    else:
        raise ValueError(
            f"Rank {orientation.ndim} material orientation not supported.  Provided orientation was {orientation}."
        )
    return e


@material_to_feb.register
def _(mat: matlib.EllipsoidalDistribution, model) -> ElementTree:
    """Convert an OrientedMaterial material instance to FEBio XML"""
    e = etree.Element("solid", type="continuous fiber distribution")
    e_fibers = material_to_feb(mat.fiber, model)  # <fibers>
    e_fibers.tag = "fibers"  # was "solid"
    e.append(e_fibers)
    e_dist = etree.SubElement(e, "distribution", type="ellipsoidal")
    etree.SubElement(e_dist, "spa").text = vec_to_text(mat.d)
    e_scheme = etree.SubElement(e, "scheme", type=mat.integration[0])
    etree.SubElement(e_scheme, "nph").text = num_to_text(mat.integration[1])
    etree.SubElement(e_scheme, "nth").text = num_to_text(mat.integration[2])
    return e


@material_to_feb.register
def _(mat: matlib.NeoHookeanFiber, model) -> ElementTree:
    """Convert NeoHookeanFiber material instance to FEBio XML"""
    e = etree.Element("material", type="fiber-NH")
    e.append(property_to_xml(mat.E, "mu", model.named["sequences"]))
    return e


@material_to_feb.register
def _(mat: matlib.NaturalNeoHookeanFiber, model) -> ElementTree:
    """Convert NaturalNeoHookeanFiber material instance to FEBio XML"""
    e = etree.Element("material", type="fiber-natural-NH")
    e.append(property_to_xml(mat.E, "ksi", model.named["sequences"]))
    e.append(property_to_xml(mat.λ0, "lam0", model.named["sequences"]))
    return e


@material_to_feb.register
def _(mat: matlib.ExponentialFiber, model) -> ElementTree:
    """Convert ExponentialFiber material instance to FEBio XML"""
    e = etree.Element("material", type="fiber-exp-pow")
    e.append(property_to_xml(mat.α, "alpha", model.named["sequences"]))
    e.append(property_to_xml(mat.β, "beta", model.named["sequences"]))
    e.append(property_to_xml(mat.ξ, "ksi", model.named["sequences"]))
    return e


@material_to_feb.register
def _(mat: matlib.PowerLinearFiber, model) -> ElementTree:
    """Convert PowerLinearFiber material instance to FEBio XML"""
    e = etree.Element("material", type="fiber-pow-linear")
    e.append(property_to_xml(mat.E, "E", model.named["sequences"]))
    e.append(property_to_xml(mat.β, "beta", model.named["sequences"]))
    e.append(property_to_xml(mat.λ0, "lam0", model.named["sequences"]))
    return e


@material_to_feb.register
def _(mat: matlib.ExpAndLinearDCFiber, model) -> ElementTree:
    """Convert ExpεAndLinεDEFiber material instance to FEBio XML"""
    e = etree.Element("material", type="fiber-exp-linear")
    e.append(property_to_xml(mat.ξ, "c3", model.named["sequences"]))
    e.append(property_to_xml(mat.α, "c4", model.named["sequences"]))
    e.append(property_to_xml(mat.λ1, "lambda", model.named["sequences"]))
    e.append(property_to_xml(mat.E, "c5", model.named["sequences"]))
    return e


@material_to_feb.register
def _(mat: matlib.EllipsoidalPowerFiber, model) -> ElementTree:
    """Covert EllipsoidalPowerFiber material instance to FEBio XML"""
    type_ = material_name_from_class[mat.__class__]
    e = etree.Element("material", type=type_)
    e_ksi = etree.SubElement(e, "ksi")
    e_ksi.text = vec_to_text(mat.ξ)
    e_beta = etree.SubElement(e, "beta")
    e_beta.text = vec_to_text(mat.β)
    return e


@material_to_feb.register
def _(mat: matlib.HolmesMow, model) -> ElementTree:
    """Convert HolmesMow material instance to FEBio XML"""
    e = etree.Element("material", type="Holmes-Mow")
    e.append(property_to_xml(mat.E, "E", model.named["sequences"]))
    e.append(property_to_xml(mat.ν, "v", model.named["sequences"]))
    e.append(property_to_xml(mat.β, "beta", model.named["sequences"]))
    return e


@material_to_feb.register
def _(mat: matlib.IsotropicElastic, model) -> ElementTree:
    """Convert IsotropicElastic material instance to FEBio XML"""
    e = etree.Element("material", type="isotropic elastic")
    E, ν = matlib.from_Lamé(mat.y, mat.mu)
    e.append(property_to_xml(E, "E", model.named["sequences"]))
    e.append(property_to_xml(ν, "v", model.named["sequences"]))
    return e


@material_to_feb.register
def _(mat: matlib.OrthotropicLinearElastic, model) -> ElementTree:
    """Convert OrthotropicElastic material instance to FEBio XML"""
    e = etree.Element("material", type="orthotropic elastic")
    # Material properties
    e.append(property_to_xml(mat.E1, "E1", model.named["sequences"]))
    e.append(property_to_xml(mat.E2, "E2", model.named["sequences"]))
    e.append(property_to_xml(mat.E3, "E3", model.named["sequences"]))
    e.append(property_to_xml(mat.G12, "G12", model.named["sequences"]))
    e.append(property_to_xml(mat.G23, "G23", model.named["sequences"]))
    e.append(property_to_xml(mat.G31, "G31", model.named["sequences"]))
    e.append(property_to_xml(mat.ν12, "v12", model.named["sequences"]))
    e.append(property_to_xml(mat.ν23, "v23", model.named["sequences"]))
    e.append(property_to_xml(mat.ν31, "v31", model.named["sequences"]))
    return e


@material_to_feb.register
def _(mat: matlib.FungOrthotropicElastic, model) -> ElementTree:
    """Convert FungOrthotropic material instance to FEBio XML"""
    e = etree.Element("material", type="Fung-ortho-compressible")
    # Material properties
    e.append(property_to_xml(mat.E1, "E1", model.named["sequences"]))
    e.append(property_to_xml(mat.E2, "E2", model.named["sequences"]))
    e.append(property_to_xml(mat.E3, "E3", model.named["sequences"]))
    e.append(property_to_xml(mat.G12, "G12", model.named["sequences"]))
    e.append(property_to_xml(mat.G23, "G23", model.named["sequences"]))
    e.append(property_to_xml(mat.G31, "G31", model.named["sequences"]))
    e.append(property_to_xml(mat.ν12, "v12", model.named["sequences"]))
    e.append(property_to_xml(mat.ν23, "v23", model.named["sequences"]))
    e.append(property_to_xml(mat.ν31, "v31", model.named["sequences"]))
    e.append(property_to_xml(mat.c, "c", model.named["sequences"]))
    e.append(property_to_xml(mat.K, "k", model.named["sequences"]))
    return e


@material_to_feb.register
def _(mat: matlib.NeoHookean, model) -> ElementTree:
    """Convert NeoHookean material instance to FEBio XML"""
    e = etree.Element("material", type="neo-Hookean")
    E, ν = matlib.from_Lamé(mat.λ, mat.μ)
    e.append(property_to_xml(E, "E", model.named["sequences"]))
    e.append(property_to_xml(ν, "v", model.named["sequences"]))
    return e


@material_to_feb.register
def _(mat: matlib.IsotropicConstantPermeability, model):
    """Convert IsotropicConstantPermeability instance to FEBio XML"""
    e = etree.Element("permeability", type="perm-const-iso")
    e.append(property_to_xml(mat.k, "perm", model.named["sequences"]))
    return e


@material_to_feb.register
def _(mat: matlib.IsotropicExponentialPermeability, model):
    """Convert IsotropicExponentialPermeability instance to FEBio XML"""
    e = etree.Element("permeability", type="perm-exp-iso")
    e.append(property_to_xml(mat.k0, "perm", model.named["sequences"]))
    e.append(property_to_xml(mat.M, "M", model.named["sequences"]))
    return e


@material_to_feb.register
def _(mat: matlib.IsotropicHolmesMowPermeability, model) -> ElementTree:
    """Convert IsotropicHolmesMowPermeability instance to FEBio XML"""
    e = etree.Element("permeability", type="perm-Holmes-Mow")
    e.append(property_to_xml(mat.k0, "perm", model.named["sequences"]))
    e.append(property_to_xml(mat.M, "M", model.named["sequences"]))
    e.append(property_to_xml(mat.α, "alpha", model.named["sequences"]))
    return e


@material_to_feb.register
def _(mat: matlib.TransIsoHolmesMowPermeability, model) -> ElementTree:
    """Convert IsotropicHolmesMowPermeability instance to FEBio XML"""
    e = etree.Element("permeability", type="perm-ref-trans-iso")
    e.append(property_to_xml(mat.k0, "perm0", model.named["sequences"]))
    e.append(property_to_xml(mat.M0, "M0", model.named["sequences"]))
    e.append(property_to_xml(mat.α0, "alpha0", model.named["sequences"]))
    e.append(property_to_xml(mat.k1a, "perm1A", model.named["sequences"]))
    e.append(property_to_xml(mat.k2a, "perm2A", model.named["sequences"]))
    e.append(property_to_xml(mat.Ma, "MA", model.named["sequences"]))
    e.append(property_to_xml(mat.αa, "alphaA", model.named["sequences"]))
    e.append(property_to_xml(mat.k1t, "perm1T", model.named["sequences"]))
    e.append(property_to_xml(mat.k2t, "perm2T", model.named["sequences"]))
    e.append(property_to_xml(mat.Mt, "MT", model.named["sequences"]))
    e.append(property_to_xml(mat.αt, "alphaT", model.named["sequences"]))
    return e


@material_to_feb.register
def _(mat: matlib.PoroelasticSolid, model) -> ElementTree:
    """Convert Poroelastic material instance to FEBio XML"""
    e = etree.Element("material", type="biphasic")
    e.append(property_to_xml(mat.solid_fraction, "phi0", model.named["sequences"]))
    e.append(
        property_to_xml(mat.fluid_density, "fluid_density", model.named["sequences"])
    )
    # Add solid material
    e_solid = material_to_feb(mat.solid_material, model)
    e_solid.tag = "solid"
    e.append(e_solid)
    # Add permeability
    e_permeability = material_to_feb(mat.permeability, model)
    e.append(e_permeability)
    return e


@material_to_feb.register
def _(mat: matlib.SolidMixture, model) -> ElementTree:
    """Convert SolidMixture material instance to FEBio XML"""
    e = etree.Element("material", type="solid mixture")
    for submat in mat.materials:
        m = material_to_feb(submat, model)
        m.tag = "solid"
        e.append(m)
    return e


@material_to_feb.register
def _(mat: matlib.Multigeneration, model) -> ElementTree:
    """Convert Multigeneration material instance to FEBio XML"""
    e = etree.Element("material", type="multigeneration")
    i = 1
    for t, submat in zip(mat.generation_times, mat.materials):
        e_generation = etree.SubElement(e, "generation")
        e_generation.attrib["id"] = str(i)
        i += 1
        etree.SubElement(e_generation, "start_time").text = str(t)
        e_submat = material_to_feb(submat, model)
        e_submat.tag = "solid"
        e_generation.append(e_submat)
    return e


@material_to_feb.register
def _(mat: matlib.Rigid, model) -> ElementTree:
    """Convert Rigid material instance to FEBio XML"""
    e = etree.Element("material", type="rigid body")
    if mat.density is None:
        density = 1
    else:
        density = mat.density
    e.append(property_to_xml(density, "density", model.named["sequences"]))
    return e


@material_to_feb.register
def donnan_to_feb(mat: matlib.DonnanSwelling, model) -> ElementTree:
    """Convert DonnanSwelling material instance to FEBio XML"""
    e = etree.Element("material", type="Donnan equilibrium")
    e.append(property_to_xml(mat.phi0_w, "phiw0", model.named["sequences"]))
    e.append(property_to_xml(mat.fcd0, "cF0", model.named["sequences"]))
    e.append(property_to_xml(mat.ext_osm, "bosm", model.named["sequences"]))
    e.append(property_to_xml(mat.osm_coef, "Phi", model.named["sequences"]))
    return e


def add_uncoupled_bulk_to_feb(
    e,
    bulk: Union[
        matlib.VolumetricLinear, matlib.VolumetricLogInverse, matlib.VolumetricHGO
    ],
    model,
):
    """Add bulk compressibility model to FEBio XML element for an uncoupled material

    All FEBio versions support integer values; only FEBio supports text values.  So
    FEBio ≤ 3 will read any text value as its default ln(J)/J law.  Therefore,
    to be safe, we write integer codes.

    """
    # Can't use @material_to_feb.register because we want to add two elements: a
    # comment and the actual bulk compression law
    e.append(property_to_xml(bulk.K, "k", model.named["sequences"]))
    law = {
        matlib.VolumetricLogInverse: "0",
        matlib.VolumetricHGO: "1",
        matlib.VolumetricLinear: "2",
    }[type(bulk)]
    e.append(etree.Comment(f"Bulk: {type(bulk).__name__}"))
    etree.SubElement(e, "pressure_model").text = law


@material_to_feb.register
def uncoupled_mooney_rivlin_to_feb(
    mat: febioxml.UncoupledMooneyRivlin, model
) -> ElementTree:
    """Convert UncoupledMooneyRivling material instance to FEBio XML"""
    e = etree.Element("material", type="Mooney-Rivlin")
    e.append(property_to_xml(mat.solid.c1, "c1", model.named["sequences"]))
    e.append(property_to_xml(mat.solid.c2, "c2", model.named["sequences"]))
    add_uncoupled_bulk_to_feb(e, mat.bulk, model)
    return e


@material_to_feb.register
def trans_iso_mooney_rivlin_febio_to_feb(
    mat: febioxml.TransIsoMooneyRivlinFEBio, model
) -> ElementTree:
    """Convert UncoupledMooneyRivling material instance to FEBio XML"""
    e = etree.Element("material", type="trans iso Mooney-Rivlin")
    e.append(property_to_xml(mat.solid.c1, "c1", model.named["sequences"]))
    e.append(property_to_xml(mat.solid.c2, "c2", model.named["sequences"]))
    for ec in material_to_feb(mat.fibers, model):
        if ec.tag == "lambda":
            ec.tag = "lam_max"  # FEBio XML inconsistency between materials
        e.append(ec)
    # e.append(property_to_xml(mat.fibers.ξ, "c3", model.named["sequences"]))
    # e.append(property_to_xml(mat.fibers.α, "c4", model.named["sequences"]))
    # e.append(property_to_xml(mat.fibers.E, "c5", model.named["sequences"]))
    # e.append(property_to_xml(mat.fibers.λ1, "lam_max", model.named["sequences"]))
    add_uncoupled_bulk_to_feb(e, mat.bulk, model)
    return e


@material_to_feb.register
def uncoupled_HGO_to_feb(mat: febioxml.UncoupledHGOFEBio, model) -> ElementTree:
    """Convert UncoupledHGOFEBio material instance to FEBio XML"""
    # Holzapfel-Gasser-Ogden was introduced in FEBio 3.2.
    e = etree.Element("material", type="Holzapfel-Gasser-Ogden")
    e.append(property_to_xml(mat.matrix.μ, "c", model.named["sequences"]))
    e.append(property_to_xml(mat.fiber_pos.material.ξ, "k1", model.named["sequences"]))
    e.append(property_to_xml(mat.fiber_pos.material.α, "k2", model.named["sequences"]))
    e.append(
        property_to_xml(mat.fiber_pos.material.κ, "kappa", model.named["sequences"])
    )
    e.append(property_to_xml(mat.azimuth, "gamma", model.named["sequences"]))
    e.append(property_to_xml(mat.bulk.K, "k", model.named["sequences"]))
    etree.SubElement(e, "pressure_model").text = "Abaqus (GOH)"
    return e


@material_to_feb.register
def prony_viscoelasticity_to_feb(
    mat: matlib.PronyViscoelasticity, model
) -> ElementTree:
    """Convert PronyViscoelasticity material instance to FEBio XML"""
    e = etree.Element("material", type="viscoelastic")
    etree.SubElement(e, "g0").text = "1"
    for i, (γ, τ) in enumerate(zip(mat.γ, mat.τ)):
        e.append(property_to_xml(γ, f"g{i + 1}", model.named["sequences"]))
        e.append(property_to_xml(γ, f"t{i + 1}", model.named["sequences"]))
    e_solid = material_to_feb(mat.material, model)
    e_solid.tag = "elastic"  # unfortunate that FEBio tag names are context-dependent
    e.append(e_solid)
    return e


########################################
# End of material conversion functions #
########################################


def add_nodeset(xml_root, name, nodes, febioxml_module):
    """Add a named node set to FEBio XML"""
    if len(nodes) == 0:
        # Tested in FEBio 3.2
        raise ValueError(
            f"Node set '{name}' is empty.  FEBio XML unfortunately does not support empty node sets."
        )
    fx = febioxml_module
    e_Mesh = xml_root.find(fx.MESH_TAG)
    for existing in xml_root.xpath(f"{fx.MESH_TAG}/NodeSet[@name='{name}']"):
        existing.getparent().remove(existing)
    e_nodeset = fx.xml_nodeset(nodes, name)
    e_Mesh.append(e_nodeset)


def sequence_time_offsets(model):
    """Return map: sequence → global start time.

    In `waffleiron`, each step has its own running time (step-local time),
    and step-related time sequences are in step-local time.  But in
    FEBio XML, all time sequences are written in global time.  This
    function calculates and returns the time offsets that must be added
    to each sequence to convert said sequence from local time to global
    time.

    This function assumes that curves are not re-used across steps.
    TODO: Figure out some way of enforcing this.

    """
    cumulative_time = 0.0
    seq_t0 = defaultdict(lambda: 0)  # dict: sequence → time offset
    for step, name in model.steps:
        curves_to_adjust = set([])
        # Gather must point curves
        dtmax = step.ticker.dtmax
        if isinstance(dtmax, Sequence) or isinstance(dtmax, ScaledSequence):
            curves_to_adjust.add(dtmax)
        # Gather variable boundary condition / constraint curves
        for i, ax_bc in step.bc["node"].items():
            for ax, d in ax_bc.items():
                if isinstance(d["sequence"], Sequence):
                    curves_to_adjust.add(d["sequence"])
                elif isinstance(d["sequence"], ScaledSequence):
                    curves_to_adjust.add(d["sequence"].sequence)
        # Gather the body constraint curves
        for body, body_constraints in step.bc["body"].items():
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
        cumulative_time += step.duration
    return seq_t0


def contact_section(
    contacts, model, named_surface_pairs, named_contacts, febioxml_module
):
    fx = febioxml_module
    e_contact_section = etree.Element("Contact")
    for contact in contacts:
        algo = CONTACT_NAME_FROM_CLASS[contact.__class__]
        contact_name = named_contacts.get_or_create_name(f"contact_-_{algo}", contact)
        # Create the bar <contact> XML element in a version-specific manner
        e_contact = fx.contact_bare_xml(
            contact, model, named_surface_pairs, contact_name=contact_name
        )
        e_contact_section.append(e_contact)
        # Fill in the contact parameters (not currently known to be specific to any
        # particular FEBio version)
        for nm, v in contact.values.items():
            # Handle None values.  In waffleiron, that means the function that parameter
            # governs is turned off.  FEBio XML doesn't have a systematic convention to
            # indicate a function is disabled, so we set the parameter to the FEBio
            # default instead.  Usually the FEBio default is magic value indicating
            # "off".  So far this hasn't been a problem.
            if v is None:
                v = CONTACT_PARAMS[nm].default
            etree.SubElement(e_contact, CONTACT_PARAMS[nm].path).text = to_text(v)
    return e_contact_section


def face_xml(face, face_id):
    """Return XML element for a face.

    face := tuple of ints; canonical face tuple.  Zero-origin.

    face_id := integer ID to use for the face.  Face IDs should be
    assigned in the order the faces' XML elements will be inserted into
    the XML tree.

    """
    nm = {3: "tri3", 4: "quad4"}
    e = etree.Element(nm[len(face)], id=str(face_id + 1))
    e.text = " " + ", ".join(f"{i+1}" for i in face) + " "
    return e


def step_xml(step, name, seq_registry, physics, febioxml_module):
    """Return <Step> XML element"""
    # We need to know what physics are being used because FEBio accepts
    # some parameters only for some physics.
    fx = febioxml_module
    e_step = etree.Element(fx.STEP_NAME, name=name)
    e_control = etree.SubElement(e_step, "Control")
    # Dynamics
    # TODO: In FEBio XML 4.0, "static" is elastic-only, and "steady-state"
    #  is biphasic-only
    e_control.append(fx.xml_dynamics(step.dynamics, physics))
    # Ticker
    for nm, p in fx.TICKER_PARAMS.items():
        parent = get_or_create_parent(e_step, p.path)
        tag = p.path.split("/")[-1]
        e = property_to_xml(getattr(step.ticker, nm), tag, seq_registry)
        # Special-case dtnom (<step_size>), since it is multiplied with the number of
        # steps to get the step duration, which is kind of important.
        if nm == "dtnom":
            e.text = f"{getattr(step.ticker, nm):.17f}"
        parent.append(e)
    for nm, p in fx.CONTROLLER_PARAMS.items():
        parent = get_or_create_parent(e_step, p.path)
        tag = p.path.split("/")[-1]
        v = getattr(step.controller, nm)
        if nm == "save_iters":
            e = etree.SubElement(parent, tag)
            e.text = v.value
        else:
            e = property_to_xml(v, tag, seq_registry)
        parent.append(e)
    for nm, p in fx.SOLVER_PARAMS.items():
        parent = get_or_create_parent(e_step, p.path)
        tag = p.path.split("/")[-1]
        if nm == "ptol" and not physics == Physics.BIPHASIC:
            continue
        else:
            e = property_to_xml(getattr(step.solver, nm), tag, seq_registry)
        parent.append(e)
    # Handle update method separately, because it has special formatting in some XML
    # versions
    e_parent = get_or_create_xml(e_step, PurePosixPath(fx.QNMETHOD_PATH_IN_STEP).parent)
    e_parent.append(fx.xml_qnmethod(step.solver))
    return e_step


def xml(model: Model, version="3.0"):
    """Convert a model to an FEBio XML tree.

    Creating an FEBio XML tree from a model is useful because it allows
    XML-editing trickery, if necessary, prior to writing the XML to an
    on-disk .feb file.

    """
    version_major, version_minor = [int(a) for a in version.split(".")]
    if version_major == 2 and version_minor == 0:
        fx = febioxml_2_0
    elif version_major == 2 and version_minor == 5:
        fx = febioxml_2_5
    elif version_major == 3 and version_minor == 0:
        fx = febioxml_3_0
    elif version_major == 4 and version_minor == 0:
        fx = febioxml_4_0
    else:
        raise NotImplementedError(
            f"Writing FEBio XML {version_major}.{version_minor} is not supported."
        )
    root = etree.Element("febio_spec", version="{}".format(version))

    # Create dictionaries to keep track of named items
    named_surface_pairs = NameRegistry()
    named_contacts = NameRegistry()
    # Gather all materials that are assigned to elements.  We do this early because in
    # FEBio XML the material ids are needed to define the geometry and meshdata
    # sections.  FEBio rigid materials will be created or duplicated later to satisfy
    # FEBio's design constraints.
    materials_used = set(
        e.material for e in model.mesh.elements if e.material is not None
    )
    # Create a new dictionary of materials → material IDs.  This dictionary will be
    # updated as new materials are autogenerated during model conversion to FEBio XML.
    # This dictionary is the canonical source of material IDs for the export process; do
    # not use model.named["materials"] again.
    material_registry = copy(model.named["materials"])

    # Assemble a list of all rigid bodies used in the model.  There is no central list
    # of rigid bodies, so we have to search for them everywhere they may be used.
    bodies = set()
    # Search fixed constraints for rigid bodies
    for k in model.fixed["body"]:
        for body in model.fixed["body"][k]:
            bodies.add(body)
    # Search variable constraints for rigid bodies
    # TODO: It's a pain to have both global and step-specific variable constraints
    for body in model.varying["body"]:
        bodies.add(body)
    # Search steps' constraints for rigid bodies
    for step, name in model.steps:
        for body in step.bc["body"]:
            bodies.add(body)
    for interface in model.constraints:
        if isinstance(interface, RigidInterface):
            bodies.add(interface.rigid_body)

    # Associate each rigid body with a rigid material.  FEBio references rigid bodies
    # by material name instead of by the geometry, so we need a 1:1 correspondence.
    rigid_material_for_body = {}
    body_for_rigid_material = {}
    for body in bodies:
        if isinstance(body, ImplicitBody):
            if body.material is not None:
                mat = body.material
            else:
                mat = matlib.Rigid()
                # TODO: both these functions add the material to the registry; they
                #  should probably be one function call.
                get_or_create_item_id(material_registry, mat)
                material_registry.get_or_create_name(
                    "implicit_rigid_body_pseudo-material", mat
                )
        else:
            mat = body.elements[0].material
        materials_used.add(mat)
        if body in rigid_material_for_body:
            if mat is not rigid_material_for_body[body]:
                raise ValueError(f"{body} has more than one rigid material.")
        rigid_material_for_body[body] = mat
        if mat in body_for_rigid_material:
            if body is not body_for_rigid_material[mat]:
                raise ValueError(f"{mat} has more than one rigid body.")
        body_for_rigid_material[mat] = body

    # Ensure each material has an ID and name
    for mat in materials_used:
        get_or_create_item_id(material_registry, mat)
        # FEBio XML ≥ 3.0 requires each material to have a name; in prior FEBio XML
        # versions this is optional.
        material_registry.get_or_create_name("Material", mat)
    assert materials_used - set(material_registry.objects()) == set()

    # Get rid of unused rigid body materials.  FEBio 2.9.0, and possibly other
    # versions, gives a zero diagonal error termination if a rigid material is
    # present but has no interface defined.
    for mat in set(material_registry.objects()) - materials_used:
        if type(mat) is matlib.Rigid:
            material_registry.remove_object(mat)
    _fixup_ordinal_ids(material_registry)

    # Set solver module
    physics = auto_physics([m for m in material_registry.objects()])
    e_module = etree.SubElement(root, "Module")
    e_module.attrib["type"] = physics.value
    # Warn if there's an incompatibility between requested materials and
    # physics.
    for mat in material_registry.objects():
        # Extract delegate material object from OrientatedMaterial
        # so that we can check it.  TODO: This is a hack; find a
        # cleaner solution that doesn't require special-casing the
        # module compatibility check.
        if isinstance(mat, matlib.OrientedMaterial):
            checked_mat = mat.material
        else:
            checked_mat = mat
        if type(checked_mat) in physics_compat_by_mat:
            if physics not in physics_compat_by_mat[type(checked_mat)]:
                raise ValueError(
                    f"Material `{type(mat)}` is not listed as compatible with Module {physics}"
                )

    # Write material tags.  Sort by id to get around FEBio bug (FEBio ignores the ID
    # attribute and just uses tag order).
    e_Material = etree.SubElement(root, "Material")
    for mat_id, mat in sorted(material_registry.pairs("ordinal_id")):
        e = material_to_feb(mat, model)
        e.attrib["id"] = str(mat_id + 1)  # FEBio XML uses 1-indexed IDs
        e.attrib["name"] = material_registry.names(mat)[0]
        if mat in body_for_rigid_material:
            body = body_for_rigid_material[mat]
            if isinstance(body, ImplicitBody):
                e.insert(0, etree.Comment("Implicit rigid body"))
            if hasattr(body, "center_of_mass"):
                etree.SubElement(e, "center_of_mass").text = vec_to_text(
                    body.center_of_mass
                )
        e_Material.append(e)

    # Write Geometry
    domains = list_domains(model)
    for e in fx.mesh_xml(model, domains, material_registry):
        root.append(e)

    # Write MeshData.  Have to do this before handling boundary
    # conditions because some boundary conditions have part of their
    # values stored in MeshData.
    e_meshdata, e_elemsets = fx.xml_meshdata(model)
    meshdata_parent = get_or_create_xml(root, fx.ELEMENTDATA_PARENT)
    for e in e_meshdata:
        meshdata_parent.append(e)
    elementset_parent = get_or_create_xml(root, fx.ELEMENTSET_PARENT)
    for e in e_elemsets:
        elementset_parent.append(e)

    e_boundary = etree.SubElement(root, "Boundary")

    # Write contact constraints
    contact_constraints = [
        c for c in model.constraints if isinstance(c, ContactConstraint)
    ]
    e_Contact = contact_section(
        contact_constraints, model, named_surface_pairs, named_contacts, fx
    )
    root.append(e_Contact)

    e_constraints = etree.SubElement(root, "Constraints")

    e_loaddata = etree.SubElement(root, "LoadData")

    Output = etree.SubElement(root, "Output")

    # Typical MKS constants
    e_Constants = etree.Element("Constants")
    if "R" in model.constants:
        etree.SubElement(e_Constants, "R").text = str(model.constants["R"])
    if "temperature" in model.environment:
        etree.SubElement(e_Constants, "T").text = str(model.environment["temperature"])
    if "F" in model.constants:
        etree.SubElement(e_Constants, "Fc").text = str(model.constants["F"])
    # Add Globals/Constants if any defined; FEBio can't cope with an empty
    # Globals element.
    if len(e_Constants.getchildren()) > 0:
        e_Globals = etree.Element("Globals")
        e_Globals.append(e_Constants)
        root.insert(root.index(e_module) + 1, e_Globals)
        # ^ FEBio requires that first element must be <Module>

    # Create rigid interfaces (defined for explicit rigid bodies)
    for interface in model.constraints:
        if type(interface) is not RigidInterface:
            continue
        mat = rigid_material_for_body[interface.rigid_body]
        mat_id = material_registry.names(mat, nametype="ordinal_id")[0]
        mat_name = material_registry.names(mat, nametype="canonical")[0]
        nodeset_name = model.named["node sets"].names(interface.node_set)[0]
        e_parent = root.find(fx.IMPBODY_PARENT)
        e_child = fx.xml_rigid_nodeset_bc(nodeset_name, mat_name, mat_id + 1)
        e_parent.append(e_child)
    # Create rigid interfaces (node sets) for implicit rigid bodies.  In FEBio XML,
    # implicit rigid bodies are created by the rigid interface.
    for body in rigid_material_for_body:
        if isinstance(body, ImplicitBody):
            mat = rigid_material_for_body[body]
            mat_id = material_registry.names(mat, nametype="ordinal_id")[0]
            mat_name = material_registry.names(mat, nametype="canonical")[0]
            if version == "2.0":
                # FEBio XML 2.0 puts rigid bodies under §Constraints
                e_interface = etree.SubElement(e_Contact, "contact", type="rigid")
                for i in body.interface:
                    etree.SubElement(
                        e_interface, "node", id=str(i + 1), rb=str(mat_id + 1)
                    )
            else:
                # Get or create the node set
                nodeset_name = model.named["node sets"].get_or_create_name(
                    f"{mat_name}_interface", body.interface
                )
                add_nodeset(root, nodeset_name, body.interface, febioxml_module=fx)
                e_parent = root.find(fx.IMPBODY_PARENT)
                e_child = fx.xml_rigid_nodeset_bc(nodeset_name, mat_name, mat_id + 1)
                e_parent.append(e_child)

    # Write other global constraints / conditions / BCs and anything that goes in
    # global <Boundary>
    #
    # Write fixed nodal constraints to global <Boundary>
    e_boundary = root.find("Boundary")
    for e in fx.xml_node_fixed_bcs(model.fixed["node"], model.named["node sets"]):
        e_boundary.append(e)
    # TODO: Write time-varying nodal constraints to global <Boundary>

    # Write global (step-independent) fixed and variable body constraints to global
    # <Boundary> or <Rigid> sections.
    e_rb_cond_parent = get_or_create_xml(root, fx.BODY_COND_PARENT)
    # Group fixed rigid body conditions by body, since we'll need to
    # write an XML element for each body.
    body_bcs = defaultdict(dict)
    for (dof, var), bodies in model.fixed["body"].items():
        for body in bodies:
            # Match the dictionary format used for variable conditions;
            # just leave out the scale and relative keys.
            body_bcs[body][dof] = {"variable": var, "sequence": "fixed"}
    # Add the variable rigid body conditions
    for body, conditions in model.varying["body"].items():
        for dof, condition in conditions.items():
            body_bcs[body][dof] = condition
    # Create the body condition XML elements
    for body, constraints in body_bcs.items():
        for e_body_cond in fx.xml_body_constraints(
            body,
            constraints,
            material_registry,
            rigid_material_for_body,
            model.named["sequences"],
        ):
            e_rb_cond_parent.append(e_body_cond)

    # Output section
    plotfile = etree.SubElement(Output, "plotfile", type="febio")
    if not model.output["variables"]:  # empty list
        output_vars = ["displacement", "stress", "relative volume"]
        if physics == Physics.BIPHASIC:
            output_vars += ["effective fluid pressure", "fluid pressure", "fluid flux"]
        rigid_bodies_present = any(
            isinstance(m, matlib.Rigid) for m in material_registry.objects()
        )
        if rigid_bodies_present:
            output_vars += ["reaction forces"]
    else:
        output_vars = model.output["variables"]
    for var in output_vars:
        etree.SubElement(plotfile, "var", type=var)

    # Step section(s)
    e_step_parent = root.find(fx.STEP_PARENT)
    if e_step_parent is None:
        e_step_parent = etree.SubElement(root, fx.STEP_PARENT)
    cumulative_time = 0.0
    visited_implicit_bodies = set()
    step_idx = 0
    for step, step_name in model.steps:
        step_idx += 1
        if step_name is None:
            step_name = f"Step{step_idx}"
        e_step = step_xml(step, step_name, model.named["sequences"], physics, fx)
        e_step_parent.append(e_step)

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
        e_Boundary = etree.Element("Boundary")
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
        # TODO: Need to split by `relative`, since each <prescribe> cannot mix
        #  relative and non-relative boundary conditions.
        node_memo = defaultdict(dict)
        for node_id in step.bc["node"]:
            for dof in step.bc["node"][node_id]:
                bc = step.bc["node"][node_id][dof]
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
                        # Sequence gets an ID).
                        node_ids, scales, rel = zip(*bc)
                        e_bc, e_nodedata = fx.xml_node_var_bc(
                            model,
                            root,
                            node_ids,
                            scales,
                            seq,
                            dof,
                            var,
                            rel[0],
                            e_step.attrib["name"],
                        )
                        e_Boundary.append(e_bc)
                        nodedata_parent = get_or_create_xml(root, fx.NODEDATA_PARENT)
                        nodedata_parent.append(e_nodedata)
                    elif kind == "fixed":
                        raise NotImplementedError

        # Temporal (step-specific) contacts
        contacts = [c for c in step.bc["contact"] if isinstance(c, ContactConstraint)]
        e_Contact = contact_section(
            contacts, model, named_surface_pairs, named_contacts, fx
        )

        # Add <Boundary> and <Contact> elements to <Step>.  <Control>
        # was already added by step_xml().
        e_step.append(e_Boundary)
        e_step.append(e_Contact)

        # Add rigid body conditions to <Step>
        e_rb_cond_parent = get_or_create_xml(e_step, fx.BODY_COND_PARENT)
        for body, constraints in step.bc["body"].items():
            for e_body_cond in fx.xml_body_constraints(
                body,
                constraints,
                material_registry,
                rigid_material_for_body,
                model.named["sequences"],
            ):
                e_rb_cond_parent.append(e_body_cond)

    # Write XML elements for sequences (load curves) that are in the
    # model's named entity registry. Sequences can be referenced in a
    # lot of places, including boundary conditions, time stepper curves,
    # and any material parameter.  Therefore, as opposed to trying to
    # find them all here at the time of writing, we require that
    # whenever an XML element that references a sequence is added to
    # the XML tree elsewhere, said sequence is also added to the model's
    # named entity registry (which has to be done anyway because FEBio
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
    # Sequence IDs should be consecutive and start at 0
    if len(seq_ids) > 0:
        assert min(seq_ids) == 0
        assert max(seq_ids) == len(seq_ids) - 1
    # Add the sequences to the XML tree
    e_seq_parent = root.find(fx.SEQUENCE_PARENT)
    for seq_id in seq_ids:
        seq = model.named["sequences"].obj(seq_id, nametype="ordinal_id")
        # Apply offset to convert from step-local to global time, if required
        if seq.steplocal:
            t0 = seq_t0[seq]
        else:
            t0 = 0.0
        e_seq = fx.sequence_xml(seq, seq_id, t0=t0)
        e_seq_parent.append(e_seq)

    # Write named geometric entities & sets.  It is better to delay
    # writing named entities & sets until now so we don't accidentally
    # write the same set twice.
    #
    # Write any named node sets that were not already written.
    for nm, node_set in model.named["node sets"].pairs():
        e_nodeset = root.find(f"{fx.MESH_TAG}/NodeSet[@name='{nm}']")
        if e_nodeset is None:
            add_nodeset(root, nm, node_set, febioxml_module=fx)
    # Write *all* named face sets ("surfaces")
    surface_parent = find_unique_tag(root, fx.MESH_TAG)
    for nm, face_set in model.named["face sets"].pairs():
        e_surface = etree.SubElement(surface_parent, "Surface", name=nm)
        for i, face in enumerate(face_set):
            e_surface.append(face_xml(face, i))
    # Write *all* named surface pairs
    for nm, (primary, secondary) in named_surface_pairs.pairs():
        e_surfpair = fx.surface_pair_xml(
            model.named["face sets"], primary, secondary, nm
        )
        surface_parent.append(e_surfpair)
    # TODO: Handle element sets too.

    tree = etree.ElementTree(root)
    return tree


def write_xml(tree, f: BinaryIO):
    """Write an XML tree to a .feb file"""
    tree.write(f, pretty_print=True, xml_declaration=True, encoding="utf-8")


def write_feb(model, f, version="3.0"):
    """Write model's FEBio XML representation to a file object.

    :param model: The model object to write to FEBio XML.

    :param f: File-like object to write to.

    :param version: FEBio XML version to target.

    Usage Example::

        with open(pth, "wb") as f:
            write_feb(model, f, version="3.0")

    """
    tree = xml(model, version=version)
    write_xml(tree, f)
