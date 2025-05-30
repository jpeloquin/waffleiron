"""Helper functions for working with FEBio XML"""

import warnings
from collections import namedtuple, defaultdict
import os
from math import cos, radians, sin
from pathlib import Path, PurePath
from typing import Union, Tuple
import urllib.request

from lxml import etree
import numpy as np

from .core import (
    Body,
    ContactSlidingElastic,
    ContactSlidingFacetOnFacet,
    ContactSlidingNodeOnFacet,
    ContactTiedElastic,
    ImplicitBody,
    Sequence,
    ScaledSequence,
    Interpolant,
    Extrapolant,
    NodeSet,
    ZeroIdxID,
)
from .control import Physics
from .element import Hex27, Quad4, Tri3, Hex8, Penta6, Element
from . import material as matlib, FaceSet, ElementSet
from .material import (
    EllipsoidalDistribution,
    DeviatoricNeoHookean,
    OrientedMaterial,
    DeviatoricHGOFiber3D,
    SolidMixture,
    VolumetricHGO,
    FungOrthotropicElastic,
    NaturalNeoHookeanFiber,
    ExpAndLinearDCFiber,
    VolumetricLogInverse,
    VolumetricLinear,
    DeviatoricMooneyRivlin,
    ExponentialFiber,
    DeviatoricFiber,
    D3,
    DeviatoricSolidMixture,
)
from .math import orthonormal_basis, vec_from_sph

# Globals (see also end of file)

SUPPORTED_FEBIO_XML_VERS = ("2.0", "2.5", "3.0", "4.0")

#######################
# Material conversion #
#######################

# Special-case FEBio materials that don't fit cleanly into the general-purpose material
# frameworks should be represented here using special classes.


class UncoupledMooneyRivlin(D3):
    """Uncoupled Mooney–Rivlin FEBio material

    Waffleiron treats uncoupled materials as just another solid mixture; FEBio uses
    special syntax for them.

    """

    def __init__(self, c1, c2, bulk):
        self.solid = DeviatoricMooneyRivlin(c1=c1, c2=c2)
        self.bulk = bulk
        self.mixture = SolidMixture([self.solid, self.bulk])

    def __getattr__(self, item):
        return getattr(self.mixture, item)


class TransIsoMooneyRivlinFEBio(D3):
    """Uncoupled Mooney–Rivlin + uncoupled piecewise exponential–linear fiber with discontinuous elasticity

    Matches "trans iso Mooney-Rivlin" material in FEBio, which is simply a mixture of
    equisting constitutive models.  Except that the active contraction option is not
    supported.

    """

    def __init__(self, c1, c2, c3, c4, λ1, c5, bulk):
        self.solid = DeviatoricMooneyRivlin(c1=c1, c2=c2)
        self.fibers = DeviatoricFiber(
            ExpAndLinearDCFiber(ξ=c3, α=c4, λ1=λ1, E=c5), np.array([1, 0, 0])
        )
        self.bulk = bulk
        self.mixture = SolidMixture(
            [DeviatoricSolidMixture([self.solid, self.fibers]), self.bulk]
        )

    def __getattr__(self, item):
        return getattr(self.mixture, item)


class UncoupledHGOFEBio(D3):
    """Uncoupled Holzapfel–Gasser–Ogden FEBio material

    An uncoupled Holzapfel–Gasser–Ogden material in FEBio is a solid mixture of
    UncoupledHGOMatrix + 2 × OrientedMaterial UncoupledHGOFiber3D + VolumetricHGO, with
    the orientated material parts parameterized by a single angle, γ.  This combination
    would be difficult to convert to/from FEBio XML, so it gets a bespoke class.

    Seems like "Gasser-Ogden–Holzapfel" was added in FEBio 2.2, which used a different
    bulk model.  The "Holzapfel-Gasser-Ogden" material as represented here was
    introduced in FEBio 3.2.

    """

    def __init__(self, c, k1, k2, γ, κ, K):
        self.azimuth = γ
        self.matrix = DeviatoricNeoHookean(μ=c)
        self.fiber_pos = OrientedMaterial(
            DeviatoricHGOFiber3D(ξ=k1, α=k2, κ=κ),
            np.array(
                [
                    [cos(radians(γ)), -sin(radians(γ)), 0],
                    [sin(radians(γ)), cos(radians(γ)), 0],
                    [0, 0, 1],
                ]
            ),
        )
        self.fiber_neg = OrientedMaterial(
            DeviatoricHGOFiber3D(ξ=k1, α=k2, κ=κ),
            np.array(
                [
                    [cos(radians(γ)), sin(radians(γ)), 0],
                    [-sin(radians(γ)), cos(radians(γ)), 0],
                    [0, 0, 1],
                ]
            ),
        )
        self.bulk = VolumetricHGO(K / 2)
        self.material = SolidMixture(
            [
                self.matrix,
                self.fiber_pos,
                self.fiber_neg,
                self.bulk,
            ]
        )

    def tstress(self, F, **kwargs):
        return self.material.tstress(F, **kwargs)


class VerbatimXMLMaterial:
    """Store FEBio XML for a material that is not yet implemented

    A future enhancement would be to make the object aware of sequences that were used
    to define time-varying material parameters, and renumber them as appropriately
    during export.

    """

    def __init__(self, xml: Union[str, etree.ElementTree]):
        if isinstance(xml, str):
            xml = etree.fromstring(xml)
        self.xml = xml


def read_fung_orthotropic(e, seqs: dict):
    """Return FungOrthotropic material"""
    return FungOrthotropicElastic(
        E1=read_parameter(find_unique_tag(e, "E1", req=True), seqs),
        E2=read_parameter(find_unique_tag(e, "E2", req=True), seqs),
        E3=read_parameter(find_unique_tag(e, "E3", req=True), seqs),
        G12=read_parameter(find_unique_tag(e, "G12", req=True), seqs),
        G23=read_parameter(find_unique_tag(e, "G23", req=True), seqs),
        G31=read_parameter(find_unique_tag(e, "G31", req=True), seqs),
        ν12=read_parameter(find_unique_tag(e, "v12", req=True), seqs),
        ν23=read_parameter(find_unique_tag(e, "v23", req=True), seqs),
        ν31=read_parameter(find_unique_tag(e, "v31", req=True), seqs),
        c=read_parameter(find_unique_tag(e, "c", req=True), seqs),
        K=read_parameter(find_unique_tag(e, "k", req=True), seqs),
    )


def read_uncoupled_bulk(e, seqs: dict):
    """Return bulk compressibility model for uncoupled material"""
    e_pressure_model = find_unique_tag(e, "pressure_model")
    K = read_parameter(find_unique_tag(e, "k", req=True), seqs)
    if e_pressure_model is not None:
        law = e_pressure_model.text
    else:
        law = "default"
    model = {
        "default": VolumetricLogInverse,
        "0": VolumetricLogInverse,
        "NIKE3D": VolumetricHGO,
        "1": VolumetricHGO,
        "Abaqus": VolumetricLinear,
        "2": VolumetricLinear,
        "Abaqus (GOH)": VolumetricHGO,
        "3": VolumetricHGO,
    }[law]
    if model is VolumetricHGO:
        K = 0.5 * K
    return model(K)


def read_mooney_rivlin(e, seqs: dict):
    """Return UncoupledMooneyRivlin material"""
    c1 = read_parameter(find_unique_tag(e, "c1", req=True), seqs)
    c2 = read_parameter(find_unique_tag(e, "c2", req=True), seqs)
    bulk = read_uncoupled_bulk(e, seqs)
    return UncoupledMooneyRivlin(c1, c2, bulk)


def read_trans_iso_mooney_rivlin(e, seqs: dict):
    """Return TransIsoMooneyRivlinFEBio (uncoupled) material"""
    c1 = read_parameter(find_unique_tag(e, "c1", req=True), seqs)
    c2 = read_parameter(find_unique_tag(e, "c2", req=True), seqs)
    c3 = read_parameter(find_unique_tag(e, "c3", req=True), seqs)
    c4 = read_parameter(find_unique_tag(e, "c4", req=True), seqs)
    λ1 = read_parameter(find_unique_tag(e, "lam_max", req=True), seqs)
    c5 = read_parameter(find_unique_tag(e, "c5", req=True), seqs)
    bulk = read_uncoupled_bulk(e, seqs)
    return TransIsoMooneyRivlinFEBio(c1, c2, c3, c4, λ1, c5, bulk)


def read_holzapfel_gasser_ogden_xml(e, seqs: dict):
    """Return UncoupledHGOFEBio material"""
    return UncoupledHGOFEBio(
        c=read_parameter(find_unique_tag(e, "c", req=True), seqs),
        k1=read_parameter(find_unique_tag(e, "k1", req=True), seqs),
        k2=read_parameter(find_unique_tag(e, "k2", req=True), seqs),
        γ=read_parameter(find_unique_tag(e, "gamma", req=True), seqs),
        κ=read_parameter(find_unique_tag(e, "kappa", req=True), seqs),
        K=read_parameter(find_unique_tag(e, "k", req=True), seqs),
    )


def read_neo_hookean_fiber(e, seqs: dict):
    """Return NeoHookeanFiber material"""
    return matlib.NeoHookeanFiber(
        E=read_parameter(find_unique_tag(e, "mu", req=True), seqs),
    )


def read_natural_neo_hookean_fiber(e, seqs: dict):
    """Return NaturalNeoHookeanFiber material"""
    return NaturalNeoHookeanFiber(
        E=read_parameter(find_unique_tag(e, "ksi", req=True), seqs),
        λ0=read_parameter(find_unique_tag(e, "lam0", req=True), seqs),
    )


def read_exponential_fiber(e, seqs: dict):
    """Return ExponentialFiber material"""
    return ExponentialFiber(
        ξ=read_parameter(find_unique_tag(e, "ksi", req=True), seqs),
        α=read_parameter(find_unique_tag(e, "alpha", req=True), seqs),
        β=read_parameter(find_unique_tag(e, "beta", req=True), seqs),
    )


def read_fiber_exp_linear(e, seqs: dict):
    """Return ExpεAndLinεDEFiber material"""
    return ExpAndLinearDCFiber(
        ξ=read_parameter(find_unique_tag(e, "c3", req=True), seqs),
        α=read_parameter(find_unique_tag(e, "c4", req=True), seqs),
        λ1=read_parameter(find_unique_tag(e, "lambda", req=True), seqs),
        E=read_parameter(find_unique_tag(e, "c5", req=True), seqs),
    )


def read_continuous_fiber_distribution_xml(e, seqs: dict):
    """Return fiber orientation distribution material"""
    dist_type = e.find("distribution").attrib["type"]
    fiber = read_material(e.find("fibers"), seqs)
    if dist_type == "ellipsoidal":
        d = vector_from_text(e.find("distribution/spa").text)
        # TODO: Support parametrized integration schemes
        return EllipsoidalDistribution(d, fiber)
    else:
        return NotImplementedError


def read_biphasic(e, seqs: dict):
    """Return biphasic material"""
    # Get solidity parameter.  FEBio doesn't require solidity in all instances,
    # but FEBio's default is zero, which is wrong.  So it should be present.
    e_solid_fraction = find_unique_tag(e, "phi0", req=True)
    solid_fraction = read_parameter(e_solid_fraction, seqs)
    # Permeability constitutive equation
    e_permeability = find_unique_tag(e, "permeability", req=True)
    perm_type = e_permeability.attrib["type"]
    if perm_type in xml_material_reader:
        permeability = xml_material_reader[perm_type](
            e_permeability, seqs, solid_fraction
        )
    else:
        perm_class = perm_class_from_name[perm_type]
        props = {c.tag: read_parameter(c, seqs) for c in e_permeability}
        props["phi0"] = solid_fraction  # needed for Holmes–Mow permeability
        permeability = perm_class.from_feb(**props)
    # Solid constituent
    constituents = [read_material(c, seqs) for c in e if c.tag == "solid"]
    if len(constituents) > 1:
        raise ValueError(
            f"A porelastic solid was encountered with {len(constituents)} solid constituents.  Poroelastic solids must have exactly one solid constituent.  The relevant poroelastic solid is at {e.base}:{e.sourceline}."
        )
    solid = constituents[0]
    return matlib.PoroelasticSolid(solid, permeability, solid_fraction)


def read_isotropic_exponential_permeability(
    e, seqs: dict, solid_volume_fraction, **kwargs
):
    """Return isotropic exponential permeability law"""
    # kwargs needed because some permeability laws need solid volume fraction,
    # which FEBio does not store in the permeability XML element
    return matlib.IsotropicExponentialPermeability(
        k0=read_parameter(find_unique_tag(e, "perm", req=True), seqs),
        M=read_parameter(find_unique_tag(e, "M", req=True), seqs),
        φ0_s=solid_volume_fraction,
    )


def read_referentially_transiso_permeability(
    e, seqs: dict, solid_volume_fraction, **kwargs
):
    """Return transversely isotropic Holmes–Mow permeability law"""
    return matlib.TransIsoHolmesMowPermeability(
        k0=read_parameter(find_unique_tag(e, "perm0", req=True), seqs),
        M0=read_parameter(find_unique_tag(e, "M0", req=True), seqs),
        α0=read_parameter(find_unique_tag(e, "alpha0", req=True), seqs),
        k1a=read_parameter(find_unique_tag(e, "perm1A", req=True), seqs),
        k2a=read_parameter(find_unique_tag(e, "perm2A", req=True), seqs),
        Ma=read_parameter(find_unique_tag(e, "MA", req=True), seqs),
        αa=read_parameter(find_unique_tag(e, "alphaA", req=True), seqs),
        k1t=read_parameter(find_unique_tag(e, "perm1T", req=True), seqs),
        k2t=read_parameter(find_unique_tag(e, "perm2T", req=True), seqs),
        Mt=read_parameter(find_unique_tag(e, "MT", req=True), seqs),
        αt=read_parameter(find_unique_tag(e, "alphaT", req=True), seqs),
        φ0_s=solid_volume_fraction,
    )


def read_prony_viscoelasticity(e, seqs: dict):
    """Return PronyViscoelasticity material"""
    γ = np.full(6, np.nan)
    τ = np.full(6, np.nan)
    n = 0
    for i in range(0, 6):
        e_γi = find_unique_tag(e, f"g{i + 1}", req=False)
        e_τi = find_unique_tag(e, f"t{i + 1}", req=False)
        if e_γi is not None and e_τi is not None:
            n = i + 1
        γ[i] = find_and_read_parameter(e, f"g{i + 1}", seqs, default=0)
        τ[i] = find_and_read_parameter(e, f"t{i + 1}", seqs, default=1)
    γ = γ[:n]
    τ = τ[:n]
    # γ0
    e_γ0 = find_unique_tag(e, "g0", req=False)
    if e_γ0 is not None:
        γ0 = read_parameter(e_γ0, seqs)
        if isinstance(γ0, (Sequence, ScaledSequence)):
            raise ValueError("γ0 should always be 1")
        γ = γ / γ0
    e_material = find_unique_tag(e, "elastic", req=True)
    material = read_material(e_material, seqs)
    return matlib.PronyViscoelasticity(material, γ, τ)


def read_rigid_material(e, seqs: dict):
    """Return rigid body material"""
    # Monkey-patch the center of mass into the material object; that seems the most
    # straightforward way to pass that information along to the Body class.
    mat = matlib.Rigid()
    e_com = e.find("center_of_mass")
    if e_com is not None:
        mat.center_of_mass = to_vec(e_com.text)
    return mat


# Map type attribute of <material>, <solid>, or <fiber> → function that returns
# waffleiron material class form the XML element
xml_material_reader = {
    "Fung-ortho-compressible": read_fung_orthotropic,
    "Mooney-Rivlin": read_mooney_rivlin,
    "trans iso Mooney-Rivlin": read_trans_iso_mooney_rivlin,
    "Holzapfel-Gasser-Ogden": read_holzapfel_gasser_ogden_xml,
    "fiber-NH": read_neo_hookean_fiber,
    "fiber-natural-NH": read_natural_neo_hookean_fiber,
    "fiber-exp-pow": read_exponential_fiber,
    "fiber-exp-linear": read_fiber_exp_linear,
    "continuous fiber distribution": read_continuous_fiber_distribution_xml,
    "biphasic": read_biphasic,
    "perm-exp-iso": read_isotropic_exponential_permeability,
    "perm-ref-trans-iso": read_referentially_transiso_permeability,
    "viscoelastic": read_prony_viscoelasticity,
    "rigid body": read_rigid_material,
}

material_from_xml_name = {
    "isotropic elastic": matlib.IsotropicElastic,
    "Holmes-Mow": matlib.HolmesMow,
    "fiber-pow-linear": matlib.PowerLinearFiber,
    "ellipsoidal fiber distribution": matlib.EllipsoidalPowerFiber,
    "neo-Hookean": matlib.NeoHookean,
    "solid mixture": matlib.SolidMixture,
    "Donnan equilibrium": matlib.DonnanSwelling,
    "multigeneration": matlib.Multigeneration,
    "orthotropic elastic": matlib.OrthotropicLinearElastic,
}
material_name_from_class = {v: k for k, v in material_from_xml_name.items()}


orientation_distribution_from_xml_type = {
    "ellipsoidal": matlib.EllipsoidalDistribution,
}

perm_class_from_name = {
    "perm-Holmes-Mow": matlib.IsotropicHolmesMowPermeability,
    "perm-const-iso": matlib.IsotropicConstantPermeability,
}
perm_name_from_class = {v: k for k, v in perm_class_from_name.items()}

# TODO: Redesign the compatibility system so that compatibility can be derived from the
#  material's type.  Although FEBio might not have a clean relationship between a
#  material's physics and its own support.
physics_compat_by_mat = {
    matlib.PoroelasticSolid: {Physics.BIPHASIC},
    matlib.Rigid: {Physics.SOLID, Physics.BIPHASIC},
    matlib.OrthotropicLinearElastic: {Physics.SOLID, Physics.BIPHASIC},
    matlib.IsotropicElastic: {Physics.SOLID, Physics.BIPHASIC},
    matlib.SolidMixture: {Physics.SOLID, Physics.BIPHASIC},
    matlib.NaturalNeoHookeanFiber: {Physics.SOLID, Physics.BIPHASIC},
    matlib.PowerLinearFiber: {Physics.SOLID, Physics.BIPHASIC},
    matlib.ExponentialFiber: {Physics.SOLID, Physics.BIPHASIC},
    matlib.HolmesMow: {Physics.SOLID, Physics.BIPHASIC},
    matlib.NeoHookean: {Physics.SOLID, Physics.BIPHASIC},
    matlib.PronyViscoelasticity: {Physics.SOLID, Physics.BIPHASIC},
}


##################
# Helper classes #
##################


BodyConstraint = namedtuple(
    "BodyConstraint",
    ["body", "dof", "variable", "constant", "sequence", "relative"],
    defaults=[None, None],
)
OptParameter = namedtuple("OptParameter", ["path", "fun", "default"])
ReqParameter = namedtuple("ReqParameter", ["path", "fun"])


##########################################################
# Model traversal functions to support XML read or write #
##########################################################


def list_domains(model):
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


def bcs_by_nodeset_and_var(fixed_conditions: dict):
    """Return BCs collated by node set name

    Meant to be used with Model.fixed["node"].  Might be useful in other situations.

    """
    by_nodeset = defaultdict(lambda: defaultdict(list))
    for (dof, var), nodeset in fixed_conditions.items():
        # Skip empty node sets
        if not nodeset:
            continue
        nodeset = NodeSet(nodeset)
        by_nodeset[nodeset][var].append(dof)
    return by_nodeset


def group_constraints_fixed_variable(constraints):
    fixed_constraints = []
    variable_constraints = []
    for dof, bc in constraints.items():
        if bc["sequence"] == "fixed":
            fixed_constraints.append((dof, bc))
        else:  # bc['sequence'] is Sequence
            variable_constraints.append((dof, bc))
    return fixed_constraints, variable_constraints


###################################
# Functions for reading FEBio XML #
###################################


def to_bool(s: str) -> bool:
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


def to_vec(s):
    """Convert string to sequence of int or float as appropriate."""
    tokens = s.split(",")
    if len(tokens) > 1:
        return [to_number(t) for t in tokens]
    else:
        raise ValueError("Provided string does not appear to be a sequence of values.")


def maybe_to_number(s):
    """Convert string to number if possible, otherwise return string."""
    try:
        return to_number(s)
    except ValueError:
        return s


def find_unique_tag(root: etree.Element, path, req=False) -> etree.Element:
    """Find and return a tag or an error if > 1 of same."""
    tags = root.findall(path)
    if len(tags) == 1:
        return tags[0]
    elif len(tags) > 1:
        raise ValueError(
            f"Multiple `{path}` tags in file `{os.path.abspath(root.base)}`"
        )
    else:
        if req:
            raise ValueError(
                f"Could not find required XML tag {path} relative to {root} at {root.base}:{root.sourceline}."
            )
        else:
            return None


def parse_nodeset_ref(s: str) -> Tuple[Union[NodeSet, FaceSet, ElementSet], str]:
    """Parse a node_set XML attribute's value

    Examples:
        "foo" → NodeSet named "foo"
        "@surface:bar": → NodeSet comprising the nodes in the FaceSet "bar"

    """
    if s.startswith("@surface:"):
        _, name = s.split(":")
        return FaceSet, name
    elif s.startswith("@elem_set:"):
        _, name = s.split(":")
        return ElementSet, name
    else:
        # As far as I can tell from the FEBio manual, other strings following the
        # '@type:name' pattern should be interpreted as simple names.  I.e., there is
        # no grammar, just special cases.
        return NodeSet, s


def read_nodeset_ref(s: str, mesh=None) -> NodeSet:
    """Return NodeSet object referenced by a node_set XML attribute

    :param mesh: Mesh instance.

    """
    cls, nm = parse_nodeset_ref(s)
    match cls.__name__:
        case NodeSet.__name__:
            return mesh.named["node sets"].obj(nm)
        case FaceSet.__name__:
            face_set = mesh.named["face sets"].obj(nm)
            return NodeSet(i for f in face_set for i in f)
        case ElementSet.__name__:
            element_ids = (lbl for lbl in mesh.named["element sets"].obj(nm))
            node_ids = [
                i
                for element_id in element_ids
                for i in mesh.named["elements"].obj(element_id).ids
            ]
            return NodeSet(node_ids)


def read_material_type(e):
    if "type" not in e.attrib:
        raise ValueError(
            f"Material is missing its `type` attribute in {e.base}:{e.sourceline}"
        )
    return e.attrib["type"]


def read_material(e, sequence_registry: dict):
    """Read a material from an XML element

    This function will not mutate `sequence_registry`.

    """

    def guess_matprops(e):
        """Return dictionary of scalar material properties

        Any XML element with a scalar numeric value is assumed to be a material
        property.

        """
        conprops = {}
        extprops = {}
        extension_tags = ("density",)
        for c in e:
            try:
                v = to_number(c.text)
            except ValueError:
                try:
                    v = to_vec(c.text)
                except ValueError:
                    continue
            if c.tag in extension_tags:
                extprops[c.tag] = v
            else:
                conprops[c.tag] = v
        return conprops, extprops

    # Check if the material type is fully supported
    material_type = read_material_type(e)
    orientation = read_material_orientation(e)
    # TODO: over time, migrate materials to reader functions.  `read_material` is
    #  growing out of control.
    if material_type in xml_material_reader:
        mat = xml_material_reader[material_type](e, sequence_registry)
        if orientation is not None:
            return OrientedMaterial(mat, orientation)
        else:
            return mat
    if material_type not in material_from_xml_name:
        warnings.warn(
            f"Reading material `{material_type}` from FEBio XML is not yet supported.  Its XML content will be stored in the Waffleiron model.  It will be reproduced verbatim (except for the material ID) if the model is written to FEBio XML."
        )
        return VerbatimXMLMaterial(e)
    cls = material_from_xml_name[material_type]
    if material_type == "solid mixture":
        constituents = [
            read_material(c, sequence_registry) for c in e if c.tag == "solid"
        ]
        # TODO: Shouldn't solid mixture support material orientation?
        material = cls(constituents)
    elif material_type == "multigeneration":
        # Constructing materials for the list of generations works just like a solid
        # mixture
        generations = []
        for g in e.findall("generation"):
            t = to_number(find_unique_tag(g, "start_time", req=True).text)
            solid = read_material(find_unique_tag(g, "solid", req=True))
            generations.append((t, solid))
        material = cls(generations)
    elif hasattr(cls, "from_feb") and callable(cls.from_feb):
        material = cls.from_feb(**guess_matprops(e)[0])
    else:
        material = cls(guess_matprops(e)[0])
    # Apply total orientation for the material (which may be a submaterial)
    if orientation is not None:
        material = matlib.OrientedMaterial(material, orientation)
    return material


def read_material_orientation(e):
    """Return orientation of a material

    :param e: <material> or <solid> XML element.

    The orientation may be returned as a 3-element vector or a matrix of 3
    orthonormal basis vectors.

    The material orientation is assumed to not vary with time.  Currently only scalar
    parameters can vary with time.

    """
    material_type = read_material_type(e)
    # Read material orientation in the form of <mat_axis> or <fiber>
    e_mat_axis = find_unique_tag(e, "mat_axis")
    e_fiber = find_unique_tag(e, "fiber")
    # Handle multiple orientation definitions
    if (e_mat_axis is not None) and (e_fiber is not None):
        # FEBio's documentation says that only one could be defined, but FEBio itself
        # accepts both, with undocumented handling regarding precedence.  So we raise
        # an error.
        raise ValueError(
            f"Found both <mat_axis> and <fiber> XML elements in material; only one may be present.  The material definition is at {e.base}:{e.sourcline}."
        )
    # <mat_axis>
    elif e_mat_axis is not None:
        if e_mat_axis.attrib["type"] == "vector":
            a = vector_from_text(
                find_unique_tag(e_mat_axis, "a", req=True).text, f=float
            )
            d = vector_from_text(
                find_unique_tag(e_mat_axis, "d", req=True).text, f=float
            )
            orientation = orthonormal_basis(a, d)
        elif e_mat_axis.attrib["type"] == "local":
            orientation = None
            # <mat_axis type="local"> is converted to a heterogeneous orientation
            # separately in FebReader.model via basis_mat_axis_local; no need to
            # handle it here.
        else:
            raise NotImplementedError(
                f"<mat_axis> orientation type '{e_mat_axis.attrib['type']}' is not yet implemented.  The relevant <mat_axis> element is at {e_mat_axis.base}:{e_mat_axis.sourceline}."
            )
    # <fiber>
    elif e_fiber is not None:
        fiber_orientation_type = e_fiber.attrib["type"]
        if fiber_orientation_type == "vector":
            orientation = vector_from_text(e_fiber.text, f=float)
        elif fiber_orientation_type == "angles":
            θ = to_number(find_unique_tag(e_fiber, "theta", req=True).text)
            φ = to_number(find_unique_tag(e_fiber, "phi", req=True).text)
            orientation = vec_from_sph(θ, φ)
        else:
            # Should support <fiber> types: local, spherical, cylindrical
            raise NotImplementedError(
                f"<fiber> orientation type '{fiber_orientation_type}' is not yet implemented.  The relevant <fiber> element is at {e_fiber.base}:{e_fiber.sourceline}."
            )
    else:
        orientation = None

    # Read "material property" orientation defined using <theta> and <phi> elements
    # on the material itself.  These *combine* with the <mat_axis> or <fiber>
    # material orientation / basis.
    e_theta = find_unique_tag(e, "theta")
    e_phi = find_unique_tag(e, "phi")
    if (e_theta is not None) and (e_phi is not None):
        θ = to_number(e_theta.text)
        φ = to_number(e_phi.text)
        matprop_orientation = vec_from_sph(θ, φ)
    elif (e_theta is not None) and (e_phi is None):
        # If either spherical angle is present, the other must be also
        raise ValueError(
            f'Found a <theta> element but no <phi> in material "{material_type}".  Both spherical angles are required to define a material orientation.  The relevant material is at {e.base}:{e.sourceline}.'
        )
    elif (e_phi is not None) and (e_theta is None):
        # If either spherical angle is present, the other must be also
        raise ValueError(
            f'Found a <phi> element but no <theta> in material "{material_type}".  Both spherical angles are required to define a material orientation.  The relevant material is at {e.base}:{e.sourceline}.'
        )
    else:
        matprop_orientation = None

    # Combine material property orientation and material orientation as appropriate.
    if matprop_orientation is not None:
        if orientation is None:
            orientation = matprop_orientation
        else:
            if orientation.ndim == 2:
                orientation = orientation @ matprop_orientation
            else:
                # `orientation` is just a vector.  Interpret it as indicating a
                # transformation from [1, 0, 0] to its value.
                raise NotImplementedError
    return orientation


def read_parameter(e, sequence_registry: dict[int, Sequence]):
    """Read a scalar parameter from an XML element.

    The parameter may be fixed or variable.  If variable, a Sequence or
    ScaledSequence will be returned.

    """
    # Check if this is a time-varying or fixed property
    if "lc" in e.attrib:
        # The property is time-varying
        seq_id = int(e.attrib["lc"]) - 1
        sequence = sequence_registry[seq_id]
        if e.text is not None and e.text.strip() != "":
            scale = to_number(e.text)
            return ScaledSequence(sequence, scale)
        else:
            return sequence
    else:
        # The property is fixed
        return to_number(e.text)


def find_and_read_parameter(root, path, seqs: dict[int, Sequence], default=None):
    """Find a scalar parameter's XML element and return its value"""
    e = find_unique_tag(root, path, req=default is None)
    if e is not None:
        return read_parameter(e, seqs)
    else:
        return default


def read_parameters(xml, paramdict):
    """Return scalar parameters for a dataclass's fields from an XML element"""
    # This might be overengineered (doesn't handle XML format revisions when they
    # introduce new attributes and children to some parameter elements)
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
                # XML element does not exist; use default
                params[k] = p.default
            elif len(e) > 1:
                parentpath = xml.getroottree().getpath(e.getparent())
                raise ValueError(
                    f"{xml.base}:{xml.sourceline} {parentpath} has {len(e)} {e.tag} elements.  It should have at most one."
                )
            else:  # len(s) == 1; one XML element exists
                e = e[0]
                if e.text is None:
                    # Use default
                    params[k] = p.default
                else:
                    params[k] = p.fun(e.text)
    return params


def read_mat_axis_xml(e: etree.Element):
    """Read material axes (basis) from an element in a mat_axis section

    :param e: XML element defining an element's material axes; e.g.,
    <elem lid="1"><a>1,0,0</a><d><0,1,0></d></elem>.

    """
    a = vector_from_text(e.find("a").text, f=float)
    d = vector_from_text(e.find("d").text, f=float)
    basis = orthonormal_basis(a, d)
    local_id = ZeroIdxID(int(e.attrib["lid"]) - 1)
    return local_id, basis


def read_point(text):
    x, y = text.split(",")
    return to_number(x), to_number(y)


def vector_from_text(text, f=to_number):
    return np.array([f(x) for x in text.split(",")])


def ids_from_text(text):
    return tuple(int(x) for x in text.split(","))


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
        # Decode file URI.  Seems to only be necessary on Windows.
        local_path = urllib.request.url2pathname(urllib.parse.urlparse(root.base).path)
        return Path(local_path).with_suffix(".log")
    elif len(paths) == 1:
        return paths[0]
    else:  # len(paths) > 1
        raise ValueError(
            f"{root} (root.base) has more than one `<logfile>` element with a `file` attribute, making the logfile name ambiguous."
        )


def normalize_xml(root):
    """Convert some items in FEBio XML to 'normal' representation.

    FEBio XML allows some items to be specified several ways.  To reduce
    the complexity of the code that converts FEBio XML to a waffleiron
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
        e_Step = etree.Element("Step")
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
            e_Step = etree.Element("Step")
            root.insert(1, e_Step)
        es_prescribe = e_rBoundary.findall("prescribe")
        # Do we need to create a Step/Boundary element?
        e_sBoundary = e_Step.find("Boundary")
        if len(es_prescribe) != 0 and e_sBoundary is None:
            e_sBoundary = etree.SubElement(e_Step, "Boundary")
        # Move the <prescribe> elements
        for e_prescribe in es_prescribe:
            e_rBoundary.remove(e_prescribe)
            e_sBoundary.append(e_prescribe)
        # Delete the <Boundary> element if it is now empty
        e_rBoundary = root.find("Boundary")
        if len(e_rBoundary) == 0:
            root.remove(e_rBoundary)
    return root


#######################################################################
# Helper functions to convert data structures to what FEBio XML needs #
#######################################################################

# These functions do not return XML; they just convert data structures.


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


###################################
# Functions for writing FEBio XML #
###################################


def body_mat_id(body, material_registry, implicit_rb_mats):
    """Return a material ID to define a rigid body in XML."""
    # Create or find the associated materials
    if isinstance(body, Body):
        # If an explicit body, its elements define its
        # materials.  We assume that the body is homogenous.
        mat = next(e for e in body.elements).material

    elif isinstance(body, ImplicitBody):
        mat = implicit_rb_mats[body]
        ids = material_registry.names(mat, nametype="ordinal_id")
    else:
        msg = (
            f"body {body} does not have a supported type.  "
            + "Supported body types are Body and ImplicitBody."
        )
        raise ValueError(msg)
    ids = material_registry.names(mat, nametype="ordinal_id")
    assert len(ids) == 1
    mat_id = ids[0]
    mat_name = material_registry.names(mat, nametype="canonical")[0]
    return mat_id, mat_name


def get_or_create_xml(root, path: Union[str, PurePath]):
    """Return XML element at path, creating it if needed"""
    path = str(path)
    if path == "" or path == ".":
        return root
    parent = root
    current = None
    parts = [p for p in path.split("/") if p != ""]
    for part in parts:
        current = find_unique_tag(parent, part)
        if current is None:
            current = etree.SubElement(parent, part)
        parent = current
    return current


def get_or_create_parent(root, path: Union[str, PurePath]):
    """Return second-to-last XML element from path, creating it if needed"""
    path = str(path)
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
    if isinstance(v, (int, np.integer)):
        return int_to_text(v)
    elif isinstance(v, float):
        return float_to_text(v)
    else:
        raise ValueError(
            f"Provided numeric value has type '{type(v).__name__}', which is not supported for conversion to XML."
        )


def vec_to_text(v):
    return ", ".join(float_to_text(a) for a in v)


def float_to_text(a):
    return f"{a:.16g}"


def property_to_xml(value, tag, seq_registry):
    """Convert a constant or variable property to FEBio XML"""
    if isinstance(value, Sequence):
        # Time-varying property, not scaled
        e = etree.Element(tag)
        seq_id = get_or_create_item_id(seq_registry, value)
        e.attrib["lc"] = str(seq_id + 1)
        e.text = "1"  # scale factor
    elif isinstance(value, ScaledSequence):
        # Time-varying property, scaled
        e = etree.Element(tag)
        seq_id = get_or_create_item_id(seq_registry, value.sequence)
        e.attrib["lc"] = str(seq_id + 1)
        e.text = num_to_text(value.scale)
    else:
        # Constant property
        e = const_property_to_xml(value, tag)
    return e


def const_property_to_xml(value, tag):
    """Convert a constant property to FEBio XML"""
    e = etree.Element(tag)
    e.text = to_text(value)
    return e


#########################
# Facts about FEBio XML #
#########################

# Define after the function definitions so that those functions can be used here

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

elem_cls_from_feb = {
    "quad4": Quad4,
    "tri3": Tri3,
    "hex8": Hex8,
    "hex27": Hex27,
    "penta6": Penta6,
}

# Map of ContactConstraint fields → elements relative to <contact>.  This really should
# be done for each contact algorithm.  According to the FEBio manual, all algorithms
# have the same defaults, but that (a) doesn't make sense and (b) is false, at least for
# `tolerance`.
CONTACT_PARAMS = {
    "tension": OptParameter("tension", to_bool, False),
    "penalty_factor": OptParameter("penalty", to_number, 1),
    "pressure_penalty_factor": OptParameter("pressure_penalty", to_number, 1),
    "two_pass": OptParameter("two_pass", to_bool, False),
    "auto_penalty": OptParameter("auto_penalty", to_bool, False),
    "update_penalty": OptParameter("update_penalty", to_bool, False),
    "symmetric_stiffness": OptParameter("symmetric_stiffness", to_bool, False),
    "use_augmented_lagrange": OptParameter("laugon", to_bool, False),
    "augmented_lagrange_rtol": OptParameter("tolerance", to_number, 0.1),
    "augmented_lagrange_gapnorm_atol": OptParameter("gaptol", maybe_to_number, 0.0),
    "augmented_lagrange_minaug": OptParameter("minaug", int, 0),
    "augmented_lagrange_maxaug": OptParameter("maxaug", int, 10),
    "smoothed_lagrangian": OptParameter("smooth_aug", to_bool, False),
    "friction_coefficient": OptParameter("fric_coeff", to_number, 0.0),
    "friction_penalty": OptParameter("fric_penalty", to_number, 0.0),
    "search_scale": OptParameter("search_radius", to_number, 1.0),
    "max_segment_updates": OptParameter("seg_up", int, 0),
    "tangential_stiffness_scale": OptParameter("ktmult", to_number, 1.0),
    "gap_tol": OptParameter("gaptol", to_number, 0.0),
    "projection_tol": OptParameter("search_tol", to_number, 0.01),
}


CONTACT_CLASS_FROM_XML = {
    "sliding-node-on-facet": ContactSlidingNodeOnFacet,
    "sliding-facet-on-facet": ContactSlidingFacetOnFacet,
    "sliding-elastic": ContactSlidingElastic,
    "tied-elastic": ContactTiedElastic,
}
CONTACT_NAME_FROM_CLASS = {v: k for k, v in CONTACT_CLASS_FROM_XML.items()}
