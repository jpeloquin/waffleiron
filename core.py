import dataclasses
from copy import copy
from dataclasses import dataclass
from enum import Enum
from operator import itemgetter
from typing import Hashable, Iterable, NewType, Union, Optional
from warnings import warn

import numpy as np

# Set tolerances
_DEFAULT_TOL = 10 * np.finfo(float).eps

CONSTANT_R = 8.31446261815324  # J/mol·K
CONSTANT_F = 96485.33212  # C/mol

ZeroIdxID = NewType("ZeroIdxID", int)
OneIdxID = NewType("OneIdxID", int)
NodeID = NewType("NodeID", ZeroIdxID)


def _validate_dof(dof, body=False):
    allowed_dofs = ["x1", "x2", "x3", "fluid", "temperature", "charge"]
    if body:
        allowed_dofs += ["α1", "α2", "α3"]
    if not dof in allowed_dofs:
        msg = f"{dof} is not a supported axis type.  The supported degrees of freedom are {','.join(allowed_dofs)}."
        raise ValueError(msg)


class NodeSet(frozenset):
    """Set of node IDs

    Recall that in Waffleiron, node IDs are zero-indexed.
    """

    # TODO: Set operations should return a NodeSet


class FaceSet(frozenset):
    """Set of facets (tuples of node IDs)"""

    # A FaceSet stores tuples of node IDs because facets are not represented elsewhere
    # by IDs or objects, so the FaceSet must contain all the information to define each
    # facet.


class ElementSet(frozenset):
    """Set of element IDs."""


class ImplicitBody:
    """A geometric body defined by its interface with a mesh."""

    # TODO: Get rid of material; it's an FEBio-ism

    def __init__(self, mesh, interface: NodeSet, material=None):
        """Constructor for ImplicitBody object.

        interface := a collection of node ids

        At some point in the future, support may be added to define an interface
        using a list of faces.

        """
        # NameRegistry needs the interface node set to be hashable
        if not isinstance(interface, NodeSet):
            interface = NodeSet(interface)
        self.mesh = mesh
        self.interface = interface
        self.material = material


class RigidInterface:
    """A rigid constraint between a rigid body and a node set."""

    # Future enhancement: There is no particular reason that a rigid
    # interface should be restricted to an explicit rigid body and a
    # node set.  Two node sets work fine; just create an implicit rigid
    # body for one of them.

    # TODO: What's the difference between a rigid interface with no associated explicit
    #  rigid body and an implicit rigid body?

    def __init__(self, rigid_body, node_set):
        self.rigid_body = rigid_body
        self.node_set = node_set


class Body:
    """A geometric body of elements.

    Explicit rigid bodies should be constructed using this type.

    """

    def __init__(self, elements: Iterable, center_of_mass=None):
        self.elements: Sequence = elements
        # self.master_inode = elements[0].ids[0]
        # TODO: calculate center of mass by volume averaging.  Requires that Element
        # support integration in space, not just its natural basis.
        if center_of_mass is not None:
            self.center_of_mass = center_of_mass
        else:
            M = np.array([0, 0, 0])
            total_volume = 0
            for e in self.elements:
                c = e.centroid()
                volume = e.integrate(lambda e, r: 1.0)
                M = M + volume * c
                total_volume += volume
            self.center_of_mass = M / total_volume

    def nodes(self):
        """Return (node_ids, xnodes) for this body."""
        nids = set()
        for e in self.elements:
            nids.update(e.ids)
        nids = np.array([i for i in nids])
        parent_mesh = next(iter(self.elements)).mesh
        # ^ assumption: all elements are from same mesh
        xnodes = np.array([parent_mesh.nodes[i] for i in nids])
        return nids, xnodes


@dataclass(eq=False)
class ContactConstraint(object):
    """A constraint defining contact between two surfaces."""

    # Only include parameters here if they are supported by all contact algorithms,
    # with the same default values.
    leader: FaceSet
    follower: FaceSet

    two_pass: bool = False
    penalty_factor: float = 1  # penalty in FEBio XML
    auto_penalty: bool = False
    use_augmented_lagrange: bool = False  # laugon in FEBio XML
    augmented_lagrange_rtol: float = 0.1  # tolerance in FEBio XML
    augmented_lagrange_gapnorm_atol: Optional[float] = None  # gaptol in FEBio XML
    projection_tol: float = 0.01  # search_tol in FEBio XML

    @property
    def values(self):
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in ("leader", "follower")
        }

    def __init__(self):
        raise NotImplementedError(
            f"{self.__class__} is a base class for specific contact implementations."
        )

    def __post_init__(self):
        if not isinstance(self.leader, FaceSet):
            self.leader = FaceSet(self.leader)
        if not isinstance(self.follower, FaceSet):
            self.follower = FaceSet(self.follower)


@dataclass(init=True, eq=False)
class ContactSlidingNodeOnFacet(ContactConstraint):
    """Sliding node on facet (N2F) contact.

    'sliding-node-on-facet' in FEBio XML

    """

    augmented_lagrange_minaug: int = 0
    augmented_lagrange_maxaug: int = 10

    max_segment_updates: Optional[int] = None

    friction_coefficient: Optional[float] = None
    friction_penalty: Optional[float] = None
    # tangential_stiffness_scale not supported in FEBio 4.3.0, contrary to docs


@dataclass(init=True, eq=False)
class ContactSlidingFacetOnFacet(ContactConstraint):
    """Sliding facet on facet (F2F) contact.

    'sliding-facet-on-facet' in FEBio XML.

    """

    update_penalty: bool = False
    augmented_lagrange_minaug: int = 0
    augmented_lagrange_maxaug: int = 10
    smoothed_lagrangian: bool = False
    search_scale: float = 1.0


@dataclass(init=True, eq=False)
class ContactSlidingElastic(ContactConstraint):
    """Sliding node on facet (N2F) contact.

    'sliding-node-on-facet' in FEBio XML

    """

    update_penalty: bool = False

    augmented_lagrange_minaug: int = 0
    augmented_lagrange_maxaug: int = 10
    smoothed_lagrangian: bool = False
    symmetric_stiffness: bool = False

    max_segment_updates: Optional[int] = None
    search_scale: float = 1.0

    friction_coefficient: Optional[float] = None
    tension: bool = False
    # tangential_stiffness_scale not supported in FEBio 4.3.0, contrary to docs


@dataclass(init=True, eq=False)
class ContactTiedElastic(ContactConstraint):
    """Tied elastic contact

    'tied-elastic' in FEBio XML.

    """

    augmented_lagrange_minaug: int = 0
    augmented_lagrange_maxaug: int = 10

    # pressure_penalty_factor: float = 1.0
    # ^ only recognized by FEBio 4, so I guess treat it as FEBio XML ≥ 4 only?
    symmetric_stiffness: bool = False
    search_scale: float = 1.0


def _canonical_face(face):
    """Return the canonical face tuple.

    The canonical face tuple is the ordering of the face nodes such
    that the lowest node id is first.  To achieve this ordering, only
    rotational shifts are allowed.

    """
    i, inode = min(enumerate(face), key=itemgetter(1))
    face = tuple(face[i:] + face[:i])
    return face


def _e_bb(elements):
    """Create bounding box array from element list."""
    bb_max = np.vstack([np.max(e.nodes, axis=0) for e in elements])
    bb_min = np.vstack([np.min(e.nodes, axis=0) for e in elements])
    bb = (bb_min, bb_max)
    return bb


class Interpolant(Enum):
    STEP = "step"
    LINEAR = "linear"
    SPLINE = "spline"


class Extrapolant(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    REPEAT = "repeat"
    REPEAT_CONTINUOUS = "repeat_continuous"


class Sequence:
    """A basic time-varying sequence.

    Defined by control points + interpolation method + extrapolation
    method.

    """

    def __init__(
        self,
        seq,
        interp: Union[Interpolant, str],
        extrap: Union[Extrapolant, str],
        steplocal: bool = True,
    ):
        # Input checking.  Technically, mypy should detect this, but
        # leave it in for now as mypy is not that widespread
        # (2021-02-09).
        if isinstance(interp, str):
            interp = Interpolant(interp)
        if not isinstance(interp, Interpolant):
            raise ValueError(
                f"Function argument `interp` has value `{interp}` with type `{type(interp)}`, but must have type `{Interpolant}`."
            )
        if isinstance(extrap, str):
            extrap = Extrapolant(extrap)
        if not isinstance(extrap, Extrapolant):
            raise ValueError(
                f"Function argument `extrap` has value `{extrap}` with type `{type(extrap)}`, but must have type `{Extrapolant}`."
            )
        # Parameters
        self.points = seq
        self.interpolant = interp
        self.extrapolant = extrap
        self.steplocal = steplocal


# TODO: Make ScaledSequence a Sequence


class ScaledSequence:
    """A time-varying sequence proportional to another sequence.

    Defined by a scaling factor and base sequence.  Since multiple
    objects may reference the same base sequence, ScaledSequence does
    not provide functions to modify the base sequence.

    """

    def __init__(self, sequence: Sequence, scale: float):
        self.scale = scale
        self.sequence = sequence

    @property
    def points(self):
        return self.sequence.points

    @property
    def interpolant(self):
        return self.sequence.interpolant

    @property
    def extrapolant(self):
        return self.sequence.extrapolant


class NameRegistry:
    """Mapping between names and objects.

    Provides `name + nametype → object` for all of an object's names and
    `object → (names + nametype)s`.  An object may have multiple names
    of the same nametype.

    """

    # TODO: It's very inconvenient to view all the names.  One of the functions
    #  should list every name in every namespace.

    def __init__(self):
        # Don't use defaultdict because we don't want to create slots accidentally
        self._from_name = {}  # dict: nametype → (dict: name → obj)
        self._from_name["canonical"] = {}  # dict: name → obj
        self._from_object = {}  # dict: obj → (dict: nametype → set of names)

    def __repr__(self):
        return str(self._from_name)

    def __getitem__(self, namespace):
        return self.get_namespace(namespace)

    def add(self, name, obj, nametype="canonical"):
        """Add a name for an object."""
        # Add the name to the name → object map
        if nametype not in self._from_name:
            self._from_name[nametype] = {}
        self._from_name[nametype][name] = obj
        # Add the name to the object → names map
        if obj not in self._from_object:
            self._from_object[obj] = {}
        if nametype not in self._from_object[obj]:
            self._from_object[obj][nametype] = set()
        self._from_object[obj][nametype].add(name)

    def remove_name(self, name, nametype="canonical"):
        """Remove a name for an object."""
        obj = self._from_name[nametype][name]
        # Remove the name from the name → object map
        del self._from_name[nametype][name]
        # Remove the name from the object → names map
        self._from_object[obj][nametype].remove(name)

    def remove_object(self, obj):
        """Remove all names for an object"""
        # Remove the object from the name → object maps
        for nametype in self._from_object[obj]:
            names = self._from_object[obj][nametype]
            for name in names:
                del self._from_name[nametype][name]
        # Remove all applicable names from the object → names map
        del self._from_object[obj]

    def names(self, obj, nametype="canonical"):
        """Return sorted tuple of names for object.

        The tuple is sorted so that picking the first name, for example,
        is deterministic.

        """
        return tuple(sorted(self._from_object[obj][nametype]))

    def obj(self, name, nametype="canonical"):
        """Return object by name (and type of name)."""
        if not nametype in self._from_name:
            raise KeyError(f"namespace '{nametype}' does not exist")
        return self._from_name[nametype][name]

    def get_or_create_name(
        self,
        base_name: str,
        item: Hashable,
        nametype="canonical",
    ):
        """Get or create a unique name for an item (mutates!).

        :param base_name: String used as the first part of the autogenerated name.

        :param item: The item that needs a name to be retrieved / created.

        Returns the new name, formatted accoridng to {base_name}_{i} where i is a
        integer to disambiguate the name from any existing names in the registry.

        """

        # Check if there's an existing name
        try:
            names = self.names(item, nametype=nametype)
        except KeyError:
            # Find the first integer not already used as a suffix for the name
            i = 1
            while f"{base_name}_{i}" in self.namespace(nametype):
                i += 1
            # Create a name using the unused integer
            name = f"{base_name}_{i}"
            # Update the dictionary so the new name persists
            self.add(name, item, nametype)
            return name
        return names[0]

    def nametypes(self):
        """Return nametypes"""
        return self._from_name.keys()

    def namespace(self, nametype="canonical"):
        """Return names of a given nametype."""
        return self._from_name.setdefault(nametype, {}).keys()

    def objects(self):
        """Return all named objects.

        This function is analogous to dict.values().

        """
        return self._from_object.keys()

    def pairs(self, nametype="canonical"):
        """Return iterable over (name, obj) pairs for nametype.

        This function is analogous to dict.items().

        """
        return self._from_name.setdefault(nametype, {}).items()

    def get_namespace(self, nametype="canonical"):
        """Return name → obj map (dict) for nametype."""
        return self._from_name.setdefault(nametype, {})

    def __copy__(self):
        """Copy dicts but not named objects."""
        new = type(self)()
        for nametype in self._from_name:
            new._from_name[nametype] = copy(self._from_name[nametype])
        new._from_object = {
            obj: {
                nametype: copy(self._from_object[obj][nametype])
                for nametype in self._from_object[obj]
            }
            for obj in self._from_object
        }
        return new
