from copy import copy
from enum import Enum
from operator import itemgetter
from typing import Hashable, NewType, Union
from warnings import warn

import numpy as np

# Set tolerances
_DEFAULT_TOL = 10 * np.finfo(float).eps

ZeroIdxID = NewType("ZeroIdxID", int)
OneIdxID = NewType("OneIdxID", int)
NodeID = NewType("NodeID", ZeroIdxID)


def _validate_dof(dof, body=False):
    allowed_dofs = ["x1", "x2", "x3", "fluid", "temperature", "charge"]
    if body:
        allowed_dofs += ["α1", "α2", "α3"]
    if not dof in allowed_dofs:
        msg = f"{dof} is not a supported axis type.  The supported degrees of freedom are {','.join(allowed_axes)}."
        raise ValueError(msg)


class Body:
    """A geometric body of elements.

    Explicit rigid bodies should be constructed using this type.

    """

    def __init__(self, elements):
        self.elements = elements
        # self.master_inode = elements[0].ids[0]

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


class NodeSet(frozenset):
    """Set of node IDs."""

    # TODO: Set operations should return a NodeSet


class FaceSet(frozenset):
    """Set of face IDs."""


class ElementSet(frozenset):
    """Set of element IDs."""


class ImplicitBody:
    """A geometric body defined by its interface with a mesh."""

    def __init__(self, mesh, interface: NodeSet, material=None):
        """Constructor for ImplicitBody object.

        interface := a collection of node ids

        At some some point in the future, support may be added to define
        an interface using a list of faces.

        """
        self.mesh = mesh
        self.interface = interface
        # ^ NameRegistry needs this to be hashable
        self.material = material


class RigidInterface:
    """A rigid constraint between a rigid body and a node set."""

    # Future enhancement: There is no particular reason that a rigid
    # interface should be restricted to an explicit rigid body and a
    # node set.  Two node sets work fine; just create an implicit rigid
    # body for one of them.

    def __init__(self, rigid_body, node_set):
        self.rigid_body = rigid_body
        self.node_set = node_set


class ContactConstraint:
    """A constraint defining contact between two surfaces."""

    def __init__(
        self,
        leader,
        follower,
        algorithm=None,
        auto_penalty=True,
        auto_penalty_scale=1,
        penalty_factor=None,
        augmented_lagrange=False,
        passes=1,
        symmetric_stiffness=False,
        tension=False,
        **kwargs,
    ):
        self.leader = leader
        self.follower = follower
        self.algorithm = algorithm
        self.tension = tension
        self.augmented_lagrange = augmented_lagrange
        self.symmetric_stiffness = symmetric_stiffness
        self.passes = passes
        if auto_penalty:
            self.penalty = {"type": "auto", "factor": auto_penalty_scale}
            if penalty_factor is not None:
                warn(
                    "A value for `penalty_factor` was provided, but automatic penalty factor calculation was also requested.  The `penalty_factor` value will not be used.  You may want to set `auto_penalty_scale` instead."
                )
        else:
            assert penalty_factor is not None
            self.penalty = {"type": "manual", "factor": penalty_factor}
        self.other_params = {}
        for k, v in kwargs.items():
            self.other_params[k] = v
        # TODO: Warn if two passes are specified and at least one of the
        # surfaces belongs to a rigid body.


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
    of the same type.

    """

    def __init__(self):
        # Don't use defaultdict because we don't want to create slots accidentally
        self._from_name = {}  # dict: nametype → (dict: name → obj)
        self._from_name["canonical"] = {}  # dict: name → obj
        self._from_object = {}  # dict: obj → (dict: nametype → set of names)

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

        `base_name` := string used as the first part of the autogenerated
        name.

        `item` := the item that needs a name to be retrieved / created.

        Returns the new name, formatted accoridng to {base_name}_{i}
        where i is a integer to disambiguate the name from any existing
        names in the registry.

        """

        # Check if there's an existing name
        try:
            names = self.names(item, nametype=nametype)
        except KeyError:
            # Find the first integer not already used as a suffix for the name
            i = 1
            while f"{base_name}{i}" in self.namespace(nametype):
                i += 1
            # Create a name using the unused integer
            name = f"{base_name}{i}"
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
