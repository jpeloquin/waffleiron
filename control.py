"""Module for controling FEBio.

This module includes functionality for working with FEBio-specific
control sections in FEBio xml files.

"""
# Base modules
from copy import deepcopy
import dataclasses
from dataclasses import dataclass
from enum import Enum
import math
from typing import Union

# Third party modules
import numpy as np

# Same-package modules
from .core import Sequence, Interpolant, Extrapolant
from .math import densify
import waffleiron.material as matlib


def auto_ticker(seq: Sequence, pts_per_segment: int = 1):
    """Return a ticker with an automatic "must point" curve in dtmax"""
    ticker = Ticker()
    duration = seq.points[-1][0] - seq.points[0][0]
    time = np.array([a for a, b in seq.points])
    dt = np.diff(time)
    dt = np.concatenate([dt, dt[-1:]])  # len(dt) == len(time)
    # Assign must point sequence
    curve_must_dt = densify([(a, b) for a, b in zip(time, dt)], n=pts_per_segment)
    # TODO: Densification should respect the curve's interpolant
    # Recalculate dt after densification.
    t = np.array(curve_must_dt)[:, 0]
    dt = np.diff(t)
    dt = np.concatenate([dt, dt[-1:]])  # len(dt) == len(time)
    curve_must_dt = [p for p in zip(t, dt)]
    seq_dtmax = Sequence(
        curve_must_dt, extrap=Extrapolant.CONSTANT, interp=Interpolant.STEP
    )
    ticker.dtmax = seq_dtmax
    # Calculate appropriate step size.  Need to work around FEBio bug
    # https://forums.febio.org/project.php?issueid=765.  FEBio skips
    # must points with t < step_size, so we must set step_size to a
    # value less than the time of the first must point.
    time_must = np.array([a for a, b in curve_must_dt])
    dt_must = np.diff(time_must)
    dt_min = np.min(dt_must)
    nsteps = math.ceil(duration / dt_min)
    dt_nominal = duration / nsteps
    if dt_min == dt_nominal:
        # Sometimes FEBio skips a must point even when the first time
        # point is coincident with the step_size.  Add an extra step to
        # avoid this.
        nsteps += 1
        dt_nominal = duration / nsteps
    ticker.n = nsteps
    ticker.dtnom = dt_nominal
    ticker.dtmin = 0.1 * dt_min
    return ticker


def auto_physics(materials):
    """Determine which module should be used to run the model.

    Currently only chooses between solid and biphasic.

    """
    module = Physics("solid")
    for m in materials:
        if isinstance(m, matlib.PoroelasticSolid):
            module = Physics("biphasic")
    return module


class SaveIters(Enum):
    """Values for FEBio plot_level setting

    This setting controls which iterations FEBio writes to the xplt
    file.

    """

    NEVER = "PLOT_NEVER"
    MAJOR = "PLOT_MAJOR_ITRS"
    MINOR = "PLOT_MINOR_ITRS"
    USER = "PLOT_MUST_POINTS"

    def __eq__(self, other):
        if isinstance(other, str):
            raise ValueError(
                "Please compare enums by identity: use `x is SaveIters.USER`.  This is meant to avoid accidental comparison of the enum to a value."
            )


@dataclass
class IterController:
    """Iteration controller settings"""

    max_retries: int = 5
    opt_iter: int = 10
    save_iters: SaveIters = SaveIters.USER


@dataclass
class Solver:
    """FE solver settings"""

    dtol: float = 0.001
    etol: float = 0.01
    rtol: float = 0  # TODO: 0 is a magic value that means "disabled"
    lstol: float = 0.9
    ptol: float = 0.01  # Biphasic-specific
    min_residual: float = 1e-20
    update_method: str = "BFGS"  # alt: 'Broyden'
    reform_each_time_step: bool = True
    reform_on_diverge: bool = True
    max_refs: int = 15
    max_ups: int = 10

    def __init__(self, physics="solid", **kwargs):
        # Not sure if there's a better alternative to overriding
        # __init__ to make the defaults depend on an input argument.
        if physics == "biphasic":
            # Only use full Newton iterations
            self.max_ups = 0
            self.reform_each_time_step = True
            # Increase max number of reformations b/c every iteration in
            # the full Newton method is a reformation
            self.max_refs = 50
            # Don't include "symmetric_biphasic"; FEBio 3 doesn't accept
            # it.  And the default in FEBio 2.5 is symmetric_biphasic =
            # False.
        # Allow fields to be set with kwargs
        fields = set(f.name for f in dataclasses.fields(self))
        for k, v in kwargs.items():
            if k in fields:
                setattr(self, k, v)
            else:
                raise TypeError(f"__init__() got an unexpected keyword argument '{k}'")


@dataclass
class Ticker:
    """Settings for time point ticker."""

    n: int = 10
    dtnom: float = 0.1
    dtmin: float = 0.01
    dtmax: Union[float, Sequence] = 0.1


class Physics(Enum):
    """Level of physics to simulate"""

    SOLID = "solid"
    BIPHASIC = "biphasic"


class Step:
    """Simulation step"""

    def __init__(
        self,
        physics: Union[Physics, str],
        ticker: Ticker,
        solver: Solver = None,
        controller: IterController = None,
    ):
        if isinstance(physics, str):
            physics = Physics(physics)
        if solver is None:
            self.solver = Solver(physics)
        else:
            self.solver = solver
        if controller is None:
            self.controller = IterController()
        else:
            self.controller = controller
        assert isinstance(ticker, Ticker)
        self.ticker = ticker
        self.bc: dict = {"node": {}, "body": {}, "contact": []}

    @classmethod
    def from_template(cls, step):
        """Return a new step that copies another step's control settings

        Boundary conditions are not copied

        """
        kwargs = {
            "solver": deepcopy(step.solver),
            "controller": deepcopy(step.controller),
            "ticker": deepcopy(step.tp_controller),
        }
        return cls(**kwargs)

    @property
    def duration(self):
        """Return the duration of a step

        In principle, the duration of a step might not be known in
        advance.  However, FEBio does not permit variable-duration steps
        at present.

        """
        if isinstance(self.ticker.dtmax, Sequence):
            # If there is a must point curve, use it
            duration = self.ticker.dtmax.points[-1][0] - self.ticker.dtmax.points[0][0]
        else:
            # Calculate duration from implicit values
            dt = self.ticker.dtnom
            duration = self.ticker.n * self.ticker.dtnom
        return duration
