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


class Dynamics(Enum):
    """Level of dynamics to simulate

    In a steady-state analysis, the time derivatives of the solid displacement and
    solute concentrations are set to zero.

    """

    # Since FEBio 4 shifted to only a static-like and dynamic setting for each
    # module, I'm doing the same.
    STATIC = "static"
    DYNAMIC = "dynamic"


class Physics(Enum):
    """Level of physics to simulate"""

    SOLID = "solid"
    BIPHASIC = "biphasic"
    MULTIPHASIC = "multiphasic"


def auto_ticker(seq: Sequence, n_steps: int = 1, r_dtmin=0.1, must_point_fix=True):
    """Return a ticker with an automatic "must point" curve in dtmax

    :param n_steps: Number of steps in the ticker.  The number of time points is
    n_steps + 1.  The number of steps may be automatically increased to work around
    an FEBio must point bug.

    :param r_dtmin: Set the minimum allowed time step to `r_dtmin` * the minimum time
    step in `seq`.

    """
    ticker = Ticker()
    duration = seq.points[-1][0] - seq.points[0][0]
    time = np.array([a for a, b in seq.points])
    dt = np.diff(time)
    dt = np.concatenate([dt, dt[-1:]])  # len(dt) == len(time)
    # Assign must point sequence
    curve_must_dt = densify([(a, b) for a, b in zip(time, dt)], n=n_steps)
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
    if must_point_fix and (dt_min == dt_nominal):
        # Sometimes FEBio skips a must point even when the first time point is
        # coincident with the step_size.  Add an extra step to avoid this.
        nsteps += 1
        dt_nominal = duration / nsteps
    ticker.n = nsteps
    ticker.dtnom = dt_nominal
    ticker.dtmin = r_dtmin * dt_min
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
    save_iters: SaveIters = dataclasses.field(default_factory=lambda: SaveIters.USER)


@dataclass
class Solver:
    """FE solver settings"""

    dtol: float = 0.001
    etol: float = 0.01
    rtol: float = 0  # TODO: 0 is a magic value that means "disabled"
    lstol: float = 0.9
    ptol: float = 0.01  # Biphasic-specific
    min_residual: float = 1e-20
    update_method: str = "BFGS"  # alt: 'Broyden' or 'Newton'
    reform_each_time_step: bool = True
    reform_on_diverge: bool = True
    max_refs: int = 15
    max_ups: int = 10

    def __init__(self, physics=Physics.SOLID, **kwargs):
        # Not sure if there's a better alternative to overriding
        # __init__ to make the defaults depend on an input argument.
        if physics == Physics.BIPHASIC:
            # Only use full Newton iterations
            self.update_method = "Newton"
            self.max_ups = 0
            self.reform_each_time_step = True
            # Increase max number of reformations, since every iteration in the full
            # Newton method is a reformation
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
        self.validate()

    def validate(self):
        if self.update_method == "Newton" and self.max_ups != 0:
            raise ValueError(
                "Chose both update_method = 'Newton' and max_ups != 0.  Due to limitations in FEBio XML syntax, you must choose max_ups = 0 if you want a Newton solver."
            )


@dataclass
class Ticker:
    """Settings for time point ticker."""

    n: int = 10
    dtnom: float = 0.1
    dtmin: float = 0.01
    dtmax: Union[float, Sequence] = 0.1


class Step:
    """Simulation step"""

    # TODO: Get rid of `physics` argument; it should be a property of the model. Here
    #  it's only used to automatically set up the solver.  Pass the model instead.
    def __init__(
        self,
        physics: Union[Physics, str],
        dynamics: Union[Dynamics, str],
        ticker: Ticker,
        solver: Solver = None,
        controller: IterController = None,
    ):
        if isinstance(physics, str):
            physics = Physics(physics)
        if isinstance(dynamics, str):
            dynamics = Dynamics(dynamics)
        self.dynamics = dynamics
        if solver is None:
            solver = Solver(physics)
        self.solver = solver
        if controller is None:
            controller = IterController()
        self.controller = controller
        if not isinstance(ticker, Ticker):
            raise TypeError("`ticker` must be a Ticker")
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
