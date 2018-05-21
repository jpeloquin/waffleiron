"""Module for controling FEBio.

This module includes functionality for working with FEBio-specific
control sections in FEBio xml files.

"""
# Base modules
import math
# Third party modules
import numpy as np
# Same-package modules
from .conditions import densify, Sequence


def auto_control_section(sequence, pts_per_segment=6):
    curve = sequence.points
    # Assign good default control settings
    control = default_control_section()
    duration = curve[-1][0] - curve[0][0]
    time = np.array([a for a, b in curve])
    dt = np.diff(time)
    dt = np.concatenate([dt, dt[-1:]])  # len(dt) == len(time)
    dt_min = np.min(dt)
    # dt_max = np.max(dt)
    nsteps = math.ceil(duration / dt_min)
    dt_nominal = duration / nsteps
    control['time steps'] = nsteps
    control['step size'] = dt_nominal
    control['time stepper']['dtmin'] = 0.1 * dt_min
    control['plot level'] = 'PLOT_MUST_POINTS'
    # Assign must point sequence
    curve_must_dt = densify([(a, b) for a, b in zip(time, dt)],
                            n=pts_per_segment)
    seq_dtmax = Sequence(curve_must_dt, extend='constant', typ='linear')
    control['time stepper']['dtmax'] = seq_dtmax

    return control


def default_control_section():
    default = {'time steps': 10,
               'step size': 0.1,
               'max refs': 15,
               'max ups': 10,
               'dtol': 0.001,
               'etol': 0.01,
               'rtol': 0,
               'lstol': 0.9,
               'time stepper': {'dtmin': 0.01,
                                'dtmax': 0.1,
                                'max retries': 5,
                                'opt iter': 10},
               'analysis type': 'static',
               'plot level': 'PLOT_MAJOR_ITRS'}
    return default


def step_duration(step):
    """Calculate the duration of a step."""
    dtmax_entry = step['control']['time stepper']['dtmax']
    if isinstance(dtmax_entry, Sequence):
        # If there is a must point curve, use it
        duration = dtmax_entry.points[-1][0] - dtmax_entry.points[0][0]
    else:
        # Calculate duration from implicit values
        n = step['control']['time steps']
        dt = step['control']['step size']
        duration = n * dt
    return duration
