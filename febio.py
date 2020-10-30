import os
from pathlib import Path
import subprocess

import psutil

from .input import load_model


# Number of threads to use for each FEBio call
FEBIO_THREADS = psutil.cpu_count(logical=False)


class FEBioError(Exception):
    """Raised when an FEBio simulation terminates in an error.

    Even when FEBio doesn't realize it terminated in an error.

    """

    pass


def _run_febio(pth_feb, threads=None):
    """Run FEBio and return the process object."""
    # FEBio's error handling is interesting, in a bad way.  XML file
    # read errors are only output to stdout.  If there is a read error,
    # no log file is created and if an old log file exists, it is not
    # updated to reflect the file read error.  Model summary info is
    # only output to the log file.  Time stepper info is output to both
    # stdout and the log file, but the verbosity of the stdout output
    # can be adjusted by the user.  We want to ensure (1) the log file
    # always reflects the last run and (2) all relevant error messages
    # are written to the log file.
    if threads is None:
        threads = FEBIO_THREADS
    pth_feb = Path(pth_feb)
    pth_log = pth_feb.with_suffix(".log")
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": f"{threads}"})
    # Check for the existance of the FEBio XML file ourselves, since if
    # the file doesn't exist FEBio will act as if it was malformed.
    if not pth_feb.exists():
        raise ValueError(f"'{pth_feb}' does not exist or is not accessible.")
    # TODO: Check for febio executable
    proc = subprocess.run(
        ["febio", "-i", pth_feb.name],
        # FEBio always writes xplt to current dir
        cwd=pth_feb.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    # If there specifically is a file read error, we need to write the
    # captured stdout to the log file, because only it has information
    # about the file read error.  Otherwise, we need to leave the log
    # file in place, because it contains unique information.  (FEBio
    # does return a error code of 1 on "Error Termination" and 0 on
    # "Normal Termination"; I checked.)
    if proc.returncode != 0:
        for ln in proc.stdout.splitlines():
            if ln.startswith("Reading file"):
                if ln.endswith("SUCCESS!"):
                    # No file read error
                    break
                elif ln.endswith("FAILED!"):
                    # File read error; send it to the log file
                    with open(pth_log, "w", encoding="utf-8") as f:
                        f.write(proc.stdout)
                else:
                    raise NotImplementedError(
                        f"febtools failed to parse FEBio file read status message '{ln}' from stdout"
                    )
    return proc


def run_febio_checked(pth_feb, threads=None):
    """Run FEBio, raising exception on error.

    In addition, perform the following checks independent of FEBio's own
    (mostly absent) error checking:

    - In any step with PLOT_MUST_POINTS, verify that the number of time
      points matches the number of must points.

    """
    if threads is None:
        threads = FEBIO_THREADS
    pth_feb = Path(pth_feb)
    proc = _run_febio(pth_feb, threads=threads)
    if proc.returncode != 0:
        raise FEBioError(
            f"FEBio returned error code {proc.returncode} while running {pth_feb}; check {pth_feb.with_suffix('.log')}."
        )
    # Perform additional checks
    model = load_model(pth_feb)
    check_must_points(model)
    return proc.returncode


def run_febio_unchecked(pth_feb, threads=None):
    """Run FEBio and return its error code."""
    if threads is None:
        threads = FEBIO_THREADS
    return _run_febio(pth_feb, threads=threads).returncode


def check_must_points(model):
    if model.solution is None:
        raise ValueError(
            f"{model.name} has no accompanying plotfile; the solution's time points cannot be checked if the solution does not exist."
        )
    # Check if all must points were included
    must_point_sim = True  # default assumption
    times = [0.0]
    # ^ According to Steve, t = 0 s is mandatory if must points are
    # used.  https://forums.febio.org/showthread.php?49-must-points
    for step in model.steps:
        record_rule = step["control"].setdefault("plot level", None)
        if record_rule != "PLOT_MUST_POINTS":
            must_point_sim = False
            break
        dtmax = step["control"]["time stepper"]["dtmax"]
        cur_times = [a for a, b in dtmax.points]
        # If the first time point in the current step is the same as the
        # last time point of the previous step, FEBio does not write it
        # the XPLT.  So we do not count it here either.
        if cur_times[0] == times[-1]:
            cur_times = cur_times[1:]
        times += cur_times
    else:
        # Run if all simulation steps use must points
        n_expected = len(times)
        n_actual = len(model.solution.step_times)
        if n_expected != n_actual:
            raise FEBioError(
                f'FEBio wrote {n_actual} time points when simulating {model.name}, but {n_expected} time points ("must points") were requested.  This may be caused by a bug in FEBio\'s time stepper or must point controller, or by requesting invalid time points that were silently ignored by FEBio.'
            )
