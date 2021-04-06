import errno
import os
from pathlib import Path
import subprocess
from typing import Sequence

from lxml import etree
import psutil

from .input import load_model
from .febioxml import logfile_name


# Default name of FEBio executable
FEBIO_CMD = os.environ.get("FEBIO_CMD", "febio")

# Number of threads to use for each FEBio call
FEBIO_THREADS = psutil.cpu_count(logical=False)


class FEBioError(Exception):
    """Raise when an FEBio simulation terminates in an error.

    Even when FEBio doesn't realize it terminated in an error.

    """

    pass


class NoSolutionError(FEBioError):
    """Raise when a solution is expected but none is found"""

    pass


class IncompleteSolutionError(FEBioError):
    """Raise when FEBio produces an incomplete solution"""


class CheckError(Exception):
    """Raise when simulation output fails a check

    Subclass CheckError to define specifically which check failed.

    """

    pass


class IncorrectTimePoints(CheckError):
    """Raise when FEBio emits extra, missing, or offset time points"""

    pass


def run_febio_unchecked(pth_feb, threads=None, cmd=FEBIO_CMD):
    """Run FEBio and return the process object.

    This function runs FEBio and captures its output, in particular
    its self-reported error status.  Some workarounds are applied to
    consistently and comprehensively report FEBio's error status,
    but there are no additional error checks or safeguards beyond
    what FEBio provides.  `run_febio_checked`, in contrast,
    runs FEBio with added safeguards.

    """
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
    try:
        proc = subprocess.run(
            [cmd, "-i", pth_feb.name],
            # FEBio always writes xplt to current dir
            cwd=pth_feb.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )
    except OSError as e:
        if e.errno == errno.ENOENT:
            raise ValueError(
                f"The OS could not find an executable file named {cmd}; ensure that an FEBio executable with that name exists on the system and is in a directory included in the system PATH variable.  Alternatively, set the environment variable FEBIO_CMD to the command used to run FEBio on your system."
            )
    # If there specifically is a file read error, we need to write the
    # captured stdout to the log file, because only stdout has
    # information about the file read error.  Otherwise, we need to
    # leave the log file in place, because it contains unique
    # information.  (FEBio does return a error code of 1 on "Error
    # Termination" and 0 on "Normal Termination"; I checked.)
    #
    # With FEBio 2, a file read error prints "Reading file foo.feb
    # ...FAILED!" to stdout as a single line.  With FEBio 3, there may
    # be warnings and blank lines in between "..." and "FAILED!".
    if proc.returncode != 0:
        reading = False
        for ln in proc.stdout.splitlines():
            if ln.startswith("Reading file "):
                reading = True
            if reading and ln.startswith("*"):
                # Skip warning boxes after "Reading file foo.feb ..."
                continue
            if ln.endswith("SUCCESS!"):
                # No file read error
                break
            elif ln.endswith("FAILED!"):
                # File read error; send it to the log file
                with open(pth_log, "w", encoding="utf-8") as f:
                    f.write(proc.stdout)
                break
        else:
            raise NotImplementedError(
                f"febtools failed to parse FEBio file read status message '{ln}' from stdout"
            )
    return proc


def run_febio_checked(pth_feb, threads=None, cmd=FEBIO_CMD):
    """Run FEBio, raising exception on error.

    In addition, perform the following checks independent of FEBio's own
    (mostly absent) error checking:

    - Verify that the solution file (xplt file) actually exists and can be read by
    febtools.

    - In any step with PLOT_MUST_POINTS, verify that the number of time
      points matches the number of must points.

    """
    if threads is None:
        threads = FEBIO_THREADS
    pth_feb = Path(pth_feb)
    dir_feb = pth_feb.parent
    with open(pth_feb, "rb") as f:
        tree = etree.parse(f)
    root = tree.getroot()
    # Delete any existing log file and xplt file because we will
    # later check for the existence of an xplt to determine if the
    # simulation even started.   Also it is potentially confusing to
    # have old output still visible even after re-running the
    # simulation.
    pth_xplt = pth_feb.with_suffix(".xplt")
    try:
        pth_xplt.unlink()
    except FileNotFoundError:
        pass
    pth_log = logfile_name(root)
    try:
        pth_log.unlink()
    except FileNotFoundError:
        pass
    proc = run_febio_unchecked(pth_feb, threads=threads, cmd=cmd)
    if proc.returncode != 0:
        msg = f"FEBio returned error code {proc.returncode} while running {pth_feb}; check {pth_feb.with_suffix('.log')}."
        if not pth_xplt.exists():
            raise NoSolutionError(msg)
        else:
            raise IncompleteSolutionError(msg)
    # Perform additional checks
    model = load_model(pth_feb)
    # Default checks
    check_solution_exists(model)
    check_must_points(model)
    return proc


def check_solution_exists(model):
    """Check if a solution exists"""
    if model.solution is None:
        raise NoSolutionError(
            f"{model.name} has no accompanying plotfile, or febtools could not recognize it as a plotfile."
        )


def check_must_points(model):
    # Check if all must points were included
    must_point_sim = True  # default assumption
    times = [0.0]
    # ^ According to Steve, t = 0 s is mandatory if must points are
    # used.  https://forums.febio.org/showthread.php?49-must-points
    for step, name in model.steps:
        if step.controller.save_iters != "PLOT_MUST_POINTS":
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
            raise IncorrectTimePoints(
                f'FEBio wrote {n_actual} time points when simulating {model.name}, but {n_expected} time points ("must points") were requested.  This may be caused by a bug in FEBio\'s time stepper or must point controller, or by requesting invalid time points that were silently ignored by FEBio.'
            )


def read_log(pth):
    """Attempt to read an FEBio log file

    This function only has partial support for reading the FEBio log
    file.  Even if the implementation were "complete", the FEBio log
    format is undocumented and there is no guarantee it will remain the
    same between FEBio versions.  Expect some things in the log file to
    be missed.

    read_log is meant to read the following items from the log file:

    - Errors boxed in asterisks, starting with a centered ERROR line.

    """
    # It's not clear which encoding FEBio uses for it's log files.
    errors = []
    with open(pth, "r") as f:
        for ln in f:
            if ln.startswith(" *") and "ERROR" in ln:
                # Found a boxed error.
                msg = []
                ln = f.readline()  # skip blank line after ERROR
                while not ln.startswith(" **********"):
                    ln = f.readline()
                    s = ln.strip("* \n\r\t")
                    msg.append(s)
                # skip end blank line + end asterisk border
                msg = msg[:-2]
                errors.append("\n".join(msg))
    return errors
