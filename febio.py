from pathlib import Path
import subprocess


class FEBioError(Exception):
    """Raised when an FEBio simulation terminates in an error.

    Even when FEBio doesn't realize it terminated in an error.

    """

    pass


def run_febio(pth_feb):
    if isinstance(pth_feb, str):
        pth_feb = Path(pth_feb)
    proc = subprocess.run(
        ["febio", "-i", pth_feb.name],
        # FEBio always writes xplt to current dir
        cwd=Path(pth_feb).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        # FEBio truly does return an error code on "Error Termination";
        # I checked.
        pth_log = Path(pth_feb).with_suffix(".log")
        with open(pth_log, "wb") as f:
            f.write(proc.stdout)  # FEBio doesn't always write a log if it
            # hits a error, but the content that would
            # have been logged is always dumped to
            # stdout.
        raise FEBioError(
            f"FEBio returned an error (return code = {proc.returncode}) while running {pth_feb}; check {pth_log}."
        )
    return proc.returncode


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
