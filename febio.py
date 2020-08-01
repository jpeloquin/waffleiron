from pathlib import Path
import subprocess


class FEBioError(Exception):
    """Raised when an FEBio simulation terminates in an error."""

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
