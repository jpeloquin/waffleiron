"""Functions to generate models commonly used in experiments"""
import numpy as np

import waffleiron as wfl
from . import _DEFAULT_TOL
from waffleiron.control import Step, auto_ticker
from .core import Sequence, NodeSet
from .math import linspaced, logspaced


def freeswell_cube(h, material, init_duration, swell_duration, init_n=10, swell_n=5):
    feb_module = "biphasic"
    A = np.array([-h / 2, -h / 2])
    B = np.array([h / 2, -h / 2])
    C = np.array([h / 2, h / 2])
    D = np.array([-h / 2, h / 2])
    AB = [A + s * (B - A) for s in linspaced(0, 1, n=3)]
    DC = [D + s * (C - D) for s in linspaced(0, 1, n=3)]
    AD = [A + s * (D - A) for s in linspaced(0, 1, n=3)]
    BC = [B + s * (C - B) for s in linspaced(0, 1, n=3)]
    base = wfl.mesh.quadrilateral(AD, BC, AB, DC)
    cube = wfl.mesh.zstack(base, wfl.math.linspaced(0, 0.1, n=3) - h / 2)
    model = wfl.Model(cube)
    for element in model.mesh.elements:
        element.material = material
    # Adjust constants for mm, g, s, C, nmol, MPa, mM system
    model.constants["R"] = 8.31446261815324e-6  # mJ/nmol·K
    model.constants["F"] = 96485.3329e-9  # C/nmol
    model.mesh.nodes = np.array(model.mesh.nodes)
    nodes = model.mesh.nodes
    midx = NodeSet(np.where(np.abs(nodes[:, 0]) < _DEFAULT_TOL)[0])
    midy = NodeSet(np.where(np.abs(nodes[:, 1]) < _DEFAULT_TOL)[0])
    midz = NodeSet(np.where(np.abs(nodes[:, 2]) < _DEFAULT_TOL)[0])
    assert len(midx) > 0
    assert len(midy) > 0
    assert len(midz) > 0
    origin = model.mesh.find_nearest_nodes(0, 0, 0)[0]
    left = NodeSet(
        np.where(np.abs(nodes[:, 0] - np.min(nodes[:, 0])) < _DEFAULT_TOL)[0]
    )
    model.named["node sets"].add("−x_surface", left)
    right = NodeSet(
        np.where(np.abs(nodes[:, 0] - np.max(nodes[:, 0])) < _DEFAULT_TOL)[0]
    )
    model.named["node sets"].add("+x_surface", right)
    front = NodeSet(
        np.where(np.abs(nodes[:, 1] - np.min(nodes[:, 1])) < _DEFAULT_TOL)[0]
    )
    model.named["node sets"].add("−y_surface", front)
    back = NodeSet(
        np.where(np.abs(nodes[:, 1] - np.max(nodes[:, 1])) < _DEFAULT_TOL)[0]
    )
    model.named["node sets"].add("+y_surface", back)
    bottom = NodeSet(
        np.where(np.abs(nodes[:, 2] - np.min(nodes[:, 2])) < _DEFAULT_TOL)[0]
    )
    model.named["node sets"].add("−z_surface", bottom)
    top = NodeSet(np.where(np.abs(nodes[:, 2] - np.max(nodes[:, 2])) < _DEFAULT_TOL)[0])
    model.named["node sets"].add("+z_surface", top)
    surface_nodes = left | right | front | back | bottom | top
    model.fixed["node"][("x1", "displacement")].update(midx)
    model.fixed["node"][("x2", "displacement")].update(midy)
    model.fixed["node"][("x3", "displacement")].update(midz)
    model.fixed["node"][("fluid", "pressure")].update(surface_nodes)
    # Initialize properties that FEBio can't initialize itself
    time = linspaced(0, init_duration, init_n)
    seq = Sequence([(t, 0) for t in time], interp="linear", extrap="constant")
    step = Step(feb_module, "static", auto_ticker(seq))
    model.add_step(step, name="Property initialization")
    # Free swell to equilibrium
    time = logspaced(0, swell_duration, swell_n, dmin=swell_duration / (swell_n - 1))
    seq = Sequence([(t, 0) for t in time], interp="linear", extrap="constant")
    step = Step(feb_module, "static", auto_ticker(seq))
    model.add_step(step, name="Free swelling")
    return model
