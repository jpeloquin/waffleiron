"""Create a uniaxial test case that rotates 180°"""
from collections import defaultdict
from math import pi, cos, sin
import numpy as np

import waffleiron as wfl

if __name__ == "__main__":
    model = wfl.load_model("uniax_stretch_cube.feb")

    # Create node displacements that satisfy the desired stretch + rotation boundary
    # conditions.
    x_ref = model.mesh.nodes.T
    N_STEP = 20
    # ^ number of time steps in displacement sequence (t = 0 doesn't count).  Note that
    # this is independent of the number of time steps in the /solution/, which is
    # controlled by the must point sequence in the template file.
    times = np.linspace(0, 1, N_STEP + 1)
    angles = np.linspace(0, pi, N_STEP + 1)
    stretches = np.linspace(1.0, 1.5, N_STEP + 1)
    displacements = []
    for λ, θ in zip(stretches, angles):
        F = np.eye(3)
        F[0,0] = λ
        R = np.array([[cos(θ), sin(θ), 0],
                      [-sin(θ), cos(θ), 0],
                      [0, 0, 1]])
        displacements.append((R @ F @ x_ref - x_ref).T)
    displacements = np.array(displacements)

    # Create Sequence for each node's path in x1, x2, or x3
    sequences = defaultdict(dict)  # keys: node index, BC axis name
    for inode in range(len(model.mesh.nodes)):
        for ix, dof in zip(range(3), ["x1", "x2", "x3"]):
            pts = [(t, u)
                   for t, u in zip(times, displacements[:, inode, ix])]
            seq = wfl.Sequence(pts, interp="spline", extrap="constant")
            # TODO: The extra .step is awkward; make NameStep fancier.
            model.apply_nodal_bc([inode], dof, "displacement", seq, model.steps[0].step)

    # Write model to disk
    with open("uniax_stretch_cube_with_180°_flip.feb", "wb") as f:
        wfl.output.write_feb(model, f)
