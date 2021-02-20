import os
import febtools as feb
from febtools.test.fixtures import DIR_OUT

mat = feb.material.NeoHookean({"E": 1.1, "v": 0.3})
mesh = feb.mesh.rectangular_prism((1, 2), (1, 2), (1, 2), material=mat)
model = feb.Model(mesh)
pth = (DIR_OUT / os.path.basename(__file__)).with_suffix(".feb")
with open(pth, "wb") as f:
    feb.output.write_feb(model, f, version="2.0")
pth.unlink()
