import febtools as feb
import os
thisdir = os.path.dirname(os.path.abspath(__file__))
fp_xplt = os.path.join(thisdir, 'fixtures', 'complex_loading.xplt')
soln = feb.input.XpltReader(fp_xplt)
fp_feb = os.path.join(thisdir, 'fixtures', 'complex_loading.feb')
febreader = feb.input.FebReader(fp_feb)
model = febreader.model()
model.apply_solution(soln, t=1.0)
model.mesh.elements[0].f((0, 0, 0))
