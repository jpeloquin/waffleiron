import febtools as feb
import numpy as np
import numpy.testing as npt

    

def f_test():
    elemdata = feb.readlog('test/fixtures/'
                           'isotropic_elastic_elem_data.txt')
    soln = feb.MeshSolution('test/fixtures/'
                            'isotropic_elastic.xplt')
    for istep in xrange(1, len(soln.reader.time)):
        Fxx = elemdata[istep-1]['Fxx'][0]
        Fyy = elemdata[istep-1]['Fyy'][0]
        Fzz = elemdata[istep-1]['Fzz'][0]
        Fxy = elemdata[istep-1]['Fxy'][0]
        Fxz = elemdata[istep-1]['Fxz'][0]
        Fyx = elemdata[istep-1]['Fyx'][0]
        Fyz = elemdata[istep-1]['Fyz'][0]
        Fzx = elemdata[istep-1]['Fzx'][0]
        Fzy = elemdata[istep-1]['Fzy'][0]
        expected = np.array([[Fxx, Fxy, Fxz],
                             [Fyx, Fyy, Fyz],
                             [Fzx, Fzy, Fzz]])
        u = soln.reader.stepdata(istep)['displacement']
        F = soln.element[0].f((0, 0, 0), u)
        yield npt.assert_array_almost_equal, F, expected

def test_integration():
    # create trapezoidal element
    node_list = ((0.0, 0.0),
                 (2.0, 0.0),
                 (1.5, 2.0),
                 (0.5, 2.0))
    nodes = (0, 1, 2, 3)
    element = feb.element.Quad4(nodes, node_list)
    # compute area
    actual = element.integrate(lambda e, r: 1.0)
    desired = 3.0 # A_trapezoid = 0.5 * (b1 + b2) * h
    npt.assert_approx_equal(actual, desired)

def test_dinterp():
    # create square element
    node_list = ((0.0, 0.0),
                 (1.0, 0.0),
                 (1.0, 1.0),
                 (0.0, 1.0))
    nodes = (0, 1, 2, 3)
    element = feb.element.Quad4(nodes, node_list)
    v = (0.0, 10.0, 11.0, 1.0)
    desired = np.array([10.0, 1.0])
    actual = element.dinterp((0,0), v).reshape(-1)
    npt.assert_allclose(actual, desired)
