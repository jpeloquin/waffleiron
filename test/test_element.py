import febtools
import numpy as np
import numpy.testing as npt

def test_integration():
    # create trapezoidal element
    node_list = ((0.0, 0.0),
                 (2.0, 0.0),
                 (1.5, 2.0),
                 (0.5, 2.0))
    nodes = (0, 1, 2, 3)
    element = febtools.element.Quad4(nodes, node_list)
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
    element = febtools.element.Quad4(nodes, node_list)
    v = (0.0, 10.0, 11.0, 1.0)
    desired = np.array([10.0, 1.0])
    actual = element.dinterp((0,0), v).reshape(-1)
    npt.assert_allclose(actual, desired)
