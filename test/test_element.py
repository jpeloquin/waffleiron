import febtools
import numpy.testing as npt

def test_integration():
    # create trapezoidal element
    node_list = ((0, 0),
                 (2, 0),
                 (1.5, 2),
                 (0.5, 2))
    nodes = (0, 1, 2, 3)
    element = febtools.element.Quad4(nodes, node_list)
    # compute area
    actual = element.integrate(lambda r: 1.0)
    desired = 3.0 # A_trapezoid = 0.5 * (b1 + b2) * h
    npt.assert_approx_equal(actual, desired)
