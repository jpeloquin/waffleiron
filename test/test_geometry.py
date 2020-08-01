# Run these tests with nose
import unittest, os
import febtools as feb
from febtools import Mesh


class RectangularPrismHex8(unittest.TestCase):
    """Test point in element for narrow parallelograms.

    This test addresses the bug in which the elements which might
    contain point P were selected by finding the node N closest to P
    and checking if P was in any element connected to N.  However, N
    is not necessarily part of the element that contains P.

    """

    def setUp(self):
        nodes = [
            (0, 0, 0),
            (1, 0, 0),
            (1, 10, 0),
            (0, 10, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 10, 1),
            (0, 10, 1),
        ]
        # a rectangle (on left)
        self.e = feb.element.Hex8.from_ids([0, 1, 2, 3, 4, 5, 6, 7], nodes)

    def test_points_in_element(self):
        points = [(0, 0, 0), (0.5, 0.5, 0), (0.9, 5, 1), (0.5, 0.5, 0.5), (0.9, 5, 0.5)]
        for p in points:
            assert feb.geometry.point_in_element(self.e, p)

    def test_points_not_in_element(self):
        points = [(-0.1, 0, 0), (-0.5, 0.4, 0.5), (10, 10, 10)]
        for p in points:
            assert not feb.geometry.point_in_element(self.e, p)
