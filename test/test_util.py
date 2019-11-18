# Run these tests with nose
from math import inf
from unittest import TestCase
import numpy as np
from febtools.util import find_closest_timestep

import warnings
warnings.simplefilter("error")

class FindClosestTimestep(TestCase):

    def setUp(self):
        self.times = [0.0, 0.5, 1.0, 1.5, 2.0]
        self.steps = [0, 1, 2, 3, 4]

    def test_in_middle(self):
        assert(find_closest_timestep(1.0, self.times, self.steps) == 2)

    def test_in_middle_atol_ok(self):
        assert(find_closest_timestep(1.2, self.times, self.steps,
                                     atol=0.2, rtol=inf) == 2)

    def test_in_middle_rtol_ok(self):
        assert(find_closest_timestep(0.75, self.times, self.steps,
                                     atol=inf, rtol=0.51) == 1)

    def test_in_middle_atol_bad(self):
        with self.assertRaisesRegex(ValueError, "absolute error > atol"):
            assert(find_closest_timestep(0.52, self.times, self.steps) == 1)

    def test_in_middle_rtol_bad(self):
        with self.assertRaisesRegex(ValueError, "relative error > rtol"):
            assert(find_closest_timestep(0.52, self.times, self.steps,
                                         atol=inf) == 1)

    def test_before_start_ok(self):
        assert(find_closest_timestep(-0.005, self.times, self.steps,
                                     atol=0.005) == 0)

    def test_before_start_bad(self):
        with self.assertRaisesRegex(ValueError, "absolute error > atol"):
            assert(find_closest_timestep(-0.005, self.times, self.steps) == 0)

    def test_at_start(self):
        assert(find_closest_timestep(0.0, self.times, self.steps) == 0)

    def test_at_end(self):
        assert(find_closest_timestep(2.0, self.times, self.steps) == 4)

    def test_past_end_ok(self):
        assert(find_closest_timestep(2.5, self.times, self.steps,
                                     atol=0.51, rtol=1.01) == 4)

    def test_past_end_bad(self):
        with self.assertRaisesRegex(ValueError, "absolute error > atol"):
            assert(find_closest_timestep(2.5, self.times, self.steps) == 4)

    def test_nonmatching_values(self):
        times = [0.0, 0.5, 1.0, 1.5, 2.0]
        steps = [0, 1, 2, 3]
        with self.assertRaisesRegex(ValueError, "len\(steps\) ≠ len\(times\)"):
            find_closest_timestep(0.5, times, steps)
