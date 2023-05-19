from unittest import TestCase
import numpy as np
import numpy.testing as npt
import waffleiron as wfl

np.seterr(all="raise")
_ZERO_ATOL = np.finfo(float).resolution * 1000


class TestLinspaced(TestCase):
    # Test that errors are produced in reponse to invalid input

    def test_inability_to_span_error(self):
        offset = 0
        n = 1
        span = 1
        with self.assertRaisesRegex(ValueError, "Cannot span distance"):
            wfl.math.linspaced(offset, span, n)
        span = 0
        wfl.math.linspaced(offset, span, n)  # Should /not/ raise

    def test_negative_n(self):
        offset = 0
        n = -1
        span = 1
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            wfl.math.linspaced(offset, span, n)

    # Test that size and span of output is as specified

    def test_range(self):
        # Includes zero-span test
        for offset in (-5, 0, 5):
            for span in (0, 1):
                for n in (2, 11):
                    start = offset
                    end = offset + span
                    x = wfl.math.linspaced(offset, span, n)
                    assert len(x) == n
                    npt.assert_almost_equal(min(x), start)
                    npt.assert_almost_equal(max(x), end)
                    npt.assert_almost_equal(x[-1] - x[0], span)

    def test_negative_span(self):
        # A negative span is ok for linspace because it never has a
        # singularity.
        offset = 3
        n = 11
        span = -17
        x = wfl.math.linspaced(offset, span, n)
        x_pos = wfl.math.linspaced(-offset, -span, n)
        npt.assert_allclose(x, -x_pos)

    def test_one_pt(self):
        n = 1
        span = 0
        for offset in (-7, 0, 7):
            x = wfl.math.linspaced(offset, span, n)
            assert len(x) == 1
            npt.assert_almost_equal(x[0], offset, decimal=7)

    def test_zero_pts(self):
        offset = 0
        n = 0
        for span in (-1, 0, 1):
            assert len(wfl.math.linspaced(offset, span, n)) == 0

    # Test that f(x[i+1]) - f(x[i]) = c invariant is satisfied

    def test_invariant(self):
        offset = 0
        span = 15
        n = 21
        x = wfl.math.linspaced(offset, span, n)

        def f(x):
            return x

        dfdi = np.diff(f(x))
        assert np.max(np.abs(np.diff(dfdi))) < _ZERO_ATOL

    # Test output against known values, where possible

    def test_midpoint(self):
        offset = 0
        n = 21
        span = 1
        mid = wfl.math.linspaced(offset, span, n)[10]
        npt.assert_almost_equal(mid, 0.5, decimal=7)


class TestLogspaced(TestCase):
    # Test that errors are produced in reponse to invalid input

    def test_inability_to_span_error(self):
        offset = 1
        n = 1
        span = 1
        with self.assertRaisesRegex(ValueError, "Cannot span distance"):
            wfl.math.logspaced(offset, span, n)
        span = 0
        wfl.math.logspaced(offset, span, n)  # Should /not/ raise

    def test_negative_offset_error(self):
        offset = -1
        n = 3
        span = 1
        with self.assertRaisesRegex(ValueError, "Offset, \\S+, must be non-negative"):
            wfl.math.logspaced(offset, span, n)

    def test_negative_span_error(self):
        offset = 0
        n = 3
        span = -1
        with self.assertRaisesRegex(ValueError, "Span, \\S+, must be non-negative"):
            wfl.math.logspaced(offset, span, n)

    def test_negative_n_error(self):
        offset = 0
        n = -1
        span = 1
        with self.assertRaisesRegex(
            ValueError, "Number of samples, .+, must be non-negative"
        ):
            wfl.math.logspaced(offset, span, n)

    # Test that size and span of output is as specified

    def test_len(self):
        span = 17
        n = 11
        for offset in (0, 3):
            assert len(wfl.math.logspaced(offset, span, n)) == n

    def test_range(self):
        """Test that output is in correct range"""
        for offset in (0, 1, 5):
            for span in (0, 1, 17):
                for n in (2, 11):
                    start = offset
                    end = offset + span
                    x = wfl.math.logspaced(offset, span, n)
                    assert len(x) == n
                    npt.assert_almost_equal(min(x), start)
                    npt.assert_almost_equal(max(x), end)
                    npt.assert_almost_equal(x[-1] - x[0], span)

    def test_one_pt(self):
        n = 1
        span = 0
        for offset in (0, 1, 7):
            x = wfl.math.logspaced(offset, span, n)
            assert len(x) == 1
            npt.assert_almost_equal(x[0], offset, decimal=7)

    def test_zero_pts(self):
        offset = 0
        n = 0
        for span in (0, 1):
            assert len(wfl.math.logspaced(offset, span, n)) == 0

    # Test that f(x[i+1]) - f(x[i]) = c invariant is satisfied

    def test_invariant(self):
        span = 15
        n = 21

        def f(x):
            return np.log(x)

        for offset in (0, 1, 7):
            x = wfl.math.logspaced(offset, span, n)
            if offset == 0:
                dfdi = np.diff(f(x[1:]))
            else:
                dfdi = np.diff(f(x))
            assert np.max(np.abs(np.diff(dfdi))) < _ZERO_ATOL

    # Test output against known values, where possible

    def test_value(self):
        offset = 0.1
        span = 14.9
        n = 21
        x = wfl.math.logspaced(offset, span, n)
        x_true = np.geomspace(0.1, 15, 21)
        npt.assert_almost_equal(x, x_true, decimal=7)


class TestPowerspaced(TestCase):
    # Test that errors are produced in reponse to invalid input

    def test_inability_to_span_error(self):
        offset = 1
        n = 1
        power = 2
        span = 1
        with self.assertRaisesRegex(ValueError, "Cannot span distance"):
            wfl.math.powerspaced(offset, span, n, power)
        span = 0
        wfl.math.powerspaced(offset, span, n, power)  # Should /not/ raise

    def test_negative_offset_error(self):
        offset = -1
        n = 3
        span = 1
        power = 1
        with self.assertRaisesRegex(ValueError, "Offset, \\S+, must be non-negative"):
            wfl.math.powerspaced(offset, span, n, power)

    def test_negative_span_error(self):
        offset = 0
        n = 3
        span = -1
        power = 1
        with self.assertRaisesRegex(ValueError, "Span, \\S+, must be non-negative"):
            wfl.math.powerspaced(offset, span, n, power)

    def test_negative_n_error(self):
        offset = 0
        n = -1
        span = 1
        power = 1
        with self.assertRaisesRegex(
            ValueError, "Number of samples, .+, must be non-negative"
        ):
            wfl.math.powerspaced(offset, span, n, power)

    # Test that size and span of output is as specified

    def test_range(self):
        """Test that output is in correct range"""
        for offset in (0, 1, 5):
            for span in (0, 1, 17):
                for n in (2, 11):
                    for power in (-2 / 3, 0, 1.5):
                        start = offset
                        end = offset + span
                        x = wfl.math.powerspaced(offset, span, n, power)
                        assert len(x) == n
                        npt.assert_almost_equal(min(x), start)
                        npt.assert_almost_equal(max(x), end)
                        npt.assert_almost_equal(x[-1] - x[0], span)

    def test_one_pt(self):
        n = 1
        span = 0
        for power in (-0.5, 0, 2):
            for offset in (0, 1, 7):
                x = wfl.math.powerspaced(offset, span, n, power)
                assert len(x) == 1
                npt.assert_almost_equal(x[0], offset, decimal=7)

    def test_zero_pts(self):
        offset = 0
        n = 0
        for span in (0, 1):
            for power in (-0.5, 0, 0.5):
                assert len(wfl.math.powerspaced(offset, span, n, power)) == 0

    # Test that f(x[i+1]) - f(x[i]) = c invariant is satisfied

    def test_invariant(self):
        span = 15
        n = 21

        def f(x):
            return x**power

        for power in (-0.5, 0, 1, 1.5):
            for offset in (0, 1, 7):
                x = wfl.math.powerspaced(offset, span, n, power)
                if offset == 0:
                    dfdi = np.diff(f(x[1:]))
                else:
                    dfdi = np.diff(f(x))
                assert np.max(np.abs(np.diff(dfdi))) < _ZERO_ATOL

    # Test output against known values, where possible

    def test_value_sqrt(self):
        offset = 0
        span = 30
        n = 21
        power = 0.5
        x = wfl.math.powerspaced(offset, span, n, power)[10]
        npt.assert_almost_equal(x, 7.5, decimal=7)

    def test_value_linear(self):
        span = 30
        n = 21
        power = 1
        for offset in (0, 1, 7):
            x = wfl.math.powerspaced(offset, span, n, power)
            x_true = wfl.math.linspaced(offset, span, n)
            npt.assert_allclose(x, x_true)

    def test_value_negsqrt(self):
        span = 30
        n = 21
        power = -0.5
        offset = 1
        x = wfl.math.powerspaced(offset, span, n, power)
        x_true = [
            1,
            1.0873783,
            1.18673017,
            1.30034758,
            1.4310983,
            1.58260873,
            1.75951804,
            1.96783756,
            2.21546832,
            2.51296134,
            2.87466049,
            3.32046444,
            3.87862317,
            4.59032614,
            5.51752514,
            6.75688689,
            8.46604867,
            10.91633893,
            14.60749942,
            20.54425558,
            31,
        ]
        npt.assert_allclose(x, x_true)

    def test_value_negsqrt_with_dmin(self):
        span = 30
        n = 21
        power = -0.5
        offset = 0
        dmin = 0.06802721088435373  # (1/n)**(-1/power) * span
        x = wfl.math.powerspaced(offset, span, n, power, dmin=dmin)
        x_true = [
            0,
            0.06802721,
            0.0753963,
            0.08403101,
            0.09423865,
            0.10642584,
            0.12113958,
            0.13912976,
            0.16144661,
            0.18959752,
            0.22580847,
            0.27347794,
            0.33800443,
            0.42838495,
            0.56052999,
            0.7647765,
            1.10498929,
            1.73529883,
            3.11117495,
            7.12031558,
            30,
        ]
        npt.assert_allclose(x, x_true)
