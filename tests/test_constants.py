"""
Regression tests for realized measure scaling constants.

Locks down the exact constant values used in BV, MedRV, MinRV so
refactors can't silently break them (Codex Round 2 recommendation).
"""

import math
import numpy as np
import pytest

from volforecast.realized.measures import (
    realized_variance,
    bipower_variation,
    median_rv,
    min_rv,
)


class TestBVConstant:
    """BV scaling: mu_1^{-2} where mu_1 = E[|Z|] = sqrt(2/pi)."""

    def test_mu1_value(self):
        mu1 = np.sqrt(2.0 / np.pi)
        assert np.isclose(mu1, 0.7978845608, rtol=1e-8)

    def test_mu1_inv_sq(self):
        mu1 = np.sqrt(2.0 / np.pi)
        mu1_inv_sq = 1.0 / mu1 ** 2
        assert np.isclose(mu1_inv_sq, np.pi / 2.0, rtol=1e-10)

    def test_bv_equals_rv_for_constant_abs_returns(self):
        """For |r_i| = c for all i, BV = mu1^{-2} * (n/(n-1)) * (n-1) * c^2 = mu1^{-2} * n * c^2."""
        c = 0.01
        n = 100
        r = np.full(n, c)
        bv = bipower_variation(r)
        # sum of |r_i|*|r_{i-1}| = (n-1)*c^2, times n/(n-1) correction, times mu1^{-2}
        expected = (np.pi / 2.0) * n * c ** 2
        assert np.isclose(bv, expected, rtol=1e-8)


class TestMedRVConstant:
    """MedRV scaling constant from Andersen, Dobrev, Schaumburg (2012).

    MedRV = (pi / (6 - 4*sqrt(3) + pi)) * sum med(|r_{i-1}|, |r_i|, |r_{i+1}|)^2
    """

    def test_medrv_constant_value(self):
        c = np.pi / (6.0 - 4.0 * np.sqrt(3.0) + np.pi)
        # Approximate value: pi / (6 - 6.9282 + 3.14159) ~ pi / 2.2134 ~ 1.4189
        assert np.isclose(c, 1.4189, atol=0.001)

    def test_medrv_for_constant_returns(self):
        """For constant |r_i| = c, med(c,c,c)^2 = c^2.
        MedRV = scaling * (n/(n-2)) * sum = scaling * (n/(n-2)) * (n-2) * c^2 = scaling * n * c^2."""
        c = 0.01
        n = 100
        r = np.full(n, c)
        mrv = median_rv(r)
        scaling = np.pi / (6.0 - 4.0 * np.sqrt(3.0) + np.pi)
        expected = scaling * n * c ** 2  # n/(n-2) correction cancels (n-2) terms
        assert np.isclose(mrv, expected, rtol=1e-6)


class TestMinRVConstant:
    """MinRV scaling from Andersen, Dobrev, Schaumburg (2012).

    MinRV = pi/(pi - 2) * sum min(|r_i|, |r_{i+1}|)^2
    """

    def test_minrv_constant_value(self):
        c = np.pi / (np.pi - 2.0)
        # pi / (pi - 2) ~ 3.14159 / 1.14159 ~ 2.7528
        assert np.isclose(c, 2.7528, atol=0.001)

    def test_minrv_for_constant_returns(self):
        """For constant |r_i| = c, min(c,c)^2 = c^2.
        MinRV = scaling * (n/(n-1)) * (n-1) * c^2 = scaling * n * c^2."""
        c = 0.01
        n = 100
        r = np.full(n, c)
        mrv = min_rv(r)
        scaling = np.pi / (np.pi - 2.0)
        expected = scaling * n * c ** 2  # n/(n-1) correction cancels (n-1) terms
        assert np.isclose(mrv, expected, rtol=1e-6)


class TestRVBasic:
    """Basic RV regression tests."""

    def test_rv_for_constant_returns(self):
        """RV of constant r = n * r^2."""
        c = 0.01
        n = 50
        r = np.full(n, c)
        rv = realized_variance(r)
        assert np.isclose(rv, n * c ** 2, rtol=1e-10)
