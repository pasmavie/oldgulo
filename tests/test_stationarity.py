import numpy as np
import unittest
from gulo.mean_reversion.stationarity import adf, hurst_exp
from gulo.brownian import brownian


class TestStationarity(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.N = 5000
        self.stationary_series = np.random.poisson(0.5, size=self.N)
        self.brownian = brownian(1, n=self.N, dt=10 / self.N, delta=2)

    def test_adf(self):
        adft = adf(self.stationary_series)
        self.assertTrue(adft[0])
        print(f"This stationary series' estimate for the lambda coefficient (first order autoregr) is {adft[1]}<0")
        adft = adf(self.brownian)
        self.assertFalse(adft[0])
        print(f"This random walk's estimate for the lambda coefficient (first order autoregr) is {adft[1]}")

    def test_hurst_exponent(self):
        # ofc I won't log the input series here as they're already what I want them to be
        h = hurst_exp(self.stationary_series)
        print(f"This stationary series has H={h[0]}<0.5 with p-value: {h[1]}")
        self.assertGreater(0.5, h[0])
        h = hurst_exp(self.brownian)
        print(f"This random walk has H={h[0]}~0.5 with p-value: {h[1]}")
        self.assertAlmostEqual(h[0], 0.5, places=1)


if __name__ == "__main__":
    unittest.main()
