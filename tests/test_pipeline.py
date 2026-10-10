"""Scientific checks using injected transits, null data, gaps and eclipses."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from tess_hunt.photometry import preprocess
from tess_hunt.search import bls_search, tls_search, conclude, vet_signal, period_relation

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "benchmark.json").read_text())


def synthetic(seed=7, transit=True):
    rng = np.random.default_rng(seed)
    t = np.arange(0, 27, 10 / 1440) + 1500
    dy = np.full(len(t), .0005)
    y = 1 + rng.normal(0, .0005, len(t))
    if transit:
        inside = abs((t - 1501.1 + 3.7 / 2) % 3.7 - 3.7 / 2) < .12 / 2
        y[inside] -= .008
    return {"time": t, "flux": y, "error": dy,
            "centroid1": np.full(len(t), np.nan), "centroid2": np.full(len(t), np.nan)}


class ScientificPipelineTests(unittest.TestCase):
    def test_blind_methods_recover_injected_period(self):
        data = synthetic()
        bls, _ = bls_search(data, {"mass": 1, "radius": 1}, CONFIG)
        tls, _ = tls_search(data, {"mass": 1, "radius": 1}, CONFIG)
        self.assertLess(abs(bls["best"]["period"] / 3.7 - 1), .005)
        self.assertLess(abs(tls["best"]["period"] / 3.7 - 1), .005)
        self.assertEqual(conclude(bls, tls, CONFIG)["conclusion"], "supported_transit_like_signal")
        self.assertTrue(.004 < tls["best"]["model_depth"] < .012)

    def test_seeded_white_noise_is_not_supported(self):
        data = synthetic(seed=81, transit=False)
        bls, _ = bls_search(data, {"mass": 1, "radius": 1}, CONFIG)
        tls, _ = tls_search(data, {"mass": 1, "radius": 1}, CONFIG)
        self.assertEqual(conclude(bls, tls, CONFIG)["conclusion"], "no_significant_signal_in_search_range")

    def test_detrending_retains_dips_and_does_not_fill_gaps(self):
        rng = np.random.default_rng(11)
        t = np.arange(0, 27, 2 / 1440) + 1500
        t = t[~((t > 1511) & (t < 1512))]
        phase = (t - 1501.1 + 3.7 / 2) % 3.7 - 3.7 / 2
        flux = 1000 * (1 + .003 * np.sin((t - 1500) * 2 * np.pi / 5) + rng.normal(0, .0003, len(t)))
        flux[abs(phase) < .06] -= 8
        columns = [fits.Column(name="TIME", format="D", array=t),
                   fits.Column(name="PDCSAP_FLUX", format="D", array=flux),
                   fits.Column(name="PDCSAP_FLUX_ERR", format="D", array=np.full(len(t), .3)),
                   fits.Column(name="QUALITY", format="J", array=np.zeros(len(t), dtype=int)),
                   fits.Column(name="MOM_CENTR1", format="D", array=np.ones(len(t))),
                   fits.Column(name="MOM_CENTR2", format="D", array=np.ones(len(t)))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "injected.fits"
            fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path)
            data, metadata = preprocess(path, CONFIG)
        self.assertFalse(np.any((data["time"] > 1511) & (data["time"] < 1512)))
        folded = (data["time"] - 1501.1 + 3.7 / 2) % 3.7 - 3.7 / 2
        recovered_depth = np.median(data["flux"][abs(folded) > .2]) - np.median(data["flux"][abs(folded) < .04])
        self.assertTrue(.005 < recovered_depth < .010)
        self.assertLess(metadata["search_points"], metadata["original_points"])

    def test_unobserved_transits_are_not_counted(self):
        t = np.r_[np.arange(1500, 1501, .00694444), np.arange(1513, 1514, .00694444)]
        y = np.ones(len(t))
        y[abs((t - 1500.3 + 2.5) % 5 - 2.5) < .06] -= .01
        data = {"time": t, "flux": y, "error": np.full(len(t), .0005)}
        result = vet_signal(data, 5, 1500.3, .12)
        self.assertEqual(result["observed_transits"], 1)
        self.assertIn("fewer_than_two_observed_transits", result["flags"])

    def test_alternating_eclipses_get_odd_even_warning(self):
        data = synthetic(transit=False)
        t = data["time"]
        inside = abs((t - 1501.1 + 3.7 / 2) % 3.7 - 3.7 / 2) < .06
        cycles = np.rint((t - 1501.1) / 3.7).astype(int)
        data["flux"][inside] -= np.where(cycles[inside] % 2 == 0, .02, .01)
        result = vet_signal(data, 3.7, 1501.1, .12)
        self.assertIn("odd_even_depth_mismatch", result["flags"])

    def test_period_harmonics_are_distinct_from_exact_matches(self):
        self.assertEqual(period_relation(3, 6), "half_period")
        self.assertEqual(period_relation(12, 6), "double_period")
        self.assertEqual(period_relation(6.01, 6), "exact")
        self.assertIsNone(period_relation(7, 6))


if __name__ == "__main__":
    unittest.main()
