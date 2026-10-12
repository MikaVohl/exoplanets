"""Checks for label safety, star leakage, gaps, thresholds and blind recovery."""
import unittest

import numpy as np
import pandas as pd
import torch

from tess_hunt.learning.dataset import ROOT, load_config
from tess_hunt.learning.labels import match_signal, partition
from tess_hunt.learning.photometry import clean, make_views, bin_search
from tess_hunt.learning.model import SignalNet, choose_threshold, metrics
from tess_hunt.learning.search import find_signals
from tess_hunt.learning.evaluate import inject_transit, injection_matches

CONFIG = load_config(ROOT / "config/learning.json")


def catalog(disposition="CP", period=3, epoch=1501):
    return pd.DataFrame([{"toi": "123.01", "tfopwg_disp": disposition, "pl_orbper": period,
                          "pl_tranmid": epoch + 2457000, "pl_trandurh": 2.4}])


class LabelTests(unittest.TestCase):
    def setUp(self):
        self.signal = {"period": 3., "t0": 1501., "duration_days": .1}
        self.time = np.arange(1500, 1527, .01)

    def test_false_positive_host_does_not_label_another_signal_negative(self):
        unknown = {**self.signal, "period": 4.7}
        self.assertEqual(match_signal(unknown, catalog("FP"), self.time, CONFIG)["y"], -1)
        self.assertEqual(match_signal(self.signal, catalog("FP"), self.time, CONFIG)["y"], 0)

    def test_period_alone_and_aliases_never_create_a_training_label(self):
        wrong_phase = {**self.signal, "t0": 1502.}
        alias = {**self.signal, "period": 1.5}
        self.assertEqual(match_signal(wrong_phase, catalog(), self.time, CONFIG)["y"], -1)
        self.assertEqual(match_signal(alias, catalog(), self.time, CONFIG)["y"], -1)
        self.assertEqual(match_signal(self.signal, catalog(), self.time, CONFIG)["y"], 1)

    def test_uncertain_and_conflicting_labels_stay_unlabeled(self):
        self.assertEqual(match_signal(self.signal, catalog("PC"), self.time, CONFIG)["y"], -1)
        conflicting = pd.concat([catalog("CP"), catalog("FP")])
        self.assertEqual(match_signal(self.signal, conflicting, self.time, CONFIG)["y"], -1)
        missing = catalog()
        missing["pl_tranmid"] = np.nan
        self.assertEqual(match_signal(self.signal, missing, self.time, CONFIG)["y"], -1)

    def test_ephemeris_drift_over_baseline_is_checked(self):
        drift = {**self.signal, "period": 3.009}
        self.assertEqual(match_signal(drift, catalog(), self.time, CONFIG)["y"], -1)

    def test_partition_is_stable_when_sample_grows(self):
        first = {tic: partition(tic, CONFIG) for tic in range(100)}
        grown = {tic: partition(tic, CONFIG) for tic in range(1000)}
        self.assertEqual(first, {tic: grown[tic] for tic in first})
        self.assertEqual(set(grown.values()), {"train", "validation", "test"})


class RepresentationAndThresholdTests(unittest.TestCase):
    def test_views_keep_missing_bins_and_padding_explicit(self):
        t = np.r_[np.arange(1500, 1502, .01), np.arange(1510, 1512, .01)]
        data = {"time": t, "flux": np.ones(len(t)), "error": np.full(len(t), .001)}
        signal = {"period": 5., "t0": 1500.5, "duration_days": .1}
        views = make_views(data, signal, CONFIG)
        missing = views["global"][2] == 0
        self.assertTrue(missing.any())
        self.assertTrue(np.all(views["global"][:2, missing] == 0))
        self.assertEqual(int(views["event_mask"].sum()), 2)
        binned = bin_search(data, 10)
        self.assertFalse(np.any((binned["time"] > 1502) & (binned["time"] < 1510)))

    def test_padding_does_not_change_network_prediction(self):
        torch.manual_seed(4)
        torch.set_num_threads(2)
        net = SignalNet(width=4).eval()
        batch = {"global": torch.randn(1, 3, 256), "local": torch.randn(1, 3, 128),
                 "events": torch.randn(1, 2, 3, 64), "event_mask": torch.ones(1, 2), "scalar": torch.zeros(1, 4)}
        padded = {**batch, "events": torch.cat([batch["events"], torch.randn(1, 3, 3, 64)], dim=1),
                  "event_mask": torch.tensor([[1., 1., 0., 0., 0.]])}
        with torch.no_grad():
            self.assertTrue(torch.allclose(net(batch), net(padded), atol=1e-6))

    def test_validation_threshold_respects_false_positive_budget_with_ties(self):
        y = np.array([0, 0, 0, 0, 1, 1])
        scores = np.array([.8, .8, .3, .1, .9, .6])
        threshold = choose_threshold(y, scores, .25)
        self.assertLessEqual(metrics(y, scores, threshold)["false_positive_rate"], .25)
        self.assertEqual(metrics(y, scores, threshold)["recall"], .5)
        threshold = choose_threshold(y, scores, 0)
        self.assertEqual(metrics(y, scores, threshold)["false_positive_rate"], 0)


class InjectionTests(unittest.TestCase):
    def test_injection_is_before_cleaning_and_does_not_change_source(self):
        rng = np.random.default_rng(8)
        t = np.arange(1500, 1527, 2 / 1440)
        t = t[~((t > 1512) & (t < 1513))]
        flux = 1000 * (1 + rng.normal(0, .0001, len(t)))
        raw = {"time": t, "flux": flux, "error": np.full(len(t), .1),
               "quality": np.zeros(len(t), dtype=int), "tic": 1, "sector": 1}
        original = flux.copy()
        modified, signal = inject_transit(raw, rng, CONFIG)
        self.assertTrue(np.array_equal(raw["flux"], original))
        self.assertLess(modified["flux"].min(), original.min())
        data = clean(modified, CONFIG)
        self.assertFalse(np.any((data["time"] > 1512) & (data["time"] < 1513)))
        # BLS is deliberately blind: neither injection period nor epoch is passed.
        candidates = find_signals(data, {**CONFIG, "search_method": "bls", "maximum_signals": 2})
        self.assertTrue(any(injection_matches(c, signal, data["time"], CONFIG) for c in candidates))
        self.assertTrue(all(np.isfinite(a).all() for a in make_views(data, candidates[0], CONFIG).values()))


if __name__ == "__main__":
    unittest.main()
