"""Scientific controls for input ablations and faithful legacy architecture."""
import importlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tess_hunt.comparison.data import detrend, rich_views
from tess_hunt.comparison.models import LegacyShapeNet, model_inputs, feature_stats
from tess_hunt.learning.photometry import clean
from tess_hunt.learning.model import BASELINE_FEATURES
from tess_hunt.comparison.experiment import fit_cell, spec


class ComparisonControlsTests(unittest.TestCase):
    def setUp(self):
        self.config = {"minimum_points": 1000, "split_gap_days": .5, "flatten_window_days": 1.4,
                       "global_bins": 256, "local_bins": 128, "local_half_width_durations": 2.5}
        self.signal = {"period": 2., "t0": 1., "duration_days": .15}

    def test_masked_cleaning_preserves_transit_and_unmasked_matches_current(self):
        t = np.arange(0, 8, 2/1440)
        phase = (t-1+1) % 2-1
        inside = abs(phase) < .075
        raw = {"time": t, "flux": 1+.003*np.sin(t/3)-.01*inside,
               "error": np.full(len(t), .0001), "quality": np.zeros(len(t), int)}
        masked = detrend(raw, self.signal, self.config)
        unmasked = detrend(raw, self.signal, self.config, mask_transits=False)
        def depth(data):
            p = (data["time"]-1+1) % 2-1
            return np.median(data["flux"][abs(p) > .2])-np.median(data["flux"][abs(p) < .05])
        self.assertAlmostEqual(depth(masked), .01, delta=.0003)
        current = clean(raw, self.config)
        for key in current:
            np.testing.assert_allclose(unmasked[key], current[key], rtol=0, atol=1e-12)

    def test_rich_views_preserve_alternating_depths_and_missing_centroids(self):
        t = np.arange(0, 8, 2/1440)
        phase = (t-1+1) % 2-1
        parity = np.rint((t-1)/2).astype(int) % 2
        y = np.ones(len(t))-np.where(parity == 1, .01, .005)*(abs(phase) < .075)
        data = {"time": t, "flux": y, "error": np.full(len(t), .0001)}
        missing = {"time": np.array([]), "flux": np.array([]), "error": np.array([])}
        views = rich_views(data, data, data, [missing, missing], self.signal, self.config)
        self.assertEqual(views["context_local"].shape, (15, 128))
        # Values are clipped at 100 noise units; choose a central bin to compare depths.
        self.assertAlmostEqual(views["context_local"][0, 64]/views["context_local"][3, 64], 2., places=4)
        self.assertTrue((views["context_local"][9:] == 0).all())
        self.assertTrue(np.isfinite(views["context_global"]).all())

    def test_feature_normalization_ignores_validation_and_handles_all_missing(self):
        values = np.array([[1., np.nan], [3., np.nan], [100., 8.]], np.float32)
        mean, std = feature_stats(values, np.array([0, 1]))
        np.testing.assert_array_equal(mean, [2, 0])
        arrays = {"legacy_features": values, "legacy_global": np.zeros((3, 1, 201), np.float32),
                  "legacy_local": np.zeros((3, 5, 61), np.float32)}
        _, normalization = model_inputs(arrays, "legacy_hybrid", np.array([0, 1]))
        val, _ = model_inputs(arrays, "legacy_hybrid", np.array([2]), normalization)
        self.assertEqual(val["features"][0, 0], 10.)
        self.assertTrue(np.isfinite(val["features"]).all())
        self.assertEqual(normalization["mean"], [2., 0.])

    def test_legacy_architecture_matches_original_weights_and_predictions(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tls_masked_search"))
        original = importlib.import_module("08_train_shape_model")
        torch.set_num_threads(1)
        for features in (False, True):
            before = original.ShapeNet(24 if features else 0).eval()
            after = LegacyShapeNet(16, features).eval()
            after.load_state_dict(before.state_dict(), strict=True)
            batch = {"global": torch.randn(3, 1, 201), "local": torch.randn(3, 5, 61),
                     "features": torch.randn(3, 24 if features else 0)}
            with torch.no_grad():
                a = before(batch["global"], batch["local"], batch["features"])
                b = after(batch)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(sum(p.numel() for p in after.parameters()), 101921 if features else 98849)

    def test_tree_training_scores_only_supplied_validation_rows(self):
        labels = np.tile([0, 1], 30)
        df = pd.DataFrame({"y": labels, "period": labels+1., "tic": np.arange(60),
                           "candidate_id": [f"candidate-{i}" for i in range(60)]})
        for feature in BASELINE_FEATURES:
            if feature not in df:
                df[feature] = 0.
        train, validation = np.arange(40), np.arange(40, 60)
        result, history, scores, checkpoint = fit_cell(df, {}, train, validation,
                                                       spec("baseline", 0, 40, 0),
                                                       {"torch_threads": 1, "target_validation_fpr": .1}, "cpu")
        self.assertEqual(len(scores), 20)
        self.assertGreater(result["validation_roc_auc"], .95)
        self.assertEqual(set(result["train_candidate_ids"]), set(df.candidate_id.iloc[train]))
        self.assertFalse(set(result["train_candidate_ids"]) & set(df.candidate_id.iloc[validation]))
        self.assertIsNone(checkpoint)


if __name__ == "__main__":
    unittest.main()
