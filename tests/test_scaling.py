"""Scientific controls for nested data scaling and training-only normalization."""
import unittest

import numpy as np
import pandas as pd
import torch

from tess_hunt.scaling.experiment import nested_subsets, tensor_views
from tess_hunt.learning.model import SignalNet


class ScalingControlsTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for tic in range(40):
            split = "train" if tic < 30 else "validation" if tic < 35 else "test"
            for signal in range(1 + tic % 3):
                rows.append({"tic": tic, "split": split, "y": tic % 2, "signal": signal})
            rows.append({"tic": tic, "split": split, "y": -1, "signal": 99})
        self.df = pd.DataFrame(rows)

    def test_nested_subsets_keep_whole_stars_and_exclude_unlabeled_and_held_out(self):
        subsets = nested_subsets(self.df, [12, 24, 40], 4)
        previous = set()
        for target, rows in subsets.items():
            chosen = self.df.iloc[rows]
            self.assertTrue(previous.issubset(set(rows)))
            self.assertTrue(chosen.split.eq("train").all())
            self.assertTrue(chosen.y.ge(0).all())
            self.assertEqual(set(chosen.y), {0, 1})
            for tic in chosen.tic.unique():
                complete = set(self.df[self.df.tic.eq(tic) & self.df.y.ge(0)].index)
                self.assertTrue(complete.issubset(set(rows)))
            self.assertGreaterEqual(len(rows), target)
            self.assertLess(len(rows), target + 3)
            previous = set(rows)

    def test_seed_reproducibility_and_full_data_control(self):
        a = nested_subsets(self.df, [12, 24], 0)
        b = nested_subsets(self.df, [12, 24], 0)
        c = nested_subsets(self.df, [12, 24], 1)
        for target in a:
            self.assertTrue(np.array_equal(a[target], b[target]))
        self.assertNotEqual(set(a[12]), set(c[12]))
        self.assertEqual(set(a[max(a)]), set(c[max(c)]))

    def test_tensor_normalization_uses_supplied_training_statistics(self):
        arrays = {"global": np.zeros((3, 3, 256), dtype=np.float32),
                  "local": np.zeros((3, 3, 128), dtype=np.float32),
                  "events": np.zeros((3, 16, 3, 64), dtype=np.float32),
                  "event_mask": np.ones((3, 16), dtype=np.float32),
                  "scalar": np.array([[0, 0, 0, 0], [2, 2, 2, 2], [100, 100, 100, 100]], dtype=np.float32)}
        views = tensor_views(arrays, np.array([2]), np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32), "cpu")
        self.assertTrue(torch.all(views["scalar"] == 99))
        self.assertEqual(sum(p.numel() for p in SignalNet(16).parameters()), 41057)


if __name__ == "__main__":
    unittest.main()
