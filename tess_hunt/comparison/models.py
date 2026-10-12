"""Current CNN, richer observational CNN, and the legacy shape architecture."""
import numpy as np
import torch
from torch import nn

from tess_hunt.learning.model import SignalNet, VIEW_KEYS


class ContextEncoder(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv1d(channels, width, 7, stride=2, padding=3), nn.GELU(),
                                    nn.Conv1d(width, 2*width, 5, stride=2, padding=2), nn.GELU(),
                                    nn.Conv1d(2*width, 2*width, 5, stride=2, padding=2), nn.GELU())

    def forward(self, x):
        x = self.layers(x)
        return torch.cat([x.mean(-1), x.amax(-1)], -1)


class RichSignalNet(SignalNet):
    def __init__(self, width=32):
        super().__init__(width)
        self.context_local_encoder = ContextEncoder(15, width)
        self.context_global_encoder = ContextEncoder(6, width)
        self.head = nn.Sequential(nn.Linear(24*width+4, 4*width), nn.GELU(),
                                  nn.Dropout(.2), nn.Linear(4*width, 1))

    def forward(self, batch):
        g = self.global_encoder(batch["global"])
        local = self.local_encoder(batch["local"])
        events = batch["events"]
        b, n, channels, length = events.shape
        encoded = self.event_encoder(events.reshape(b*n, channels, length)).reshape(b, n, -1)
        mask = batch["event_mask"].unsqueeze(-1)
        count = mask.sum(1).clamp(min=1)
        mean = (encoded*mask).sum(1)/count
        spread = torch.sqrt(((encoded-mean.unsqueeze(1)).square()*mask).sum(1)/count+1e-6)
        context = [self.context_local_encoder(batch["context_local"]), self.context_global_encoder(batch["context_global"])]
        return self.head(torch.cat([g, local, mean, spread, *context, batch["scalar"]], -1)).squeeze(-1)


def conv_branch(channels, blocks):
    layers = []
    for out in blocks:
        layers += [nn.Conv1d(channels, out, 5, padding=2), nn.ReLU(), nn.Conv1d(out, out, 5, padding=2),
                   nn.ReLU(), nn.MaxPool1d(3, 2)]
        channels = out
    return nn.Sequential(*layers, nn.AdaptiveMaxPool1d(4), nn.Flatten())


class LegacyShapeNet(nn.Module):
    """At width 16, exactly the architecture of 08_train_shape_model.ShapeNet."""
    def __init__(self, width=16, features=True):
        super().__init__()
        self.glob = conv_branch(1, [width, 2*width, 4*width])
        self.loc = conv_branch(5, [width, 2*width])
        self.head = nn.Sequential(nn.Linear(24*width+(24 if features else 0), 8*width), nn.ReLU(),
                                  nn.Dropout(.3), nn.Linear(8*width, 1))

    def forward(self, batch):
        return self.head(torch.cat([self.glob(batch["global"]), self.loc(batch["local"]), batch["features"]], 1)).squeeze(1)


def feature_stats(features, rows):
    train = features[rows]
    finite = np.isfinite(train)
    count = finite.sum(0)
    mean = np.where(finite, train, 0).sum(0)/np.maximum(count, 1)
    variance = np.where(finite, (train-mean)**2, 0).sum(0)/np.maximum(count, 1)
    return mean.astype(np.float32), np.maximum(np.sqrt(variance), 1e-9).astype(np.float32)


def normalize_features(features, mean, std):
    missing = ~np.isfinite(features)
    z = np.where(missing, 0, np.clip((features-mean)/std, -10, 10))
    return np.concatenate([z, missing.astype(np.float32)], 1).astype(np.float32)


def model_inputs(arrays, family, rows, normalization=None):
    """Normalization always fitted on explicitly supplied training indices."""
    legacy = family.startswith("legacy")
    if legacy:
        if normalization is None:
            mean, std = feature_stats(arrays["legacy_features"], rows)
            normalization = {"mean": mean.tolist(), "std": std.tolist()}
        feature = (normalize_features(arrays["legacy_features"][rows], np.asarray(normalization["mean"]),
                                      np.asarray(normalization["std"])) if family == "legacy_hybrid"
                   else np.zeros((len(rows), 0), np.float32))
        return {"global": arrays["legacy_global"][rows], "local": arrays["legacy_local"][rows], "features": feature}, normalization
    prefix = "masked_" if "masked" in family else ""
    if normalization is None:
        scalar = arrays[prefix+"scalar"][rows]
        normalization = {"mean": scalar.mean(0).tolist(), "std": np.maximum(scalar.std(0), .1).tolist()}
    inputs = {k: arrays[prefix+k][rows] for k in VIEW_KEYS}
    inputs["scalar"] = ((inputs["scalar"]-np.asarray(normalization["mean"], np.float32))/np.asarray(normalization["std"], np.float32))
    if family.startswith("rich"):
        inputs.update({k: arrays[prefix+k][rows] for k in ("context_global", "context_local")})
    return inputs, normalization


def make_model(family, width):
    if family.startswith("legacy"):
        return LegacyShapeNet(width, family == "legacy_hybrid")
    return RichSignalNet(width) if family.startswith("rich") else SignalNet(width)
