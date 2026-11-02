"""Feature extractors (backbones). Each returns features h of shape [B, D]; the head is separate."""
import torch.nn as nn


class SmallCNN(nn.Module):
    """Two conv layers + one FC layer for MNIST. Output dimension D = feature_dim."""

    def __init__(self, in_channels=1, feature_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),           # 14 -> 7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, feature_dim), nn.ReLU(),
        )
        self.out_dim = feature_dim

    def forward(self, x):
        return self.features(x)


class Classifier(nn.Module):
    """backbone -> head. Head returns log p(y|x), so the loss is always F.nll_loss."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))


def build_backbone(cfg):
    name = cfg["name"]
    if name == "small_cnn":
        return SmallCNN(cfg.get("in_channels", 1), cfg.get("feature_dim", 128))
    raise ValueError(f"Unknown backbone: {name}")
