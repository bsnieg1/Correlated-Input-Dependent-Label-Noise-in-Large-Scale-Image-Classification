"""Feature extractors (backbones). Each returns features h of shape [B, D]; the head is separate."""
import torch
import torch.nn as nn
import torchvision


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


class PretrainedResNet(nn.Module):
    """torchvision ResNet with ImageNet weights, classification layer removed. Input: normalized images."""

    def __init__(self, name="resnet50", pretrained=True, hub_dir=None):
        super().__init__()
        if hub_dir:
            torch.hub.set_dir(hub_dir)
        weights = {"resnet18": "IMAGENET1K_V1", "resnet50": "IMAGENET1K_V2"}[name] if pretrained else None
        net = getattr(torchvision.models, name)(weights=weights)
        self.out_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.net = net

    def forward(self, x):
        return self.net(x)


class ResNet18Small(nn.Module):
    """ResNet-18 from scratch for small images (e.g. 64x64): first conv 3x3 stride 1, no max-pooling (D = 512)."""

    def __init__(self, in_channels=3):
        super().__init__()
        net = torchvision.models.resnet18(weights=None)
        net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        self.out_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.net = net

    def forward(self, x):
        return self.net(x)


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
    if name == "resnet18_small":
        return ResNet18Small(cfg.get("in_channels", 3))
    if name in ("resnet18", "resnet50"):
        return PretrainedResNet(name, cfg.get("pretrained", True), cfg.get("hub_dir"))
    raise ValueError(f"Unknown backbone: {name}")
