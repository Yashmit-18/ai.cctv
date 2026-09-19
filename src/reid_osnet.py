"""OSNet (Omni-Scale Network) -- forward-only person ReID feature extractor.

Vendored from the official torchreid implementation
(https://github.com/KaiyangZhou/deep-person-reid) so that a genuine trained
person-ReID model can run in this project without a torchreid dependency.
Only the CNN backbone + embedding head needed for *inference* is included
(no gdown, no identity classifier, no training paths).

Attribution / license
---------------------
* Omni-Scale Feature Learning for Person Re-Identification.  Zhou, Yang,
  Cavallaro.  ICCV 2019.
* Learning Generalisable Omni-Scale Representations for Person
  Re-Identification.  Zhou, Yang, Cavallaro.  TPAMI 2021.
* torchreid is distributed under the **MIT License** (Kaiyang Zhou).  This
  file is an adapted, trimmed copy of ``torchreid/models/osnet.py`` and
  inherits the MIT license.

Pretrained weights (MSMT17-trained, MIT) are downloaded from the official
HuggingFace mirror ``kaiyangzhou/osnet`` and stored under ``models/`` -- see
``MODELS.md`` and the Phase 49 report.  This module never downloads anything
on its own: the caller supplies a local ``.pth`` file.

Semantics
---------
``OsnetEmbedder.extract(crop_bgr)`` returns the raw 512-dim feature vector
(the output of the embedding head, *before* L2 normalisation).  The caller
(``AppearanceExtractor``) normalises exactly as it does for the built-in
descriptor, so cosine similarity and the Phase 44 identity-fusion rules are
unchanged.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

try:  # torch is an optional convenience; the built-in descriptor never needs it
    import torch
    from torch import nn
    from torch.nn import functional as _F
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - env dependent
    _TORCH_AVAILABLE = False

#: Fixed input size OSNet was trained on (height, width) -- torchreid default.
OSNET_INPUT_H = 256
OSNET_INPUT_W = 128

#: ImageNet statistics, applied exactly as torchreid does (RGB, 0-1 range).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Basic layers
# ---------------------------------------------------------------------------


class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, IN=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False,
                              groups=groups)
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride,
                              padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 convolution + optional bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1, bn=True):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride,
                              padding=0, bias=False)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution: 1x1 (linear) + dw 3x3 (nonlinear)."""

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1,
                               padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1,
                               padding=1, bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConvStream(nn.Module):
    """Lightweight convolution stream (chain of ``depth`` LightConv3x3)."""

    def __init__(self, in_channels, out_channels, depth):
        super(LightConvStream, self).__init__()
        assert depth >= 1
        layers = [LightConv3x3(in_channels, out_channels)]
        for _i in range(depth - 1):
            layers += [LightConv3x3(out_channels, out_channels)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ChannelGate(nn.Module):
    """Mini-network generating channel-wise gates from the input tensor."""

    def __init__(self, in_channels, num_gates=None, return_gates=False,
                 gate_activation="sigmoid", reduction=16, layer_norm=False):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction,
                             kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, num_gates,
                             kernel_size=1, bias=True, padding=0)
        if gate_activation == "sigmoid":
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == "relu":
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == "linear":
            self.gate_activation = None
        else:
            raise RuntimeError(
                "Unknown gate activation: {}".format(gate_activation))

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(self, in_channels, out_channels, IN=False,
                 bottleneck_reduction=4, **kwargs):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return _F.relu(out)


class OSBlockAIN(nn.Module):
    """Omni-scale block used by OSNet-AIN (per-stream gate + shared gate)."""

    def __init__(self, in_channels, out_channels, reduction=4, T=4, **kwargs):
        super(OSBlockAIN, self).__init__()
        assert T >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        mid_channels = out_channels // reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, T + 1):
            self.conv2.append(LightConvStream(mid_channels, mid_channels, t))
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2 = x2 + self.gate(conv2_t(x1))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return _F.relu(out)


class OSBlockINAIN(nn.Module):
    """OSNet-AIN block with instance normalisation inside the residual."""

    def __init__(self, in_channels, out_channels, reduction=4, T=4, **kwargs):
        super(OSBlockINAIN, self).__init__()
        assert T >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        mid_channels = out_channels // reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, T + 1):
            self.conv2.append(LightConvStream(mid_channels, mid_channels, t))
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels, bn=False)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2 = x2 + self.gate(conv2_t(x1))
        x3 = self.conv3(x2)
        x3 = self.IN(x3)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return _F.relu(out)


# ---------------------------------------------------------------------------
# Network (OSNet-AIN then vanilla OSNet)
# ---------------------------------------------------------------------------


class OSNet(nn.Module):
    """Omni-scale person-ReID backbone -- feature/embedded representation only.

    The identity ``classifier`` head (used only for metric-learning training)
    is intentionally omitted; the released checkpoints simply leave those
    weights unmatched when loading.
    """

    def __init__(self, blocks, layers, channels, feature_dim=512, IN=False,
                 conv1_IN=False, **kwargs):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.feature_dim = feature_dim

        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3,
                               IN=(IN or conv1_IN))
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0],
                                      channels[1], reduce_spatial_size=True)
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1],
                                      channels[2], reduce_spatial_size=True)
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2],
                                      channels[3], reduce_spatial_size=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[3], feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self._init_params()


class OSNetAIN(nn.Module):
    """OSNet-AIN backbone (attention + instance-norm; TPAMI-2021 variant).

    Mirrors ``torchreid/models/osnet_ain.py`` (feature head only): conv1 uses
    instance normalisation, conv3 blocks interpolate OSBlockINin/OSBlock with
    IN inside the residual, and each stage pool is a separate 1x1+avgpool.
    """

    def __init__(self, blocks, layers, channels, feature_dim=512, **kwargs):
        super(OSNetAIN, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.feature_dim = feature_dim

        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0],
                                      channels[1])
        self.pool2 = nn.Sequential(Conv1x1(channels[1], channels[1]),
                                   nn.AvgPool2d(2, stride=2))
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1],
                                      channels[2])
        self.pool3 = nn.Sequential(Conv1x1(channels[2], channels[2]),
                                   nn.AvgPool2d(2, stride=2))
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2],
                                      channels[3])
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[3], feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self._init_params()

    def _make_layer(self, block_list, layer, in_channels, out_channels):
        blocks = [block_list[0](in_channels, out_channels)]
        for _i in range(1, len(block_list)):
            blocks.append(block_list[_i](out_channels, out_channels))
        return nn.Sequential(*blocks)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x):
        return self._embed(x)

    def _embed(self, x):
        x = self.featuremaps(x)
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        v = self.fc(v)
        return v


class OSNet(nn.Module):
    """Omni-scale person-ReID backbone -- feature/embedded representation only.

    The identity ``classifier`` head (used only for metric-learning training)
    is intentionally omitted; the released checkpoints simply leave those
    weights unmatched when loading.
    """

    def __init__(self, blocks, layers, channels, feature_dim=512, IN=False,
                 conv1_IN=False, **kwargs):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.feature_dim = feature_dim

        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3,
                               IN=(IN or conv1_IN))
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0],
                                      channels[1], reduce_spatial_size=True)
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1],
                                      channels[2], reduce_spatial_size=True)
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2],
                                      channels[3], reduce_spatial_size=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[3], feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self._init_params()

    def _make_layer(self, block, layer, in_channels, out_channels,
                    reduce_spatial_size):
        layers = [block(in_channels, out_channels)]
        for _i in range(1, layer):
            layers.append(block(out_channels, out_channels))
        if reduce_spatial_size:
            layers.append(
                nn.Sequential(Conv1x1(out_channels, out_channels),
                              nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x):
        x = self.featuremaps(x)
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        v = self.fc(v)
        return v


# ---------------------------------------------------------------------------
# Instantiation + weight loading
# ---------------------------------------------------------------------------

#: name -> (blocks, layers, channels) for the released OSNet family.  AIN
#: variants use a list-of-lists block layout (one list per stage).
_OSNET_VARIANTS = {
    "osnet_x1_0": ([OSBlock, OSBlock, OSBlock], [2, 2, 2],
                   [64, 256, 384, 512]),
    "osnet_x0_5": ([OSBlock, OSBlock, OSBlock], [2, 2, 2],
                   [32, 128, 192, 256]),
    "osnet_x0_25": ([OSBlock, OSBlock, OSBlock], [2, 2, 2],
                    [16, 64, 96, 128]),
    "osnet_ain_x1_0": ([[OSBlockINAIN, OSBlockINAIN],
                        [OSBlockAIN, OSBlockINAIN],
                        [OSBlockINAIN, OSBlockAIN]],
                       [2, 2, 2], [64, 256, 384, 512]),
    "osnet_ain_x0_5": ([[OSBlockINAIN, OSBlockINAIN],
                        [OSBlockAIN, OSBlockINAIN],
                        [OSBlockINAIN, OSBlockAIN]],
                       [2, 2, 2], [32, 128, 192, 256]),
    "osnet_ain_x0_25": ([[OSBlockINAIN, OSBlockINAIN],
                         [OSBlockAIN, OSBlockINAIN],
                         [OSBlockINAIN, OSBlockAIN]],
                        [2, 2, 2], [16, 64, 96, 128]),
}


def _detect_variant(model_path: str) -> str:
    """Infer the OSNet variant from the filename (``osnet[`_ain`]_x…``)."""
    name = os.path.basename(str(model_path)).lower().replace(".pth", "")
    ain = "_ain_" in name or name.startswith("osnet_ain")
    for scale in ("x1_0", "x0_75", "x0_5", "x0_25"):
        if scale in name:
            base = f"osnet_{scale}"
            return f"{base.replace('osnet_', 'osnet_ain_', 1) if ain else base}"
    return "osnet_ain_x1_0" if ain else "osnet_x1_0"


def build_osnet(variant: str = "osnet_x1_0"):
    """Build the feature-only OSNet for a supported channel scaling."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is not available to run an OSNet ReID model")
    if variant not in _OSNET_VARIANTS:
        raise ValueError(f"unsupported OSNet variant: {variant!r}")
    blocks, layers, channels = _OSNET_VARIANTS[variant]
    if variant.startswith("osnet_ain"):
        return OSNetAIN(blocks=blocks, layers=layers, channels=channels,
                        feature_dim=512)
    return OSNet(blocks=blocks, layers=layers, channels=channels,
                 feature_dim=512)


def load_osnet_weights(model: OSNet, model_path: str) -> None:
    """Load a torchreid checkpoint into ``model`` (feature keys only).

    ``classifier.*`` and any ``module.`` prefix are handled; an assert fails
    loudly if any feature key present in the checkpoint cannot be matched,
    because silent partial loads would silently corrupt the embeddings.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is not available to run an OSNet ReID model")
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    keep = {k: v for k, v in state.items() if k in dict(model.state_dict())}
    missing = [k for k in model.state_dict() if k not in keep]
    if not keep:
        raise RuntimeError(f"no feature weights matched in {model_path}")
    if missing:
        raise RuntimeError("OSNet checkpoint missing feature keys: "
                           + ", ".join(sorted(missing)[:8]))
    model.load_state_dict(keep, strict=False)


def preprocess_osnet(crop_bgr: np.ndarray, height: int = OSNET_INPUT_H,
                     width: int = OSNET_INPUT_W) -> np.ndarray:
    """BGR person crop -> normalized float32 NCHW input, exactly like torchreid.

    torchreid resizes crops directly to (256, 128) with bicubic interpolation
    (the same distribution the model saw at training time) and applies the
    ImageNet mean/std over RGB in the 0-1 range.
    """
    import cv2 as _cv2
    h, w = crop_bgr.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("empty person crop")
    resized = _cv2.resize(crop_bgr, (int(width), int(height)),
                          interpolation=_cv2.INTER_CUBIC)
    rgb = _cv2.cvtColor(resized, _cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb[..., 0] = (rgb[..., 0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    rgb[..., 1] = (rgb[..., 1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    rgb[..., 2] = (rgb[..., 2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
    chw = np.transpose(rgb, (2, 0, 1))
    return chw[None, ...].astype(np.float32)


class OsnetEmbedder:
    """Loads one OSNet checkpoint and produces 512-dim embeddings on CPU.

    ``extract`` returns the **raw** (non-L2) feature vector; normalisation is
    left to the caller so the whole pipeline uses one normalisation path.
    """

    def __init__(self, model_path: str, variant: str | None = None):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is not available to run an OSNet model")
        self.model_path = str(model_path)
        self.variant = variant or _detect_variant(self.model_path)
        self.model = build_osnet(self.variant)
        load_osnet_weights(self.model, self.model_path)
        self.model.eval()
        for _p in self.model.parameters():
            _p.requires_grad_(False)
        self.feature_dim = int(self.model.feature_dim)
        # One warmup pass so the first live crop does not pay lazy-init cost.
        with torch.no_grad():
            self.model(torch.zeros(1, 3, OSNET_INPUT_H, OSNET_INPUT_W))

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Return the raw 512-dim embedding, or None on any failure."""
        try:
            x = preprocess_osnet(crop_bgr)
            with torch.no_grad():
                out = self.model(torch.from_numpy(x)).numpy()
            emb = np.asarray(out, dtype=np.float32).reshape(-1)
            if emb.size == 0 or not np.all(np.isfinite(emb)):
                return None
            return emb
        except Exception:
            return None