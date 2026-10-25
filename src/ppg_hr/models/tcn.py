"""Compact offline residual temporal convolutions, independently implemented."""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, kernel, dilation, dropout):
        super().__init__()
        padding = (kernel // 2) * dilation
        self.layers = nn.Sequential(
            nn.Conv1d(incoming, outgoing, kernel, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(outgoing, outgoing, kernel, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()

    def forward(self, x):
        return torch.relu(self.layers(x) + self.skip(x))


class Encoder(nn.Module):
    def __init__(self, incoming, channels, kernel, dilations, dropout):
        super().__init__()
        if len(channels) != len(dilations) or kernel % 2 != 1:
            raise ValueError("one dilation per channel block and odd kernel required")
        layers = []
        for outgoing, dilation in zip(channels, dilations, strict=True):
            layers.append(ResidualBlock(incoming, outgoing, kernel, dilation, dropout))
            incoming = outgoing
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).mean(dim=-1)


class PPGTCN(nn.Module):
    def __init__(self, channels=(16, 32, 64), kernel=5, dilations=(1, 2, 4), dropout=0.1):
        super().__init__()
        self.ppg = Encoder(1, channels, kernel, dilations, dropout)
        self.head = nn.Linear(channels[-1], 1)

    def forward(self, ppg):
        return self.head(self.ppg(ppg)).squeeze(-1)


class FusionTCN(nn.Module):
    def __init__(self, channels=(16, 32, 64), kernel=5, dilations=(1, 2, 4), dropout=0.1):
        super().__init__()
        self.ppg = Encoder(1, channels, kernel, dilations, dropout)
        self.acc = Encoder(3, channels, kernel, dilations, dropout)
        self.head = nn.Sequential(nn.Linear(2 * channels[-1], 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, ppg, acc):
        return self.head(torch.cat((self.ppg(ppg), self.acc(acc)), dim=1)).squeeze(-1)


class BPMModel(nn.Module):
    """Target transformation is embedded; inputs must use stored normalization."""

    def __init__(self, kind, config, stats):
        super().__init__()
        if kind not in ("ppg_tcn", "fusion_tcn"):
            raise ValueError("unknown model kind")
        cls = PPGTCN if kind == "ppg_tcn" else FusionTCN
        self.model = cls(
            config["channels"], config["kernel_size"], config["dilations"], config["dropout"]
        )
        self.fusion = kind == "fusion_tcn"
        self.register_buffer("target_mean", torch.tensor(stats["target_mean"], dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(stats["target_std"], dtype=torch.float32))

    def forward(self, ppg, acc=None):
        output = self.model(ppg, acc) if self.fusion else self.model(ppg)
        return output * self.target_std + self.target_mean


class ONNXSingle(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ppg):
        return self.model(ppg)


class ONNXFusion(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ppg, acc):
        return self.model(ppg, acc)
