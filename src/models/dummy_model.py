import torch
from torch import nn

import logging as log

class DummyModel(nn.Module):
    def __init__(self, config) -> None:
        super(DummyModel, self).__init__()

        self.config = config

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels = 3,
                out_channels = 16,
                kernel_size=5,
                stride=2,
                padding=0
            ),
            nn.MaxPool2d(3),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.onlyLayer = nn.Sequential(
            nn.Linear(16*19*47, 1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.N = x.size(0)
        self.L = x.size(1)
        NL = self.N * self.L
        x = x.view(NL, 3, 118, 290)
        x = self.conv(x)
        log.info(x.shape)
        x = x.reshape(self.N, self.L, -1)
        x = self.onlyLayer(x)
        return x.squeeze()