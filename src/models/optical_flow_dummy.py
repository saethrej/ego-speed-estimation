import torch
from torch import nn
import logging as log

class OpticalFlowDummy(nn.Module):
    def __init__(self, config) -> None:
        super(OpticalFlowDummy, self).__init__()

        #store config
        self.config = config
        self.N = self.config.model.batch_size
        self.L = self.config.dataset_config.sample_length

        self.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels = 2,
                out_channels = 12,
                kernel_size = 5,
                stride=1,
                padding=0
            ),
            nn.AvgPool2d(4),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(23856, 64),
            nn.ReLU()
        )
        self.regression = nn.Sequential(
            nn.Linear(64, 1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.N = x.size(0)
        self.L = x.size(1)
        self.H = x.size(3)
        self.W = x.size(4)
        NL = self.N * self.L
        log.debug("Input shape x: {}".format(x.shape))

        # Combine first two dimensions and apply conv layer
        x2 = x.reshape(NL, 2, self.H, self.W)
        log.debug("Input shape to conv - x2: {}".format(x2.shape))
        x3 = self.first_conv(x2)
        log.debug("Output of conv - x3: {}".format(x3.shape))

        # Flatten and apply linear layers
        x4 = x3.reshape(self.N, self.L, -1)
        log.debug("Input to first linear layer - x4: {}".format(x4.shape))
        x5 = self.fc1(x4)
        log.debug("Input to regression layer - x5: {}".format(x5.shape))

        # Apply final regression layer
        x6 = self.regression(x5)
        log.debug("Final output before squeeze - x6: {}".format(x6.shape))
        return x6.squeeze()