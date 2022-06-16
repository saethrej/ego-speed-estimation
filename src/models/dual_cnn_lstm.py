'''

Implementation of the Dual-CNN-LSTM model

'''

from turtle import forward
import torch
from torch import nn

import logging as log

class DualCnnLstm(nn.Module):
    def __init__(self, config) -> None:
        super(DualCnnLstm, self).__init__()

        # store config file
        self.config = config
        self.N = self.config.model.batch_size
        self.L = self.config.dataset_config.sample_length
        self.W = 0
        self.H = 0

        # set number of channels according to preprocessing steps
        self.channels_in = 3
        if config.preprocessing.video_transform == 'depth_opticalflow':
            self.channels_in = 4
        elif config.preprocessing.video_transform == 'opticalflow_gray':
            self.channels_in = 3
        elif config.preprocessing.video_transform == 'opticalflow':
            self.channels_in = 2

        # First convolution
        self.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels = self.channels_in,
                out_channels = 16,
                kernel_size=12,
                stride=1,
                padding=0
            ),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # Branch for global features
        self.global_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=24,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.global_conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=24,
                stride=1,
                padding=0
            ),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # Branch for local features

        self.local_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=12,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.local_conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=12,
                stride=1,
                padding=0
            ),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # LSTM

        self.lstm = nn.LSTM(
            input_size = 16128,
            hidden_size=64,
            num_layers=1,
            batch_first = True
        )

        # Speed estimation from feature vectors

        self.fc1 = nn.Sequential(
            nn.Linear(64,128),
            nn.ReLU()
        )
        self.regression = nn.Sequential(
            nn.Linear(128, 1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.N = x.size(0)
        self.L = x.size(1)
        self.H = x.size(3)
        self.W = x.size(4)
        NL = self.N * self.L
        log.debug("Input shape x: {}".format(x.shape))

        # combine batch and frame dimensions [N, L, 3, H, W] -> [N*L, 3, H, W]
        x2 = x.view(NL, self.channels_in, self.H, self.W)
        log.debug("Input shape of first_conv - x2: {}".format(x2.shape))

        # Apply first conv layer
        x3 = self.first_conv(x2)
        log.debug("Input shape of branches - x3: {}".format(x3.shape))

        # Branch for global features
        x4_1 = self.global_conv1(x3)
        x5_1 = self.global_conv2(x4_1)
        log.debug("Output shape of global branch - x5_1: {}".format(x5_1.shape))

        # Branch for local features
        x4_2 = self.local_conv1(x3)
        x5_2 = self.local_conv2(x4_2)
        log.debug("Output shape of local branch - x5_1: {}".format(x5_2.shape))

        # Flatten and concat both branches for LSTM
        x6_1 = x5_1.reshape(self.N, self.L, -1)
        x6_2 = x5_2.reshape(self.N, self.L, -1)
        x7 = torch.cat((x6_1, x6_2), dim=2)
        log.debug("Input shape of LSTM - x7: {}".format(x7.shape))

        # Apply LSTM
        x8, _ = self.lstm(x7)
        log.debug("Output of LSTM - x8: {}".format(x8.shape))

        # Apply fully connected layer
        x9 = self.fc1(x8)
        log.debug("Output of Fully Connected Layer - x9: {}".format(x9.shape))

        # Apply final regression layer
        x10 = self.regression(x9)
        log.debug("Output of regression layer - x10: {}".format(x10.shape))

        return x10.squeeze()

