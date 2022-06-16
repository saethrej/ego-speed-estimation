'''

Implementation of the Bandari CNN LSTM model

'''

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

import logging as log


class BandariBaseline(nn.Module):
    def __init__(self, config):
        super(BandariBaseline, self).__init__()

        # store config file if needed
        self.config = config
        self.N = self.config.model.batch_size
        self.L = self.config.dataset_config.sample_length
        
        # set number of channels according to preprocessing steps
        self.channels_in = 3
        if config.preprocessing.video_transform == 'depth_opticalflow':
            self.channels_in = 4
        elif config.preprocessing.video_transform == 'opticalflow_gray':
            self.channels_in = 3
        elif config.preprocessing.video_transform == 'opticalflow':
            self.channels_in = 2
        
        # 1 conv layer
        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels_in,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(2),
            nn.ReLU()
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=130416,
            hidden_size=256,
            num_layers=1,
            batch_first = True
        )

        # FC applied on output of LSTM
        self.fc1 = nn.Linear(256, 256)
        self.regression = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        self.N = x.size(0)
        log.debug("Input shape to forward() - x: {}".format(x.shape))

        # combine batch and frame dimensions of 5D input tensor
        # [N,L,3,H,W] --> [N*L,3,H,W]
        NL = self.N * self.L
        x2 = x.view(NL, self.channels_in, x.size(3), x.size(4))
        log.debug("Input shape to conv2d() - x2: {}".format(x2.shape))

        # apply convolutional layer
        x3 = self.conv_layer(x2)
        log.debug("Output of conv2d() - x3: {}".format(x3.shape))
        
        # flatten for LSTM
        x4 = x3.view(self.N, self.L, -1)
        log.debug("Input to LSTM - x4: {}".format(x4.shape))

        # apply LSTM
        x5, _ = self.lstm(x4)
        log.debug("Output of LSTM - x5: {}".format(x5.shape))

        # apply fully connected layer
        x6 = self.fc1(x5)
        log.debug("Output of Fully Connected Layer - x6: {}".format(x6.shape))

        # apply final regression layer
        x7 = self.regression(x6)
        log.debug("Output of regression layer - x7: {}".format(x7.shape))

        return x7.squeeze()



