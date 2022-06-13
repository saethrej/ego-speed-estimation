import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

import logging as log


class BandariBaseline3DConv(nn.Module):
    def __init__(self, config):
        super(BandariBaseline3DConv, self).__init__()

        # store config file if needed
        self.config = config
        self.N = self.config.model.batch_size
        self.L = self.config.dataset_config.sample_length
        
        # define 3d convolution layers
        self.conv_layer_1 = self._3d_conv_layer(3, 32, (2,2,3))
        self.conv_layer_2 = self._3d_conv_layer(32, 64, (2,2,3))

        # define flatten
        self.flatten = nn.Flatten()

        # define fc
        self.fc1 = nn.Linear(22848, 1028)
        self.fc2 = nn.Linear(1028, self.config.dataset_config.sample_length)

    def _3d_conv_layer(self, in_channels, out_channels, kernel):
        '''returns a 3D convolution layer with activation and pooling'''

        return nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=(2,2,2),
                padding=0
            ),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )

    def forward(self, x: torch.Tensor):
        self.N = x.size(0)
        log.debug("Input shape to forward() - x: {}".format(x.shape))

        # permute input for 3d convolution [N,C,D,H,W]
        x1 = torch.permute(x, (0, 2, 1, 3,4))
        log.debug("Output of permute() - x1: {}".format(x1.shape))

        # apply 1st 3d convolutional layer
        x3 = self.conv_layer_1(x1)
        log.debug("Output of conv3d() - x3: {}".format(x3.shape))

        # apply 2nd 3d convolutional layer
        x4 = self.conv_layer_2(x3)
        log.debug("Output of conv3d() - x4: {}".format(x4.shape))

        # flatten for FC
        x5= self.flatten(x4)
        log.debug("Input to Fully Connected Layer - x5: {}".format(x5.shape))
        
        # apply fully connected layers
        x6 = self.fc1(x5)
        log.debug("Output of Fully Connected Layer - x6: {}".format(x6.shape))

        x7 = self.fc2(x6)
        log.debug("Output of regression layer - x7: {}".format(x7.shape))

        return x7.squeeze()



