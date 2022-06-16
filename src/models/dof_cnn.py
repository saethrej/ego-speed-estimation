'''

Implementation of the CNN (ours) model with optional temporal smoothing

'''

import torch
from torch import nn

import logging as log

class DOFCNN(nn.Module):
    def __init__(self, config) -> None:
        super(DOFCNN, self).__init__()

        #store config
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

        # Read from config if temporal smoothing (i.e. 1D conv) should be applied
        self.no_temp_smoothing = (config.model.settings and config.model.settings == 'no_temp')
        if self.no_temp_smoothing:
            log.info("Model settings: No temporal smoothing")

        # 3 conv layers of our CNN
        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels = self.channels_in,
                out_channels = 12,
                kernel_size = 5,
                stride=1,
                padding=0
            ),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 12,
                out_channels = 12,
                kernel_size = 5,
                stride=1,
                padding=0
            ),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(
                in_channels = 12,
                out_channels = 12,
                kernel_size = 5,
                stride=1,
                padding=0
            ),
            nn.AvgPool2d(2),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )

        # FC applied after CNN
        self.fc1 = nn.Sequential(
            nn.Linear(4224, 64),
            nn.ReLU()
        )
        self.regression = nn.Sequential(
            nn.Linear(64, 1),
            nn.ReLU()
        )

        # Optional temporal smoothing layer
        self.final_1dconv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding='same',
            padding_mode='replicate'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.N = x.size(0)
        self.L = x.size(1)
        self.H = x.size(3)
        self.W = x.size(4)
        NL = self.N * self.L
        log.debug("Input shape x: {}".format(x.shape))

        # Combine first two dimensions and apply conv layers
        x2 = x.reshape(NL, self.channels_in, self.H, self.W)
        log.debug("Input shape to conv - x2: {}".format(x2.shape))
        x3 = self.conv_1(x2)
        log.debug("Output of first conv - x3: {}".format(x3.shape))
        x4 = self.conv_2(x3)
        log.debug("Output of second conv - x4: {}".format(x4.shape))
        x4 = self.conv_3(x4)
        log.debug("Output of third conv - x4 {}".format(x4.shape))


        # Flatten and apply linear layer
        x4 = x4.reshape(self.N, self.L, -1)
        log.debug("Input to first linear layer - x4: {}".format(x4.shape))
        x5 = self.fc1(x4)
        log.debug("Input to regression layer - x5: {}".format(x5.shape))

        # Apply final regression layer
        x6 = self.regression(x5)
        log.debug("Output regression - x6: {}".format(x6.shape))
        if self.no_temp_smoothing:
            return x6.squeeze()
        x7 = torch.transpose(x6, 2, 1)
        x7 = self.final_1dconv(x7)
        return x7.squeeze()