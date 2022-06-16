'''

Implementation of the CNN (ours) LSTM model

'''

import torch
from torch import nn

import logging as log

class DOFCNNLSTM(nn.Module):
    def __init__(self, config) -> None:
        super(DOFCNNLSTM, self).__init__()

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

        # LSTM

        self.lstm = nn.LSTM(
            input_size = 4224,
            hidden_size = 64,
            num_layers=1,
            batch_first=True
        )

        # FC applied on output of LSTM
        self.fc1 = nn.Sequential(
            nn.Linear(64, 64),
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

        # Combine first two dimensions and apply conv layers
        x2 = x.reshape(NL, self.channels_in, self.H, self.W)
        log.debug("Input shape to conv - x2: {}".format(x2.shape))
        x3 = self.conv_1(x2)
        log.debug("Output of first conv - x3: {}".format(x3.shape))
        x4 = self.conv_2(x3)
        log.debug("Output of second conv - x4: {}".format(x4.shape))
        x4 = self.conv_3(x4)
        log.debug("Output of third conv - x4 {}".format(x4.shape))


        # Flatten and apply LSTM
        x4 = x4.reshape(self.N, self.L, -1)
        log.debug("Input shape of LSTM - x4: {}".format(x4.shape))
        x5, _ = self.lstm(x4)
        log.debug("Output of LSTM - x5: {}".format(x5.shape))

        # Apply fully connected an final regression layer
        x6 = self.fc1(x5)
        log.debug("Output of Fully Connected Layer - x6: {}".format(x6.shape))
        x7 = self.regression(x6)
        log.debug("Output of regression layer - x7: {}".format(x7.shape))
        
        return x7.squeeze()