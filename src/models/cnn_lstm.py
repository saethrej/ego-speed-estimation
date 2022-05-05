
from tkinter import E
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

import logging as log


class CNNLSTM(nn.Module):
    def __init__(self, config):
        super(CNNLSTM, self).__init__()

        self.config = config
        
        self.resnet_layer = models.resnet18(pretrained=True)
        self.lstm_layer = nn.LSTM(input_size=256, hidden_size=8, num_layers=2)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x_3d):
        x_3d_size = x_3d.size()
        print(x_3d_size)
        # Flatten video sequences
        x_2d = torch.reshape(x_3d, (x_3d.size(dim=0) * x_3d.size(dim=1), x_3d.size(dim=2), x_3d.size(dim=3), x_3d.size(dim=4)))
        
        # Obtain CNN embedding
        # x_2d = self.resnet_layer(x_2d)
        x_2d = nn.Conv2d(3, 33, 3, stride=2)(x_2d)
        print(x_2d.size())
        exit()

        # Reshape into video sequences

        # Apply LSTM 

        # 
        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x
        return logits