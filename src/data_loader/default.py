"""

Triggers initialization of preprocessing modules
Initializes data loader for train/val/test and passes preprocessing modules to them
Fixes seeds in dataloader

"""


import os
import torch
from torchvision import datasets
import numpy as np

from src.preprocessing.runner_preprocessing import test_preprocessing, test_video_transform, train_preprocessing, train_video_transform
from src.data_loader.datasets.runner_datasets import get_video_dataset


def train_dataloader(config):  

    # Load data transforms
    data_transforms = train_preprocessing(config)

    # Load video transforms
    video_transforms = train_video_transform(config)

    # Build dataloader
    video_dataset = {x: get_video_dataset(config=config, mode=x, frame_transfroms=data_transforms, video_transforms=video_transforms) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(video_dataset[x], batch_size=config.model.batch_size, shuffle=True, worker_init_fn=np.random.seed(config.seed)) for x in ['train', 'val']}

    return dataloaders

def test_dataloader(config):  

    # Load data transforms
    data_transforms = test_preprocessing(config)

    # Load video transforms
    video_transforms = test_video_transform(config)

    # Build dataloader
    video_dataset = get_video_dataset(config=config, mode="test", frame_transfroms=data_transforms, video_transforms=video_transforms)
    dataloader = torch.utils.data.DataLoader(video_dataset, batch_size=config.model.batch_size, shuffle=False, worker_init_fn=np.random.seed(config.seed))

    return dataloader