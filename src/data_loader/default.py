
import os
import torch
from torchvision import datasets

from src.preprocessing.default import test_preprocessing, train_preprocessing
from src.data_loader.datasets.runner_datasets import get_video_dataset


def train_dataloader(config):  

    # Load data transforms
    data_transforms = train_preprocessing(config)

    # Build dataloader
    video_dataset = {x: get_video_dataset(config=config, mode=x, frame_transfroms=data_transforms) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(video_dataset[x], batch_size=config.model.batch_size, shuffle=True) for x in ['train', 'val']}

    return dataloaders

def test_dataloader(config):  

   raise NotImplementedError