from src.data_loader import default
from src.preprocessing.optical_flow import train_optical_flow


def train_preprocessing(config):

    if config.preprocessing.data_transform == 'default':
        return default.train_preprocessing(config)
    else:
        return None


def test_preprocessing(config):

    if config.preprocessing.data_transform == 'default':
        return default.test_preprocessing(config)
    else:
        return None

def train_video_transform(config):
    if config.preprocessing.video_transform == 'opticalflow':
        return train_optical_flow
    else:
        return None

def test_video_transform(config):
    if config.preprocessing.video_transform == 'opticalflow':
        return train_optical_flow
    else:
        return None