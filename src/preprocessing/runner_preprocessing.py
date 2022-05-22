from src.data_loader import default
from src.preprocessing.optical_flow import OpticalFlow, OpticalFlowDepth


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
        return OpticalFlow()
    elif config.preprocessing.video_transform == 'depth_opticalflow':
        return OpticalFlowDepth()
    else:
        return None

def test_video_transform(config):
    if config.preprocessing.video_transform == 'opticalflow':
        return OpticalFlow()
    elif config.preprocessing.video_transform == 'depth_opticalflow':
        return OpticalFlowDepth()
    else:
        return None