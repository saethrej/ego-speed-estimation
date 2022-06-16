"""

This file loads the preprocessing module that was specified in the configuration

"""

import logging as log

from src.data_loader import default
from src.preprocessing.optical_flow import OpticalFlow, OpticalFlowDepth, OpticalFlowGrayScale


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
        log.info("Selected Preprocessing: Optical Flow")
        return OpticalFlow()
    elif config.preprocessing.video_transform == 'opticalflow_gray':
        log.info("Selected Preprocessing: Optical Flow, Grayscale")
        return OpticalFlowGrayScale()
    elif config.preprocessing.video_transform == 'depth_opticalflow':
        log.info("Selected Preprocessing: Optical Flow, Depth Estimation, Grayscale")
        return OpticalFlowDepth()
    else:
        log.info("No preprocessing selected")
        return None

def test_video_transform(config):
    if config.preprocessing.video_transform == 'opticalflow':
        log.info("Selected Preprocessing: Optical Flow")
        return OpticalFlow()
    elif config.preprocessing.video_transform == 'opticalflow_gray':
        log.info("Selected Preprocessing: Optical Flow, Grayscale")
        return OpticalFlowGrayScale()
    elif config.preprocessing.video_transform == 'depth_opticalflow':
        log.info("Selected Preprocessing: Optical Flow, Depth Estimation, Grayscale")
        return OpticalFlowDepth()
    else:
        log.info("No preprocessing selected")
        return None