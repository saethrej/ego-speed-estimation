'''

Processes configuration, sets paths, and fixes seeds for training and testing

'''


import os
from xmlrpc.client import Boolean
import yaml
from datetime import datetime as dt
import argparse
import numpy as np
import random as python_random
from random import randint
from box import Box
import logging as log
import torch

from src.initializer.initialization_dataset import dataset_config


def init_train(local: Boolean = False):

    base_path = "config" 

    # Load default config
    config = Box.from_yaml(filename=base_path + os.sep + "default.yaml", Loader=yaml.FullLoader)
    log.info("Loaded default configuration.")

    # Set ouput path for this run
    config.run_id = 'run_' + dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    if local:
        config.paths.input_path = config.paths.local.input_path
        config.paths.output_path = config.paths.local.output_path + os.sep + config.run_id
    else:
        config.paths.input_path = config.paths.euler.input_path
        config.paths.output_path = config.paths.euler.output_path + os.sep + config.run_id
    if not os.path.exists(config.paths.output_path):
        os.makedirs(config.paths.output_path)

    log.info("Run ID: {}.".format(config.run_id))
    log.info("Output Directory: {}.".format(config.paths.output_path))

    # Load dataset config
    config = dataset_config(config)
    log.info("Loaded dataset configuration.")

    # Fix seeds
    np.random.seed(config.seed)
    python_random.seed(config.seed)
    torch.manual_seed(config.seed)

    return config


def init_test(args):
    
    base_path = "config" 

    # Load default config
    config = Box.from_yaml(filename=base_path + os.sep + "default.yaml", Loader=yaml.FullLoader)
    log.info("Loaded default config")

    # Parse arguments
    if args.local:
        log.info("This is a local run.")
        config.paths = config.paths.local
    else:
        log.info("This is a run on euler.")
        config.paths = config.paths.euler
    log.info(config.paths.input_path)
    if args.weights:
        # Load weights according to given input argument
        log.info("Loading weights from input: ./out/{}/model_weights.pth".format(args.weights))
        config.paths.weights_path = "./out/" + args.weights + "/model_weights.pth"
    else:
        # Load weights according to configuration file
        log.info("Loading weights from config: {}".format(config.paths.weights_path))

    # Load dataset config
    config = dataset_config(config)
    log.info("Loaded dataset configuration.")

    # Fix seeds
    np.random.seed(config.seed)
    python_random.seed(config.seed)
    torch.manual_seed(config.seed)

    return config