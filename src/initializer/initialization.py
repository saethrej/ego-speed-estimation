
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
    # TODO: Fix seeds for torch
    np.random.seed(config.seed)
    python_random.seed(config.seed)

    return config


def init_test():
    
    base_path = "config" 

    # Load default config
    config = Box.from_yaml(filename=base_path + os.sep + "default.yaml", Loader=yaml.FullLoader)
    print("Initializer: Loaded default config")

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_run', type=str, required=True,
                    help='Load pretrained model from run_XXX')
    args = parser.parse_args()
    
    config.load_run = args.load_run
    config.run_id = args.load_run

    # Set ouput path according to selected run
    config.paths.output_path = config.paths.output_path + os.sep + config.load_run

    # Load dataset config
    config = dataset_config(config)

    # Fix seeds
    # TODO: Fix seeds for torch
    np.random.seed(config.seed)
    python_random.seed(config.seed)

    return config