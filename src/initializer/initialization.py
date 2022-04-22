
import os
import yaml
import argparse
import numpy as np
import random as python_random
from random import randint
from box import Box

from src.initializer.initialization_dataset import dataset_config


def init_train():

    base_path = "config" 

    # Load default config
    config = Box.from_yaml(filename=base_path + os.sep + "default.yaml", Loader=yaml.FullLoader)
    print("Initializer: Loaded default config")

    # Set ouput path for this run
    config.run_id = 'run_' + str(randint(1000, 9999))
    config.paths.output_path = config.paths.output_path + os.sep + config.run_id
    if not os.path.exists(config.paths.output_path):
            os.makedirs(config.paths.output_path)

    print("Initializer: Run ID", config.run_id)
    print("Initializer: Ouput stored under", config.paths.output_path )

    # Load dataset config
    config = dataset_config(config)

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