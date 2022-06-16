"""

Loads dataset configuration file

"""


import os
import yaml
from box import Box


def dataset_config(config):

    if config.dataloader.dataset_name == 'comma-ai':
        config = _dataset_config_experimental(config)
    else:
        raise NotImplementedError

    return config

def _dataset_config_experimental(config):

    # Load dataset config
    base_path = "config" + os.sep + "dataset_config"
    config.dataset_config = Box.from_yaml(filename= base_path + os.sep + "commai-ai.yaml", Loader=yaml.FullLoader)
   
    return config