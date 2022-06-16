"""

The testing script.

"""

import os
import torch
import math
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

import logging as log
import argparse
from tqdm import tqdm, trange

from src.initializer.initialization import init_test
from src.data_loader.default import test_dataloader
from src.models.runner_models import build_model
from src.test_loop import test_loop

# initialize logging environment
def test(args):
    log.basicConfig(format='[%(module)15s @ %(asctime)s]: %(message)s', datefmt='%H:%M:%S', level=log.INFO)
    log.debug("Initialized logger")

    # Load config
    if args.local:
        log.info("This is a local run.")
    config = init_test(args)

    # Select device
    device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info("Using device: {}".format(device))

    # Get Dataloader
    dataloader = test_dataloader(config)

    # Build model and load weights
    model = build_model(config)
    model.load_state_dict(torch.load(config.paths.weights_path))
    model.to(device)
    model.eval()
    log.info("Built model. Start prediction loop.")

    # Define loss metric
    loss_fn = nn.MSELoss()

    # Start test loop
    val_loss = test_loop(dataloader, model, loss_fn, device, True)

    log.info("Testing complete.")

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", "-l", action="store_true", help="inidicate that this is a local run.")
    parser.add_argument("--weights", "-w", type=str, help="Subdirectory of out with weights that should be used.")
    args = parser.parse_args()
    test(args)