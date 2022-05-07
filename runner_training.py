
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

import logging as log
import argparse
from tqdm import tqdm, trange

from src.initializer.initialization import init_train
from src.train_loop import train_loop
from src.test_loop import test_loop
from src.models.runner_models import build_model
from src.data_loader.default import train_dataloader

# initialize logging environment
log.basicConfig(format='[%(module)15s @ %(asctime)s]: %(message)s', datefmt='%H:%M:%S', level=log.INFO)
log.debug("Initialized logger")

# parse command line arguments to see whether run is local
parser = argparse.ArgumentParser()
parser.add_argument("--local", "-l", action="store_true", help="inidicate that this is a local run")
args = parser.parse_args()

# Initialize training run 
if args.local:
    log.info("This is a local run.")
    config = init_train(True)
else:
    config = init_train()

# Get dataloader
dataloaders = train_dataloader(config)
log.info("Initialized dataloader.")

# Build model
model = build_model(config)
log.info("Built model. Starting training loop.")

# Training Loop
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=config.model.learning_rate)

epochs = 10
for t in range(epochs):
    log.info("Starting epoch {}/{}.".format(t+1, epochs))
    train_loop(dataloaders['train'], model, loss_fn, optimizer)
    test_loop(dataloaders['val'], model, loss_fn)
log.info("Training complete.")

# Store model weights
model_path = os.path.join(config.paths.output_path, 'model_weights.pth')
torch.save(model.state_dict(), model_path)
log.info("Model weights stored under {}.".format(model_path))