
import os
import torch
import math
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

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log.info("Using device: {}".format(device))
torch.backends.cudnn.benchmark = True

# Get dataloader
dataloaders = train_dataloader(config)
log.info("Initialized dataloader.")

# Build model
model = build_model(config)
model.load_state_dict(torch.load('/cluster/project/infk/courses/252-0579-00L/group10/ego-speed-estimation/out/run_2022-05-24_09-28-27/model_weights.pth'))
model.to(device)
model_path = os.path.join(config.paths.output_path, 'model_weights.pth')
log.info("Built model. Starting training loop.")

# Training Loop
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate)

best_loss = float('inf')
num_epochs = config.model.num_epochs

for t in range(num_epochs):
    model.train()
    log.info("Starting epoch {}/{}.\n".format(t+1, num_epochs) )
    train_loop(dataloaders['train'], model, loss_fn, optimizer, device)
    model.eval()
    val_loss = test_loop(dataloaders['val'], model, loss_fn, device)

    # save model if it has the best validation score
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), model_path)
        log.info("Model weights stored under {}.".format(model_path))

log.info("Training complete.")


