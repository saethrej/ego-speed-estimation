
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from src.initializer.initialization import init_train
from src.train_loop import train_loop
from src.test_loop import test_loop
from src.models.runner_models import build_model
from src.data_loader.default import train_dataloader


# Initialize training run
config = init_train()

# Get dataloader
dataloaders = train_dataloader(config)

# Build model
model = build_model(config)

# Training Loop
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=config.model.learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(dataloaders['train'], model, loss_fn, optimizer)
    test_loop(dataloaders['val'], model, loss_fn)
print("Done!")

# Store model weights
output_path = config.paths.output_path + os.sep + 'model_weights'
filename = 'model_weights.pth'
torch.save(model.state_dict(), output_path + os.sep + filename)