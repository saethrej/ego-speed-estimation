"""

Definition of test/validation loop

"""


import torch
import logging as log

def test_loop(dataloader, model, loss_fn, device, test=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item() * X.shape[0]

    test_loss /= size
    if test:
        log.info(f"TEST LOSS (MSE): {test_loss:>8f} \n")
    else:
        log.info(f"[Validation] Avg Batch Loss: {test_loss:>8f} \n")
    return test_loss