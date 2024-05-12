import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .constants import DEVICE


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # when running on the CuDNN backend, two further options must be set
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_params(model):
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()

    return total_params


def train(model, loss_fn, optimizer, batch_size, dataset, epochs):
    """
    train the model for the provided dataset
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for step, (batch, labels) in enumerate(dataloader):
            optimizer.zero_grad()

            batch, labels = batch.to(DEVICE), labels.to(DEVICE)
            logits = model(batch)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / (len(dataset) // batch_size)}')


def evaluate_accuracy(model, criterion, dataset, batch_size):
    """
    get test accuracy
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size)

    correct = 0
    for step, (batch, labels) in enumerate(dataloader):
        batch, labels = batch.to(DEVICE), labels.to(DEVICE)
        logits = model(batch)
        correct += criterion(logits, labels)

    return 100 * correct / len(dataset)
