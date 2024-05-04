"""
implementing the example problem mentioned in the paper
"""
import os
import gc
import numpy as np
import torch
import matplotlib.pyplot as plt

from ...util.constants import DEVICE
from ...wrappers.modeling_vector_sum import LowDimWrapper


class VectorSum(torch.nn.Module):
    """
    given a vector theta_D optimize it for provided satisfying constraints
    for example, for D = 1000, optimize theta_D starting from a random normal
    to a vector such that first 100 elements sum to 1, next 100 sum to 2, and so on...
    """

    def __init__(self, D, seed=42):
        super(VectorSum, self).__init__()
        self.D = D
        self.theta_D = torch.nn.Parameter(torch.empty(self.D))
        self.reset_parameters(seed)

    def reset_parameters(self, seed):
        torch.manual_seed(seed)
        torch.nn.init.normal_(self.theta_D)

    def forward(self, _):
        return self.theta_D


def apply_constraint(y, n_sums):
    return y.view(n_sums, y.size()[0] // n_sums).sum(1)


def score_fn(input, target):
    """
    as defined in the paper, score for this problem is given as exp(-loss)
    """
    return round(torch.exp(-torch.nn.functional.mse_loss(input, target)).item(), 2)


def search(model, optimizer, n_sums):
    target = torch.arange(1, n_sums + 1, dtype=torch.float32).to(DEVICE)

    epoch = 0
    model.train()
    for epoch in range(epoch, 10000):
        optimizer.zero_grad()

        y = model(None)
        y = apply_constraint(y, n_sums)
        loss = torch.nn.functional.mse_loss(y, target)

        if loss <= optimizer.param_groups[0]['lr']:
            break

        loss.backward()
        optimizer.step()

    model.eval()
    target = target.cpu().detach()
    result = apply_constraint(model(None).cpu().detach(), n_sums)
    performance = score_fn(result, target)
    return epoch, result, performance


if __name__ == '__main__':
    D = 1000
    n_sums = 10
    print(f"Using {DEVICE} device")

    bubble_size = 100
    fig, ax = plt.subplots()

    seed = np.random.randint(10e8)
    model = VectorSum(D, seed=seed).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    epoch, result, score = search(model, optimizer, n_sums)
    print("Base: Vector after epoch {}: {} with score: {}".format(epoch, result, score))

    ax.axhline(y=score, linestyle='--', color='black', linewidth=1)

    del seed
    del model
    del epoch
    del result
    del optimizer
    gc.collect()

    history = {}
    for dint in range(1, 51):
        seed = np.random.randint(10e8)
        model = LowDimWrapper(VectorSum(D, seed=seed).to(DEVICE), dint, seed=seed).to(DEVICE)

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        epoch, result, history[dint] = search(model, optimizer, n_sums)

        print("Vector after epoch {}: {} for intrinsic dim: {} with score: {}".format(epoch, result, dint, history[dint]))

        del seed
        del model
        del epoch
        del result
        del optimizer
        gc.collect()

    ax.scatter(history.keys(), history.values(), s=bubble_size, alpha=0.5, edgecolors='black', color='navy')
    ax.plot(history.keys(), history.values(), linestyle='-', color='navy', linewidth=1)

    ax.set_xlabel('d_intrinsic_dim')
    ax.set_ylabel('Performance (exp(-loss))')

    basedir = "intrinsic_dimension/experiments/example"
    # Show plot
    if not os.path.exists(os.path.join(basedir, "plot")):
        os.mkdir(os.path.join(basedir, "plot"))

    plt.savefig(os.path.join(basedir, "plot/vector-sum-{}D.png".format(D)))
