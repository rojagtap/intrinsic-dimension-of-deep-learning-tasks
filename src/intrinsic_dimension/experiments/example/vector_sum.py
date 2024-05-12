"""
implementing the example problem mentioned in the paper
"""
import gc

import numpy as np
import torch

from ...util.util import set_seed
from ...util.constants import DEVICE
from ...util.plotter import plot_results
from ...wrappers.modeling_vector_sum import VectorSubspaceWrapper, VectorSum


def apply_constraint(y, n_sums):
    return y.view(n_sums, y.size()[0] // n_sums).sum(1)


def score_fn(inputs, target):
    """
    as defined in the paper, score for this problem is given as exp(-loss)
    """
    return round(torch.exp(-torch.nn.functional.mse_loss(inputs, target)).item(), 2)


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
    print(f"Using {DEVICE}")

    set_seed(np.random.randint(10e8))
    model = VectorSum(D).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    epoch, result, score = search(model, optimizer, n_sums)
    print(f"Base: Vector after epoch {epoch}: {result} with score: {score}")

    del model
    del epoch
    del result
    del optimizer
    gc.collect()

    history = {}
    for dint in range(1, 51):
        set_seed(np.random.randint(10e8))
        model = VectorSubspaceWrapper(VectorSum(D).to(DEVICE), dint).to(DEVICE)

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        epoch, result, history[dint] = search(model, optimizer, n_sums)

        print(f"Vector after epoch {epoch}: {result} for intrinsic dim: {dint} with score: {history[dint]}")

        del model
        del epoch
        del result
        del optimizer
        gc.collect()

    plot_results(
        baseline=score,
        dints=history.keys(),
        performance=history.values(),
        basedir="intrinsic_dimension/experiments/example",
        name=f"vector-sum-{D}D",
        xlabel="d",
        ylabel="performance (exp(-loss))"
    )
