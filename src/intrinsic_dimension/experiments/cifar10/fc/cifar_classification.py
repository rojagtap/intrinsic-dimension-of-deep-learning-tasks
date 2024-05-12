import gc

import numpy as np
import torch
from torch.utils.data import DataLoader

from ....util.constants import DEVICE
from ....util.data import get_dataset
from ....util.util import set_seed, count_params, train, evaluate_accuracy
from ....util.plotter import plot_results, plot_model
from ....wrappers.modeling_fc import SequentialSubspaceWrapper, FC


def accuracy_criterion(logits, labels):
    return torch.sum(logits.argmax(1) == labels).item()


if __name__ == '__main__':
    # sum(input_dim x output_dim + bias)
    D = (32 * 32 * 3 * 200 + 200) + (200 * 200 + 200) + (200 * 10 + 10)

    # defining hyperparams
    lr = 2e-4
    epochs = 10
    batch_size = 64
    num_classes = 10
    hidden_size = 200
    print(f"Using {DEVICE}")

    basedir = "src/intrinsic_dimension/experiments/cifar10/fc"

    train_dataset, test_dataset = get_dataset("cifar10")
    sample_images, sample_labels = next(iter(DataLoader(test_dataset, batch_size=batch_size, shuffle=False)))
    sample_images, sample_labels = sample_images.to(DEVICE), sample_labels.to(DEVICE)

    # baseline
    set_seed(np.random.randint(10e8))
    model = FC(input_size=sample_images.size()[1:], hidden_size=hidden_size, num_classes=num_classes).to(DEVICE)
    total_params = 0

    # number of params in base model
    n_params = count_params(model)
    assert D == n_params
    print("n_params in base model: ", n_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train(model, torch.nn.functional.cross_entropy, optimizer, batch_size, train_dataset, epochs)
    baseline_accuracy = evaluate_accuracy(model, accuracy_criterion, train_dataset, batch_size)
    print(f"Baseline Accuracy: {baseline_accuracy}")
    plot_model(model, "baseline", basedir, sample_images, torch.nn.functional.cross_entropy, sample_labels)

    del model
    del optimizer
    gc.collect()

    # lr needs to be lower for training with intrinsic dim as we start with
    # 0 weights and hence objective space needs to be traversed in small steps
    lr = 1e-4

    # run for d values from 1 to 1200 with increment of 50
    history = {}
    for dint in range(1000, 12001, 500):
        set_seed(np.random.randint(10e8))

        # wrap all linear layers with the subspace layer
        base_model = FC(input_size=sample_images.size()[1:], hidden_size=hidden_size, num_classes=num_classes).to(DEVICE)
        model = SequentialSubspaceWrapper(base_model=base_model, dint=dint).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train(model, torch.nn.functional.cross_entropy, optimizer, batch_size, train_dataset, epochs)
        history[dint] = evaluate_accuracy(model, accuracy_criterion, test_dataset, batch_size)
        print(f"Accuracy: {history[dint]}, for dint: {dint}")

        if dint - 1 == 9000:
            n_params = count_params(model)
            print("n_params in intrinsic model: ", n_params)
            plot_model(model, "intrinsic_dim", basedir, sample_images, torch.nn.functional.cross_entropy, sample_labels)

        del model
        del optimizer
        del base_model
        gc.collect()

    plot_results(
        baseline=baseline_accuracy,
        dints=history.keys(),
        performance=history.values(),
        basedir=basedir,
        name=f"cifar10-{D}D",
        xlabel="subspace dim d",
        ylabel="validation accuracy",
        show_dint90=True
    )
