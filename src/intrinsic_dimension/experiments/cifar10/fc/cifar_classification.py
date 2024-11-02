import gc

import numpy as np
import torch
from torch.utils.data import DataLoader

from ....models.modeling_fc import FC
from ....util.constants import DEVICE
from ....util.data import get_dataset
from ....util.plotter import plot_results, plot_model
from ....util.util import set_seed, count_params, train, evaluate_accuracy
from ....wrappers.modeling_container import SequentialSubspaceWrapper


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
    lr = 1e-5

    # run for d values from 1000 to 12001 with increment of 500
    history = {}
    for dint in range(1000, 12001, 500):
        set_seed(np.random.randint(10e8))

        # wrap all linear layers with the subspace layer
        model = FC(input_size=sample_images.size()[1:], hidden_size=hidden_size, num_classes=num_classes).to(DEVICE)
        model.linear_relu_stack = SequentialSubspaceWrapper(base_model=model.linear_relu_stack, dint=dint).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train(model, torch.nn.functional.cross_entropy, optimizer, batch_size, train_dataset, epochs)
        history[dint] = evaluate_accuracy(model, accuracy_criterion, test_dataset, batch_size)
        print(f"Accuracy: {history[dint]}, for dint: {dint}")

        if dint == 9000:
            n_params = count_params(model)
            print("n_params in intrinsic model: ", n_params)
            plot_model(model, "intrinsic_dim", basedir, sample_images, torch.nn.functional.cross_entropy, sample_labels)

        del model
        del optimizer
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
