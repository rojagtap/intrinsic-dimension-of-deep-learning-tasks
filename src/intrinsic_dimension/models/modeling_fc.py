from collections import OrderedDict

import numpy as np
import torch


class FC(torch.nn.Module):
    """
    as defined in the paper, we train a 786-200-200-10 fc network on the mnist dataset
    """

    def __init__(self, input_size, hidden_size, num_classes, n_hidden_layers=1, use_bias=True):
        super(FC, self).__init__()
        self.flatten = torch.nn.Flatten()

        layers = OrderedDict([
            ("linear_input", torch.nn.Linear(np.prod(input_size), hidden_size, bias=use_bias)),
            ("relu_input", torch.nn.ReLU())
        ])
        for i in range(n_hidden_layers):
            layers[f"linear_{i}"] = torch.nn.Linear(hidden_size, hidden_size, bias=use_bias)
            layers[f"relu_{i}"] = torch.nn.ReLU()
        layers["linear_output"] = torch.nn.Linear(hidden_size, num_classes, bias=use_bias)

        self.linear_relu_stack = torch.nn.Sequential(layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
