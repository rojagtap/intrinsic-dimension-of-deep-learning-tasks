import numpy as np
import torch


class FC(torch.nn.Module):
    def __init__(self, input_shape):
        super(FC, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(np.prod(input_shape), 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
