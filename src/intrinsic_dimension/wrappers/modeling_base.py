import torch
import numpy as np


class BaseSubspaceWrapper(torch.nn.Module):
    def __init__(self, layer, theta, dint):
        super(BaseSubspaceWrapper, self).__init__()
        self.dint = dint
        self.layer = layer
        self.theta = theta

        # P (projection matrix) here is basically the flat size of the weight vector (or bias vector) x dint
        self.P_weight = torch.nn.Parameter(torch.empty(np.prod(self.layer.weight.size()), self.dint))
        if self.layer.bias is not None:
            self.P_bias = torch.nn.Parameter(torch.empty(np.prod(self.layer.bias.size()), self.dint))
        self.reset_parameters()

    def reset_parameters(self):
        self.P_weight.requires_grad_(False)
        if self.layer.bias is not None:
            self.P_bias.requires_grad_(False)

        for parameter in self.layer.parameters():
            parameter.requires_grad_(False)

        torch.nn.init.normal_(self.P_weight)
        if self.layer.bias is not None:
            torch.nn.init.normal_(self.P_bias)

    def forward(self, x):
        delta_theta_weight = self.P_weight @ self.theta
        weight = self.layer.weight + delta_theta_weight.view(self.layer.weight.size())

        bias = None
        if self.layer.bias is not None:
            delta_theta_bias = self.P_bias @ self.theta
            bias = self.layer.bias + delta_theta_bias.view(self.layer.bias.size())

        return weight, bias
