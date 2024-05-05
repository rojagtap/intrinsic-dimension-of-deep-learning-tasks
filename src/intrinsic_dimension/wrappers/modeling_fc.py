import numpy as np
import torch


class LinearSubspaceWrapper(torch.nn.Module):
    """
    wrapper for linear pytorch layer for using low dimensional weight search
    here we make the original weights non-trainable and introduce non-trainable
    P (projection vector), and theta (low dim parameter vector) which is trainable
    for each weight entity (in this case, weight and bias).

    size of theta is (dint,)
    size of P is (flat size of weight, dint)

    so P x theta will give a vector of weight size which can be reshaped for addition
    """

    def __init__(self, layer, theta, dint):
        """
        theta will be shared across layers, but the projection matrix P will be unique to layer
        """

        super(LinearSubspaceWrapper, self).__init__()
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

        return torch.nn.functional.linear(x, weight, bias)


class SequentialSubspaceWrapper(torch.nn.Module):
    def __init__(self, base_model, dint):
        super(SequentialSubspaceWrapper, self).__init__()

        # theta is share across all layers
        self.base_model = base_model
        self.theta = torch.nn.Parameter(torch.empty(dint))
        self.reset_parameters()

        for name, module in self.base_model.linear_relu_stack.named_modules():
            if isinstance(module, torch.nn.Linear) and name.startswith('linear'):
                setattr(self.base_model.linear_relu_stack, name, LinearSubspaceWrapper(layer=module, theta=self.theta, dint=dint))

    def reset_parameters(self):
        self.theta.requires_grad_(True)
        torch.nn.init.zeros_(self.theta)

    def forward(self, x):
        return self.base_model(x)
