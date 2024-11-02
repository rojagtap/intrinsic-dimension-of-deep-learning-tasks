"""
module for container classes like Sequential
"""
import torch

from ..util.constants import LAYER_MAP


class SequentialSubspaceWrapper(torch.nn.Module):
    """
    wrapper class for torch.nn.Sequential
    """
    def __init__(self, base_model, dint, layer_map=LAYER_MAP, theta=None):
        if not isinstance(base_model, torch.nn.Sequential):
            raise ValueError(f"base_model expected to be of the type torch.nn.Sequential, got {type(base_model)}")

        super(SequentialSubspaceWrapper, self).__init__()

        if theta is None:
            # theta is shared across all layers
            self.theta = torch.nn.Parameter(torch.empty(dint))
            self.reset_parameters()
        else:
            self.theta = theta

        self.base_model = base_model

        # map layer to the corresponding wrapper layer if present
        layer_map = layer_map if layer_map is not None else {}
        for name, module in self.base_model.named_modules():
            if isinstance(module, tuple(layer_map.keys())):
                setattr(self.base_model, name, layer_map[type(module)](layer=module, theta=self.theta, dint=dint))

    def reset_parameters(self):
        self.theta.requires_grad_(True)
        torch.nn.init.zeros_(self.theta)

    def forward(self, x):
        return self.base_model(x)
