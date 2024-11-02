from .modeling_base import BaseSubspaceWrapper


class ConvNdSubspaceWrapper(BaseSubspaceWrapper):
    def __init__(self, layer, theta, dint):
        super(ConvNdSubspaceWrapper, self).__init__(layer, theta, dint)

    def forward(self, x):
        weights, bias = super(ConvNdSubspaceWrapper, self).forward(x)
        return self.layer._conv_forward(x, weights, bias)
