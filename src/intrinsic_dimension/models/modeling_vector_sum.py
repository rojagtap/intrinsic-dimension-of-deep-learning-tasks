import torch


class VectorSum(torch.nn.Module):
    """
    given a vector theta_D optimize it for provided satisfying constraints
    for example, for D = 1000, optimize theta_D starting from a random normal
    to a vector such that first 100 elements sum to 1, next 100 sum to 2, and so on...
    """

    def __init__(self, D):
        super(VectorSum, self).__init__()
        self.D = D
        self.theta_D = torch.nn.Parameter(torch.empty(self.D))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.theta_D)

    def forward(self, _):
        return self.theta_D
