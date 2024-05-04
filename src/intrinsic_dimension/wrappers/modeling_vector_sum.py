import torch


class LowDimWrapper(torch.nn.Module):
    """
    wrapper for DimSum where theta_D is not trainable and a vector
    from its subspace is optimized to find the real theta*_D
    """

    def __init__(self, model, dint, seed=42):
        super(LowDimWrapper, self).__init__()
        self.dint = dint
        self.model = model
        self.theta_dint = torch.nn.Parameter(torch.empty(self.dint))
        self.P = torch.nn.Parameter(torch.empty(self.model.D, self.dint))
        self.reset_parameters(seed)

    def reset_parameters(self, seed):
        self.P.requires_grad_(False)
        self.theta_dint.requires_grad_(True)
        self.model.theta_D.requires_grad_(False)

        torch.manual_seed(seed)
        torch.nn.init.normal_(self.P)
        torch.nn.init.zeros_(self.theta_dint)

    def forward(self, _):
        delta_theta = self.theta_dint @ self.P.T
        return self.model(_) + delta_theta
