import torch

from ..wrappers.modeling_conv import ConvNdSubspaceWrapper
from ..wrappers.modeling_fc import LinearSubspaceWrapper

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

LAYER_MAP = {
    torch.nn.Linear: LinearSubspaceWrapper,
    torch.nn.Conv1d: ConvNdSubspaceWrapper,
    torch.nn.Conv2d: ConvNdSubspaceWrapper,
    torch.nn.Conv3d: ConvNdSubspaceWrapper
}
