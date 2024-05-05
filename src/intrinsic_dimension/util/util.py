import os
import random

import numpy as np
import torch


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # when running on the CuDNN backend, two further options must be set
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_params(model):
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()

    return total_params