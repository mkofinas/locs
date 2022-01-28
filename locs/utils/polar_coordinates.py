import numpy as np
import torch


def cart2polar(x):
    """Transform Cartesian to Polar Coordinates

    x: torch.Tensor, [dims] x 2
    return rho, theta
    """
    rho = torch.norm(x, p=2, dim=-1).unsqueeze(-1)
    theta = torch.atan2(x[..., 1], x[..., 0]).unsqueeze(-1)
    theta = theta + (theta < 0).type_as(theta) * (2 * np.pi)

    return rho, theta
