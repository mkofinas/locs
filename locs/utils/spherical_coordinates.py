import numpy as np
import torch
import kornia


def cart2spherical(x):
    """Transform Cartesian to Spherical Coordinates

    x: torch.Tensor, [dims] x 3
    return rho, phi, theta
    Physics convention, phi: azimuth angle, angle in x-y plane
    """
    rho = torch.norm(x, p=2, dim=-1).unsqueeze(-1)
    phi = torch.atan2(x[..., 1], x[..., 0]).unsqueeze(-1)
    phi = phi + (phi < 0).type_as(phi) * (2 * np.pi)

    EPS = 1e-8
    theta = torch.acos(
        torch.clamp(x[..., 2] / (rho + EPS).squeeze(-1), min=-1.0, max=1.0)
    ).unsqueeze(-1)

    return rho, phi, theta


def R_sph(phi, theta):
    """
    phi: azimuth angle
    theta: elevation angle
    """
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    R = torch.stack([
            torch.cat([cos_phi * cos_theta, sin_phi * cos_theta, -sin_theta], -1),
            torch.cat([-sin_phi, cos_phi, torch.zeros_like(cos_phi)], -1),
            torch.cat([cos_phi * sin_theta, sin_phi * sin_theta, cos_theta], -1)], -2)
    return R


def quat2euler(q):
    """
    q: scalar-last format
    return yaw, pitch, roll
    """
    q1, q2, q3, q0 = torch.chunk(q, 4, dim=-1)
    return torch.cat([
        torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2)),
        torch.asin(torch.clamp(2 * (q0 * q2 - q3 * q1), min=-1.0, max=1.0)),
        torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2)),
    ], -1)


def rotation_matrix_to_euler(r):
    return quat2euler(kornia.conversions.rotation_matrix_to_quaternion(r))
