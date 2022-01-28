import numpy as np
import torch
from pytorch3d.transforms.rotation_conversions import matrix_to_euler_angles

from locs.utils import spherical_coordinates, polar_coordinates


def wrap_angles(theta, normalize=False):
    theta = theta + (theta <= -np.pi).type_as(theta) * (2 * np.pi)
    theta = theta - (theta > np.pi).type_as(theta) * (2 * np.pi)
    if normalize:
        theta = theta / np.pi
    return theta


def canonicalize_inputs(inputs, use_3d=False, trans_only=False):
    if use_3d:
        if inputs.size(-1) != 6:
            raise NotImplementedError
        if trans_only:
            vel = inputs[..., 3:]
            Rinv = None
            canon_inputs = torch.zeros_like(inputs)
            canon_inputs[..., 3:] = vel
        else:
            vel = inputs[..., 3:]
            # Rinv = kornia.angle_axis_to_rotation_matrix(vel.reshape(-1, 3)).reshape(*vel.shape[:-1], 3, 3)

            _, phi, theta = spherical_coordinates.cart2spherical(vel)
            r = spherical_coordinates.R_sph(phi, theta)
            Rinv = r.transpose(-1, -2)

            rot_vel = (r @ vel.unsqueeze(-1)).squeeze(-1)
            canon_inputs = torch.cat([torch.zeros_like(inputs[..., :3]), rot_vel], dim=-1)
    else:
        if trans_only:
            vel = inputs[..., 2:]
            Rinv = None
            canon_inputs = torch.zeros_like(inputs)
            canon_inputs[..., 2:] = vel
        else:
            vel = inputs[..., 2:]
            angle = torch.atan2(vel[..., 1], vel[..., 0])
            # R = rotation_matrix(-angle)
            Rinv = rotation_matrix(angle)
            # rot_vel = (R @ vel.unsqueeze(-1)).squeeze(-1)
            # canon_inputs = torch.cat([torch.zeros_like(rot_vel), rot_vel], dim=-1)
            canon_inputs = torch.zeros_like(inputs)
            canon_inputs[..., 2] = torch.norm(vel, dim=-1)
    return canon_inputs, Rinv


def rotation_matrix(theta):
    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)
    return torch.stack([torch.stack([costheta, sintheta], -1),
                        torch.stack([-sintheta, costheta], -1)], -1)


def inv_rotation(y_pred, Rinv):
    return torch.cat(
        [(Rinv @ y_pred[..., :2].unsqueeze(-1)).squeeze(-1),
         (Rinv @ y_pred[..., 2:].unsqueeze(-1)).squeeze(-1)], -1)


def inv_rotation3d(y_pred, Rinv):
    return torch.cat(
        [(Rinv @ y_pred[..., :3].unsqueeze(-1)).squeeze(-1),
         (Rinv @ y_pred[..., 3:].unsqueeze(-1)).squeeze(-1)], -1)


def sender_receiver_features(x, send_edges, recv_edges, batched=False):
    """
    batched: used in dynamicvars settings
    """
    if batched:
        x_j = x[send_edges]
        x_i = x[recv_edges]
    else:
        x_j = x[:, send_edges]
        x_i = x[:, recv_edges]

    return x_j, x_i


def create_trans_edge_attr_pos_vel(x, send_edges, recv_edges, batched=False):
    x_j, x_i = sender_receiver_features(
        x, send_edges, recv_edges, batched=batched)

    # delta_yaw is the signed difference in yaws
    delta_yaw = angle_diff(x_i[..., 2:], x_j[..., 2:]).unsqueeze(-1)
    relative_positions = x_j[..., :2] - x_i[..., :2]
    node_distance, delta_theta = polar_coordinates.cart2polar(relative_positions)
    delta_theta = wrap_angles(delta_theta, normalize=True)
    velocities = x_j[..., 2:]

    edge_attr = torch.cat(
        [
         relative_positions,
         delta_yaw,
         node_distance,
         delta_theta,
         # torch.atan2(x_j[..., 3] - x_i[..., 3],
                     # x_j[..., 2] - x_i[..., 2]).unsqueeze(-1) / np.pi,
         velocities,
        ], -1)
    return edge_attr


def create_edge_attr_pos_vel(x, send_edges, recv_edges, batched=False):
    x_j, x_i = sender_receiver_features(
        x, send_edges, recv_edges, batched=batched)

    # recv_yaw is the yaw angle, approximated via the velocity vector
    recv_yaw = torch.atan2(x_i[..., 3], x_i[..., 2])
    r = rotation_matrix(-recv_yaw)

    # delta_yaw is the signed difference in yaws
    delta_yaw = angle_diff(x_i[..., 2:], x_j[..., 2:]).unsqueeze(-1)
    rotated_relative_positions = (r @ (x_j[..., :2] - x_i[..., :2]).unsqueeze(-1)).squeeze(-1)
    node_distance = torch.norm(x_j[..., :2] - x_i[..., :2], dim=-1, keepdim=True)
    # delta_theta is the rotated azimuth. Subtracting the receiving yaw angle
    # is equal to a rotation
    delta_theta = (
        torch.atan2(x_j[..., 1] - x_i[..., 1], x_j[..., 0] - x_i[..., 0])
        - recv_yaw).unsqueeze(-1)
    delta_theta = wrap_angles(delta_theta, normalize=True)

    rotated_velocities = (r @ x_j[..., 2:].unsqueeze(-1)).squeeze(-1)

    edge_attr = torch.cat(
        [
         rotated_relative_positions,
         delta_yaw,
         node_distance,
         delta_theta,
         # torch.norm(send_embed[..., 2:] - recv_embed[..., 2:], dim=-1,
                    # keepdim=True),
         # (torch.atan2(x_j[..., 3] - x_i[..., 3],
                      # x_j[..., 2] - x_i[..., 2])
         # - recv_yaw).unsqueeze(-1) / np.pi,
         rotated_velocities,
        ], -1)
    return edge_attr


def create_trans_3d_edge_attr_pos_vel(x, send_edges, recv_edges, batched=False):
    send_embed, recv_embed = sender_receiver_features(
        x, send_edges, recv_edges, batched=batched)

    _, send_yaw, send_pitch = spherical_coordinates.cart2spherical(send_embed[..., 3:])

    node_distance, _, _ = spherical_coordinates.cart2spherical(send_embed[..., :3] - recv_embed[..., :3])

    send_r = spherical_coordinates.R_sph(send_yaw, send_pitch)
    euler_angles = matrix_to_euler_angles(send_r, 'ZYX')

    relative_positions = send_embed[..., :3] - recv_embed[..., :3]
    velocities = send_embed[..., 3:]
    # Theta: azimuth, phi: elevation
    _, delta_theta, delta_phi = spherical_coordinates.cart2spherical(relative_positions)

    edge_attr = torch.cat(
        [
         relative_positions,
         euler_angles,
         node_distance,
         delta_theta,
         delta_phi,
         velocities,
        ], -1)
    return edge_attr


def create_3d_edge_attr_pos_vel(x, send_edges, recv_edges, batched=False):
    send_embed, recv_embed = sender_receiver_features(
        x, send_edges, recv_edges, batched=batched)
    # r = kornia.angle_axis_to_rotation_matrix(recv_embed[..., 3:].view(-1, 3)).reshape(*recv_embed.shape[:-1], 3, 3)

    _, send_yaw, send_pitch = spherical_coordinates.cart2spherical(send_embed[..., 3:])
    _, recv_yaw, recv_pitch = spherical_coordinates.cart2spherical(recv_embed[..., 3:])
    r = spherical_coordinates.R_sph(recv_yaw, recv_pitch)

    node_distance, _, _ = spherical_coordinates.cart2spherical(send_embed[..., :3] - recv_embed[..., :3])

    send_r = spherical_coordinates.R_sph(send_yaw, send_pitch)
    rotated_euler = matrix_to_euler_angles(r @ send_r, 'ZYX')

    rotated_relative_positions = (r @ (send_embed[..., :3] - recv_embed[..., :3]).unsqueeze(-1)).squeeze(-1)
    rotated_velocities = (r @ send_embed[..., 3:].unsqueeze(-1)).squeeze(-1)
    # Theta: azimuth, phi: elevation
    _, delta_theta, delta_phi = spherical_coordinates.cart2spherical(rotated_relative_positions)

    edge_attr = torch.cat(
        [
         rotated_relative_positions,
         rotated_euler,
         node_distance,
         delta_theta,
         delta_phi,
         rotated_velocities,
        ], -1)
    return edge_attr


def angle_diff(v1, v2):
    # x1 = v1[..., 0]
    # y1 = v1[..., 1]
    # x2 = v2[..., 0]
    # y2 = v2[..., 1]
    # return torch.atan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2)
    delta_angle = (torch.atan2(v2[..., 1], v2[..., 0])
                   - torch.atan2(v1[..., 1], v1[..., 0]))
    delta_angle[delta_angle >= np.pi] -= 2 * np.pi
    delta_angle[delta_angle < -np.pi] += 2 * np.pi
    delta_angle = delta_angle / np.pi
    return delta_angle
