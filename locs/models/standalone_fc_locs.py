import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


"""
Geometric primitives, rotations, cartesian to spherical
"""


def rotation_matrix(ndim, theta, phi=None, psi=None, /):
    """
    theta, phi, psi: yaw, pitch, roll

    NOTE: We assume that each angle is has the shape [dims] x 1
    """
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    if ndim == 2:
        R = torch.stack([torch.cat([cos_theta, -sin_theta], -1),
                         torch.cat([sin_theta, cos_theta], -1)], -2)
        return R
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    R = torch.stack([
            torch.cat([cos_phi * cos_theta, -sin_theta, sin_phi * cos_theta], -1),
            torch.cat([cos_phi * sin_theta, cos_theta, sin_phi * sin_theta], -1),
            torch.cat([-sin_phi, torch.zeros_like(cos_theta), cos_phi], -1)], -2)
    return R


def cart_to_n_spherical(x, symmetric_theta=False):
    """Transform Cartesian to n-Spherical Coordinates

    NOTE: Not tested thoroughly for n > 3

    Math convention, theta: azimuth angle, angle in x-y plane

    x: torch.Tensor, [dims] x D
    return rho, theta, phi
    """
    ndim = x.size(-1)

    rho = torch.norm(x, p=2, dim=-1, keepdim=True)

    theta = torch.atan2(x[..., [1]], x[..., [0]])
    if not symmetric_theta:
        theta = theta + (theta < 0).type_as(theta) * (2 * np.pi)

    if ndim == 2:
        return rho, theta

    cum_sqr = (rho if ndim == 3
               else torch.sqrt(torch.cumsum(torch.flip(x ** 2, [-1]), dim=-1))[..., 2:])
    EPS = 1e-7
    phi = torch.acos(
        torch.clamp(x[..., 2:] / (cum_sqr + EPS), min=-1.0, max=1.0)
    )

    return rho, theta, phi


def velocity_to_rotation_matrix(vel):
    num_dims = vel.size(-1)
    orientations = cart_to_n_spherical(vel)[1:]
    R = rotation_matrix(num_dims, *orientations)
    return R


def rotation_matrix_to_euler(R, num_dims, normalize=True):
    """Convert rotation matrix to euler angles

    In 3 dimensions, we follow the ZYX convention
    """
    if num_dims == 2:
        euler = torch.atan2(R[..., 1, [0]], R[..., 0, [0]])
    else:
        euler = torch.stack([
            torch.atan2(R[..., 1, 0], R[..., 0, 0]),
            torch.asin(-R[..., 2, 0]),
            torch.atan2(R[..., 2, 1], R[..., 2, 2]),
        ], -1)

    if normalize:
        euler = euler / np.pi
    return euler


def rotate(x, R):
    return torch.einsum('...ij,...j->...i', R, x)


"""
Transformation from global to local coordinates and vice versa
"""


class Localizer(nn.Module):
    def __init__(self, num_dims: int = 2):
        super().__init__()
        self.num_dims = num_dims

    def set_edge_index(self, send_edges, recv_edges):
        self.send_edges = send_edges
        self.recv_edges = recv_edges

    def sender_receiver_features(self, x):
        batch_range = torch.arange(x.size(0), device=x.device).unsqueeze(-1)
        x_j = x[batch_range, self.send_edges]
        x_i = x[batch_range, self.recv_edges]
        return x_j, x_i

    def canonicalize_inputs(self, inputs):
        vel = inputs[..., self.num_dims:2*self.num_dims]
        R = velocity_to_rotation_matrix(vel)
        Rinv = R.transpose(-1, -2)

        canon_vel = rotate(vel, Rinv)
        canon_inputs = torch.cat([torch.zeros_like(canon_vel), canon_vel], dim=-1)

        return canon_inputs, R

    def create_edge_attr(self, x):
        x_j, x_i = self.sender_receiver_features(x)

        # We approximate orientations via the velocity vector
        R = velocity_to_rotation_matrix(x_i[..., self.num_dims:2*self.num_dims])
        R_inv = R.transpose(-1, -2)

        # Positions
        relative_positions = x_j[..., :self.num_dims] - x_i[..., :self.num_dims]
        rotated_relative_positions = rotate(relative_positions, R_inv)

        # Orientations
        send_R = velocity_to_rotation_matrix(x_j[..., self.num_dims:2*self.num_dims])
        rotated_orientations = R_inv @ send_R
        rotated_euler = rotation_matrix_to_euler(rotated_orientations, self.num_dims)

        # Rotated relative positions in spherical coordinates
        node_distance = torch.norm(relative_positions, p=2, dim=-1, keepdim=True)
        spherical_relative_positions = torch.cat(
            cart_to_n_spherical(rotated_relative_positions, symmetric_theta=True)[1:], -1)

        # Velocities
        rotated_velocities = rotate(x_j[..., self.num_dims:2*self.num_dims], R_inv)

        edge_attr = torch.cat([
            rotated_relative_positions,
            rotated_euler,
            node_distance,
            spherical_relative_positions,
            rotated_velocities,
        ], -1)
        return edge_attr

    def forward(self, x, edges):
        self.set_edge_index(*edges)
        rel_feat, R = self.canonicalize_inputs(x)
        edge_attr = self.create_edge_attr(x)

        batch_range = torch.arange(x.size(0), device=x.device).unsqueeze(-1)
        edge_attr = torch.cat([edge_attr, rel_feat[batch_range, self.recv_edges]], -1)
        return rel_feat, R, edge_attr


class Globalizer(nn.Module):
    def __init__(self, num_dims: int = 2):
        super().__init__()
        self.num_dims = num_dims

    def forward(self, x, R):
        return torch.cat(
            [rotate(xi, R) for xi in x.split(self.num_dims, dim=-1)], -1)


"""
LoCS Neural Network
"""


class FullyConnectedLoCS(nn.Module):
    def __init__(self, params):
        super().__init__()
        # Model Params
        self.network = (MarkovNetwork(params)
                        if params.get('network_type', None) == 'markov'
                        else RecurrentNetwork(params))

        # Training params
        self.prior_variance = params.get('prior_variance')
        self.kl_coef = 1.0

        self.num_objects = params['num_vars']
        self.edges = torch.where(~torch.eye(self.num_objects, dtype=bool))

    def calculate_loss(self, inputs, return_logits=False):
        network_hidden = self.network.get_initial_hidden(inputs)
        num_time_steps = inputs.size(1)
        all_predictions = []

        # We train using teacher forcing
        for step in range(num_time_steps-1):
            current_inputs = inputs[:, step]
            predictions, network_hidden = self.network(
                current_inputs, self.edges, network_hidden)
            all_predictions.append(predictions)
        all_predictions = torch.stack(all_predictions, dim=1)
        target = inputs[:, 1:, :, :]
        loss_nll = self.nll(all_predictions, target)
        loss = loss_nll.mean()

        if return_logits:
            return loss, loss_nll, torch.FloatTensor([0.0]), None, all_predictions
        else:
            return loss, loss_nll, torch.FloatTensor([0.0])

    def predict_future(self, inputs, prediction_steps, return_everything=False):
        burn_in_timesteps = inputs.size(1)
        network_hidden = self.network.get_initial_hidden(inputs)
        all_predictions = []
        for step in range(burn_in_timesteps-1):
            current_inputs = inputs[:, step]
            predictions, network_hidden = self.network(
                current_inputs, self.edges, network_hidden)
            if return_everything:
                all_predictions.append(predictions)
        predictions = inputs[:, burn_in_timesteps-1]
        for step in range(prediction_steps):
            predictions, network_hidden = self.network(
                predictions, self.edges, network_hidden)
            all_predictions.append(predictions)

        predictions = torch.stack(all_predictions, dim=1)
        return predictions

    def nll(self, preds, target):
        return self.nll_gaussian(preds, target, self.prior_variance)

    @staticmethod
    def nll_gaussian(preds, target, variance):
        neg_log_p = ((preds - target) ** 2 / (2 * variance))
        const = 0.5 * np.log(2 * np.pi * variance)
        return (neg_log_p.sum(-1) + const).view(preds.size(0), -1).mean(dim=1)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class MarkovNetwork(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.num_dims = 3 if params.get('use_3d', False) else 2
        self.gnn = GNN(params['input_size'], params['decoder_hidden'],
                       params['decoder_dropout'], self.num_dims)
        self.localizer = Localizer(params['num_dims'])
        self.globalizer = Globalizer(params['num_dims'])

    def get_initial_hidden(self, inputs):
        return None

    def _forward(self, inputs, edges):
        """inputs shape: [batch_size, num_objects, input_size]"""
        # Global to Local
        rel_feat, Rinv, edge_attr = self.localizer(inputs, edges)

        # GNN
        pred = self.gnn(rel_feat, edge_attr, edges)

        # Local to Global
        pred = self.globalizer(pred, Rinv)

        # Predict position/velocity difference and integrate
        outputs = inputs + pred
        return outputs

    def forward(self, inputs, edges, hidden):
        outputs = self._forward(inputs, edges)
        return outputs, None


class GNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob, num_dims):
        super().__init__()
        self.num_dims = num_dims
        out_size = input_size

        self.num_orientations = self.num_dims * (self.num_dims - 1) // 2
        # Relative features include: positions, orientations, positions in
        # spherical coordinates, and velocities
        self.num_relative_features = 3 * self.num_dims + self.num_orientations

        num_edge_features = self.num_relative_features + input_size

        # Neural Network Layers
        self.edge_filter = nn.Sequential(
            nn.Linear(num_edge_features, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.res1 = nn.Linear(input_size, hidden_size)

        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, inputs, edge_attr, edges):
        """
        inputs shape: [batch_size, num_objects, input_size]
        """
        # Edge embeddings
        edge_attr = self.edge_filter(edge_attr)
        # Aggregate all msgs to receiver
        agg_msgs = scatter(
            edge_attr, edges[1].to(inputs.device), dim=1,
            reduce='mean').contiguous()

        # Skip connection
        aug_inputs = agg_msgs + self.res1(inputs)

        # Output MLP
        pred = self.out_mlp(aug_inputs)
        return pred


class RecurrentNetwork(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hidden_size = params['decoder_hidden']
        self.num_dims = 3 if params.get('use_3d', False) else 2
        self.gnn = RecurrentGNN(
            params['input_size'], self.hidden_size,
            params['decoder_dropout'], self.num_dims)

        self.localizer = Localizer(params['num_dims'])
        self.globalizer = Globalizer(params['num_dims'])

    def get_initial_hidden(self, inputs):
        return torch.zeros(inputs.size(0), inputs.size(2), self.hidden_size, device=inputs.device)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs, edges, hidden):
        """
        inputs size: [batch, num_objects, input_size]
        hidden size: [batch, num_objects, hidden_size]
        """
        rel_feat, Rinv, edge_attr = self.localizer(inputs, edges)

        pred = self.gnn(rel_feat, edge_attr, edges, hidden)

        pred = self.globalizer(pred, Rinv)

        outputs = inputs + pred
        return outputs, hidden


class RecurrentGNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob, num_dims):
        super().__init__()
        self.num_dims = num_dims
        out_size = input_size

        self.num_orientations = self.num_dims * (self.num_dims - 1) // 2
        # Relative features include: positions, orientations, positions in
        # spherical coordinates, and velocities
        self.num_relative_features = 3 * self.num_dims + self.num_orientations

        num_edge_features = self.num_relative_features + input_size

        self.msg_mlp = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.present_msg_mlp = nn.Sequential(
            nn.Linear(num_edge_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.res_mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.input_r = nn.Linear(input_size, hidden_size, bias=True)
        self.input_i = nn.Linear(input_size, hidden_size, bias=True)
        self.input_n = nn.Linear(input_size, hidden_size, bias=True)

        self.hidden_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_h = nn.Linear(hidden_size, hidden_size, bias=False)

        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, out_size),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs, edge_attr, edges, hidden):
        """
        inputs size: [batch, num_objects, input_size]
        edge_attr size: [batch, num_edges, num_edge_features]
        hidden size: [batch, num_objects, hidden_size]
        """
        send_edges, recv_edges = edges

        # node2edge
        receivers = hidden[:, recv_edges]
        senders = hidden[:, send_edges]

        # hidden_messages: [batch, num_edges, 2 * hidden_size]
        hidden_messages = torch.cat([receivers, senders], dim=-1)
        hidden_messages = self.msg_mlp(hidden_messages)
        hidden_node_emb = scatter(hidden_messages, recv_edges.cuda(), dim=1, reduce='mean').contiguous()

        # Present messages
        present_messages = self.present_msg_mlp(edge_attr)
        present_node_emb = scatter(present_messages, recv_edges.cuda(), dim=1, reduce='mean').contiguous()
        present_node_emb = self.res_mlp(inputs) + present_node_emb

        # GRU-style gated aggregation
        r = torch.sigmoid(self.input_r(present_node_emb) + self.hidden_r(hidden_node_emb))
        i = torch.sigmoid(self.input_i(present_node_emb) + self.hidden_i(hidden_node_emb))
        n = torch.tanh(self.input_n(present_node_emb) + r*self.hidden_h(hidden_node_emb))
        hidden = (1 - i) * n + i * hidden

        # Output MLP
        pred = self.out_mlp(hidden)
        return pred, hidden


"""
PyTorch Geometric LoCS implementation
"""


class PyGNetwork(MessagePassing):
    def __init__(self, input_size, hidden_size, dropout_prob, num_dims):
        super().__init__(aggr='mean')
        self.num_dims = num_dims
        out_size = input_size

        self.num_orientations = self.num_dims * (self.num_dims - 1) // 2
        # Relative features include: positions, orientations, positions in
        # spherical coordinates, and velocities
        self.num_relative_features = 3 * self.num_dims + self.num_orientations

        num_edge_features = self.num_relative_features + input_size

        # Neural Network Layers
        self.edge_filter = nn.Sequential(
            nn.Linear(num_edge_features, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.res1 = nn.Linear(input_size, hidden_size)

        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, out_size),
        )

        self.localizer = Localizer(num_dims)
        self.globalizer = Globalizer(num_dims)

    def get_initial_hidden(self, inputs):
        return None

    def _forward(self, inputs, edges):
        """
        inputs shape: [batch_size, num_objects, input_size]
        """
        # Global to Local
        rel_feat, Rinv, edge_attr = self.localizer(inputs)

        pred = self.propagate(edges, edge_attr=edge_attr)

        pred = pred + self.res1(rel_feat)

        # Output MLP
        pred = self.out_mlp(pred)

        # Local to Global
        pred = self.globalizer(pred, Rinv)

        # Predict position/velocity difference and integrate
        outputs = inputs + pred
        return outputs

    def message(self, edge_attr):
        edge_attr = self.edge_filter(edge_attr)
        return edge_attr

    def forward(self, inputs, edges, hidden):
        outputs = self._forward(inputs, edges)
        return outputs, None


"""
Functions to train and evaluate the model
"""


def set_random_seeds(seed_val):
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    random.seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(model, train_data, val_data, params):
    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1000)
    val_batch_size = params.get('val_batch_size', batch_size)
    num_epochs = params.get('num_epochs', 100)
    continue_training = params.get('continue_training', False)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_data_loader = DataLoader(val_data, batch_size=val_batch_size)

    model_params = [param for param in model.parameters() if param.requires_grad]
    opt = torch.optim.Adam(model_params, lr=params['lr'])

    working_dir = params['working_dir']
    best_path = os.path.join(working_dir, 'best_model')
    checkpoint_dir = os.path.join(working_dir, 'model_checkpoint')
    training_path = os.path.join(working_dir, 'training_checkpoint')
    if continue_training:
        print("RESUMING TRAINING")
        model.load(checkpoint_dir)
        train_params = torch.load(training_path)
        start_epoch = train_params['epoch']
        opt.load_state_dict(train_params['optimizer'])
        best_val_result = train_params['best_val_result']
        best_val_epoch = train_params['best_val_epoch']
        print("STARTING EPOCH: ", start_epoch)
    else:
        start_epoch = 1
        best_val_epoch = -1
        best_val_result = float('inf')

    set_random_seeds(1)
    for epoch in range(start_epoch, num_epochs+1):
        print("EPOCH", epoch)
        model.train()
        for batch in train_data_loader:
            inputs = batch['inputs']
            if gpu:
                inputs = inputs.cuda(non_blocking=True)
            loss, _ = model.calculate_loss(inputs)
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_data_loader:
                inputs = batch['inputs']
                if gpu:
                    inputs = inputs.cuda(non_blocking=True)
                loss, loss_nll = model.calculate_loss(inputs)
                total_loss += loss_nll.sum().item()
        total_loss /= len(val_data)

        if total_loss < best_val_result:
            best_val_epoch = epoch
            best_val_result = total_loss
            print("BEST VAL RESULT. SAVING MODEL...")
            model.save(best_path)
        model.save(checkpoint_dir)
        torch.save({
                    'epoch': epoch+1,
                    'optimizer': opt.state_dict(),
                    'best_val_result': best_val_result,
                    'best_val_epoch': best_val_epoch,
                   }, training_path)
        print(f"EPOCH {epoch} EVAL:")
        print(f"\tCURRENT VAL LOSS: {total_loss}")
        print(f"\tBEST VAL LOSS:    {best_val_result}")
        print(f"\tBEST VAL EPOCH:   {best_val_epoch}")


@torch.no_grad()
def evaluate(model, dataset, burn_in_steps, forward_pred_steps, params):
    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1000)
    data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=gpu)

    model.eval()
    total_se = 0
    for batch in data_loader:
        inputs = batch['inputs']
        model_inputs = inputs[:, :burn_in_steps]
        gt_predictions = inputs[:, burn_in_steps:burn_in_steps+forward_pred_steps]
        if gpu:
            model_inputs = model_inputs.cuda(non_blocking=True)
        model_preds = model.predict_future(model_inputs, forward_pred_steps).cpu()
        unnorm_model_preds = dataset.torch_unnormalize(model_preds)
        unnorm_gt_predictions = dataset.torch_unnormalize(gt_predictions)

        total_se += F.mse_loss(unnorm_model_preds, unnorm_gt_predictions, reduction='none').view(unnorm_model_preds.size(0), unnorm_model_preds.size(1), -1).mean(dim=-1).sum(dim=0)
    return total_se / len(dataset)
