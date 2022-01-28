"""
E(n) Equivariant Graph Neural Networks
Implementation copied and adapted from https://github.com/vgsatorras/egnn
"""

import numpy as np
import torch
from torch import nn
from torch_scatter import scatter_sum, scatter_mean

from locs.models import activations


class EGNN(nn.Module):
    def __init__(self, params):
        super().__init__()
        # Model Params
        self.num_vars = params['num_vars']
        self.decoder = Decoder(params)
        self.num_edge_types = params.get('num_edge_types')

        # Training params
        self.gumbel_temp = params.get('gumbel_temp')
        self.train_hard_sample = params.get('train_hard_sample')
        self.teacher_forcing_steps = params.get('teacher_forcing_steps', -1)

        self.normalize_kl = params.get('normalize_kl', False)
        self.normalize_kl_per_var = params.get('normalize_kl_per_var', False)
        self.normalize_nll = params.get('normalize_nll', False)
        self.normalize_nll_per_var = params.get('normalize_nll_per_var', False)
        self.kl_coef = params.get('kl_coef', 1.)
        self.nll_loss_type = params.get('nll_loss_type', 'crossent')
        self.prior_variance = params.get('prior_variance')
        self.timesteps = params.get('timesteps', 0)
        self.burn_in_steps = params.get('train_burn_in_steps')
        self.teacher_forcing_prior = params.get('teacher_forcing_prior', False)
        self.val_teacher_forcing_steps = params.get('val_teacher_forcing_steps', -1)
        self.add_uniform_prior = params.get('add_uniform_prior')
        if self.add_uniform_prior:
            if params.get('no_edge_prior') is not None:
                prior = np.zeros(self.num_edge_types)
                prior.fill((1 - params['no_edge_prior'])/(self.num_edge_types - 1))
                prior[0] = params['no_edge_prior']
                log_prior = torch.FloatTensor(np.log(prior))
                log_prior = torch.unsqueeze(log_prior, 0)
                log_prior = torch.unsqueeze(log_prior, 0)
                if params['gpu']:
                    log_prior = log_prior.cuda(non_blocking=True)
                self.log_prior = log_prior
                print("USING NO EDGE PRIOR: ",self.log_prior)
            else:
                print("USING UNIFORM PRIOR")
                prior = np.zeros(self.num_edge_types)
                prior.fill(1.0/self.num_edge_types)
                log_prior = torch.FloatTensor(np.log(prior))
                log_prior = torch.unsqueeze(log_prior, 0)
                log_prior = torch.unsqueeze(log_prior, 0)
                if params['gpu']:
                    log_prior = log_prior.cuda(non_blocking=True)
                self.log_prior = log_prior

    def single_step_forward(self, inputs):
        predictions = self.decoder(inputs)
        return predictions

    def calculate_loss(self, inputs, is_train=False, teacher_forcing=True, return_logits=False):
        num_time_steps = inputs.size(1)
        all_predictions = []
        self.decoder.reset_hidden_state(inputs)
        if not is_train:
            teacher_forcing_steps = self.val_teacher_forcing_steps
        else:
            teacher_forcing_steps = self.teacher_forcing_steps
        for step in range(num_time_steps-1):
            if (teacher_forcing and (teacher_forcing_steps == -1 or step < teacher_forcing_steps)) or step == 0:
                current_inputs = inputs[:, step]
            else:
                current_inputs = predictions
            predictions = self.single_step_forward(current_inputs)
            all_predictions.append(predictions)
        all_predictions = torch.stack(all_predictions, dim=1)
        target = inputs[:, 1:, :, :]
        loss_nll = self.nll(all_predictions, target)
        loss_kl = torch.FloatTensor([0.0])
        loss = loss_nll
        loss = loss.mean()

        if return_logits:
            return loss, loss_nll, loss_kl, None, all_predictions
        else:
            return loss, loss_nll, loss_kl

    def predict_future(self, inputs, prediction_steps, return_everything=False):
        burn_in_timesteps = inputs.size(1)
        all_predictions = []

        self.decoder.reset_hidden_state(inputs)
        for step in range(burn_in_timesteps-1):
            current_inputs = inputs[:, step]
            predictions = self.single_step_forward(current_inputs)
            if return_everything:
                all_predictions.append(predictions)
        predictions = inputs[:, burn_in_timesteps-1]
        for step in range(prediction_steps):
            predictions = self.single_step_forward(predictions)
            all_predictions.append(predictions)

        predictions = torch.stack(all_predictions, dim=1)
        return predictions

    def nll(self, preds, target):
        if self.nll_loss_type == 'crossent':
            return self.nll_crossent(preds, target)
        elif self.nll_loss_type == 'gaussian':
            return self.nll_gaussian(preds, target)
        elif self.nll_loss_type == 'poisson':
            return self.nll_poisson(preds, target)

    def nll_gaussian(self, preds, target, add_const=False):
        neg_log_p = ((preds - target) ** 2 / (2 * self.prior_variance))
        const = 0.5 * np.log(2 * np.pi * self.prior_variance)
        #neg_log_p += const
        if self.normalize_nll_per_var:
            return neg_log_p.sum() / (target.size(0) * target.size(2))
        elif self.normalize_nll:
            return (neg_log_p.sum(-1) + const).view(preds.size(0), -1).mean(dim=1)
        else:
            return neg_log_p.view(target.size(0), -1).sum() / (target.size(1))

    def nll_crossent(self, preds, target):
        if self.normalize_nll:
            return nn.BCEWithLogitsLoss(reduction='none')(preds, target).view(preds.size(0), -1).mean(dim=1)
        else:
            return nn.BCEWithLogitsLoss(reduction='none')(preds, target).view(preds.size(0), -1).sum(dim=1)

    def nll_poisson(self, preds, target):
        if self.normalize_nll:
            return nn.PoissonNLLLoss(reduction='none')(preds, target).view(preds.size(0), -1).mean(dim=1)
        else:
            return nn.PoissonNLLLoss(reduction='none')(preds, target).view(preds.size(0), -1).sum(dim=1)

    def kl_categorical_learned(self, preds, prior_logits):
        log_prior = nn.LogSoftmax(dim=-1)(prior_logits)
        kl_div = preds*(torch.log(preds + 1e-16) - log_prior)
        if self.normalize_kl:
            return kl_div.sum(-1).view(preds.size(0), -1).mean(dim=1)
        elif self.normalize_kl_per_var:
            return kl_div.sum() / (self.num_vars * preds.size(0))
        else:
            return kl_div.view(preds.size(0), -1).sum(dim=1)

    def kl_categorical_avg(self, preds, eps=1e-16):
        avg_preds = preds.mean(dim=2)
        kl_div = avg_preds*(torch.log(avg_preds+eps) - self.log_prior)
        if self.normalize_kl:
            return kl_div.sum(-1).view(preds.size(0), -1).mean(dim=1)
        elif self.normalize_kl_per_var:
            return kl_div.sum() / (self.num_vars * preds.size(0))
        else:
            return kl_div.view(preds.size(0), -1).sum(dim=1)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        num_vars = params['num_vars']
        edge_types = params['num_edge_types']

        self.num_vars = num_vars
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]

        self.num_layers = 4
        self.layers = EGNN_vel(markov=not (params['decoder_type'] ==
                                           'recurrent'))

        self.hidden_size = 64
        self.hidden_embedding = nn.Linear(1, self.hidden_size)

    def reset_hidden_state(self, inputs):
        self.layers.reset_hidden_state(inputs)

    def forward(self, inputs):
        # single_timestep_inputs has shape
        # [batch_size, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_atoms*(num_atoms-1), num_edge_types]
        # Node2edge
        # The suffix _v refers to vertex features
        # The suffices _send and _recv refer to sender and receiver edge
        # features, respectively
        pos_v, vel_v = torch.chunk(inputs, 2, dim=-1)

        hidden_v = torch.norm(vel_v, dim=-1, keepdim=True)

        hidden_v, pos_v, vel_v = self.layers(
            hidden_v, pos_v, (torch.from_numpy(self.send_edges).cuda(),
                              torch.from_numpy(self.recv_edges).cuda()), vel_v)

        outputs = torch.cat([pos_v, vel_v], -1)
        # Predict position/velocity difference
        return outputs


class EGNN_vel(nn.Module):
    def __init__(self, in_node_nf=1, in_edge_nf=2, hidden_nf=64, device='cpu',
                 act_fn=activations.SiLU(), n_layers=4, coords_weight=1.0,
                 recurrent=True, norm_diff=False, tanh=False, markov=True):
        super(EGNN_vel, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.markov = markov
        print('Markov', markov)
        #self.reg = reg
        ### Encoder
        #self.add_module("gcl_0", E_GCL(in_node_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i, E_GCL_vel(self.hidden_nf, self.hidden_nf,
                                        self.hidden_nf, edges_in_d=0,
                                        act_fn=act_fn,
                                        coords_weight=coords_weight,
                                        recurrent=recurrent,
                                        norm_diff=norm_diff, tanh=tanh,
                                        markov=markov))
        self.to(self.device)

    def reset_hidden_state(self, inputs):
        if not self.markov:
            for i in range(0, self.n_layers):
                self._modules["gcl_%d" % i].reset_hidden_state(inputs)


    def forward(self, h, x, edges, vel):
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, v = self._modules["gcl_%d" % i](h, edges, x, vel)
        return h, x, v


class E_GCL_vel(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0,
                 nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True,
                 coords_weight=1.0, attention=False, clamp=False,
                 norm_diff=False, tanh=False, markov=True):
        super().__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1
        self.hidden_dim = hidden_nf

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)


        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

        self.markov = markov
        if not markov:
            self.gru = nn.GRUCell(hidden_nf, hidden_nf)
        self.norm_diff = norm_diff
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1))

    def reset_hidden_state(self, inputs):
        print(inputs.shape)
        self.gru_hidden = torch.zeros(inputs.size(0) * inputs.size(2), self.hidden_dim, device=inputs.device)

    def edge_model(self, source, target, radial):
        out = torch.cat([source, target, radial], dim=-1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg = scatter_sum(edge_attr, col, dim=1, dim_size=x.size(1))
        agg = torch.cat([x, agg], dim=-1)
        out = self.node_mlp(agg)
        if not self.markov:
            pre_shape = out.shape
            self.gru_hidden = self.gru(out.reshape(-1, out.shape[-1]), self.gru_hidden)
            out = self.gru_hidden.reshape(pre_shape)
        if self.recurrent:
            out = out + x
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        agg = scatter_mean(trans, col, dim=1, dim_size=coord.size(1))
        new_vel = agg*self.coords_weight
        return new_vel


    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[:, row] - coord[:, col]
        radial = torch.sum((coord_diff)**2, -1, keepdim=True)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff/(norm)

        return radial, coord_diff

    def forward(self, h, edge_index, coord, vel):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[:, row], h[:, col], radial)
        new_vel = self.coord_model(coord, edge_index, coord_diff, edge_feat)

        new_vel = new_vel + self.coord_mlp_vel(h) * vel
        new_coord = coord + new_vel
        h, agg = self.node_model(h, edge_index, edge_feat)
        return h, new_coord, new_vel
