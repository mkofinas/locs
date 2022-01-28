import math

import numpy as np
import torch
from torch import nn
from torch_scatter import scatter_sum, scatter_mean

from locs.models import activations


class EGNNDynamicVars(nn.Module):
    def __init__(self, params):
        super().__init__()
        # Model Params
        self.decoder = DynamicVarsDecoder(params)
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
        self.no_prior = params.get('no_prior', False)
        self.avg_prior = params.get('avg_prior', False)
        self.learned_prior = params.get('use_learned_prior', False)
        self.anneal_teacher_forcing = params.get('anneal_teacher_forcing', False)
        self.teacher_forcing_prior = params.get('teacher_forcing_prior', False)
        self.steps = 0

    def get_graph_info(self, masks):
        num_vars = masks.size(-1)
        edges = torch.ones(num_vars, device=masks.device) - torch.eye(num_vars, device=masks.device)
        tmp = torch.where(edges)
        send_edges = tmp[0]
        recv_edges = tmp[1]
        tmp_inds = torch.tensor(list(range(num_vars)), device=masks.device, dtype=torch.long).unsqueeze_(1) #TODO: should initialize as long
        edge2node_inds = (tmp_inds == recv_edges.unsqueeze(0)).nonzero()[:, 1].contiguous().view(-1, num_vars-1)
        edge_masks = masks[:, :, send_edges]*masks[:, :, recv_edges] #TODO: gotta figure this one out still
        return send_edges, recv_edges, edge2node_inds, edge_masks

    def single_step_forward(self, inputs, node_masks, graph_info, decoder_hidden, edge_logits, hard_sample):
        edges = None
        predictions, decoder_hidden = self.decoder(inputs, decoder_hidden, edges, node_masks, graph_info)
        return predictions, decoder_hidden, edges

    #@profile
    def calculate_loss(self, inputs, node_masks, node_inds, graph_info, is_train=False, teacher_forcing=True, return_edges=False, return_logits=False, use_prior_logits=False, normalized_inputs=None):
        # node_masks.nonzero()[:, -1]
        self.decoder.reset_hidden_state(inputs)
        decoder_hidden = None
        num_time_steps = inputs.size(1)
        all_edges = []
        all_predictions = []
        all_priors = []
        hard_sample = (not is_train) or self.train_hard_sample
        if self.anneal_teacher_forcing:
            teacher_forcing_steps = math.ceil((1 - self.train_percent)*num_time_steps)
        else:
            teacher_forcing_steps = self.teacher_forcing_steps
        edge_ind = 0
        for step in range(num_time_steps-1):
            if (teacher_forcing and (teacher_forcing_steps == -1 or step < teacher_forcing_steps)) or step == 0:
                current_inputs = inputs[:, step]
            else:
                current_inputs = predictions
            current_node_masks = node_masks[:, step]
            node_inds = current_node_masks.nonzero()[:, -1]
            num_edges = len(node_inds)*(len(node_inds)-1)
            current_graph_info = graph_info[0][step]
            current_p_logits = None
            edge_ind += num_edges
            predictions, decoder_hidden, edges = self.single_step_forward(current_inputs, current_node_masks, current_graph_info, decoder_hidden, current_p_logits, hard_sample)
            all_predictions.append(predictions)
            all_edges.append(edges)
        all_predictions = torch.stack(all_predictions, dim=1)
        target = inputs[:, 1:, :, :]
        target_masks = ((node_masks[:, :-1] == 1)*(node_masks[:, 1:] == 1)).float()
        loss_nll = self.nll(all_predictions, target, target_masks)
        loss_kl = torch.FloatTensor([0.0]).cuda()
        loss = loss_nll + self.kl_coef*loss_kl
        loss = loss.mean()
        if return_edges:
            return loss, loss_nll, loss_kl, edges
        elif return_logits:
            return loss, loss_nll, loss_kl, None, all_predictions
        else:
            return loss, loss_nll, loss_kl

    def get_prior_posterior(self, inputs, student_force=False, burn_in_steps=None):
        self.eval()
        posterior_logits = self.encoder(inputs)
        posterior_probs = torch.softmax(posterior_logits, dim=-1)
        prior_hidden = self.prior_model.get_initial_hidden(inputs)
        all_logits = []
        if student_force:
            decoder_hidden = self.decoder.get_initial_hidden(inputs)
            for step in range(burn_in_steps):
                current_inputs= inputs[:, step]
                predictions, prior_hidden, decoder_hidden, _, prior_logits = self.single_step_forward(current_inputs, prior_hidden, decoder_hidden, None, True)
                all_logits.append(prior_logits)
            for step in range(inputs.size(1) - burn_in_steps):
                predictions, prior_hidden, decoder_hidden, _, prior_logits = self.single_step_forward(predictions, prior_hidden, decoder_hidden, None, True)
                all_logits.append(prior_logits)
        else:
            for step in range(inputs.size(1)):
                current_inputs = inputs[:, step]
                prior_logits, prior_hidden = self.prior_model(prior_hidden, current_inputs)
                all_logits.append(prior_logits)
        logits = torch.stack(all_logits, dim=1)
        prior_probs = torch.softmax(logits, dim=-1)
        return prior_probs, posterior_probs

    def get_edge_probs(self, inputs):
        self.eval()
        prior_hidden = self.prior_model.get_initial_hidden(inputs)
        all_logits = []
        for step in range(inputs.size(1)):
            current_inputs = inputs[:, step]
            prior_logits, prior_hidden = self.prior_model(prior_hidden, current_inputs)
            all_logits.append(prior_logits)
        logits = torch.stack(all_logits, dim=1)
        edge_probs = torch.softmax(logits, dim=-1)
        return edge_probs

    def predict_future(self, inputs, masks, node_inds, graph_info, burn_in_masks):
        '''
        Here, we assume the following:
        * inputs contains all of the gt inputs, including for the time steps we're predicting
        * masks keeps track of the variables that are being tracked
        * burn_in_masks is set to 1 whenever we're supposed to feed in that variable's state
          for a given time step
        '''
        total_timesteps = inputs.size(1)
        self.decoder.reset_hidden_state(inputs)
        decoder_hidden = None
        predictions = inputs[:, 0]
        preds = []
        for step in range(total_timesteps-1):
            current_masks = masks[:, step]
            current_burn_in_masks = burn_in_masks[:, step].unsqueeze(-1).type(inputs.dtype)
            current_inps = inputs[:, step]
            current_node_inds = node_inds[0][step] #TODO: check what's passed in here
            current_graph_info = graph_info[0][step]
            encoder_inp = current_burn_in_masks*current_inps + (1-current_burn_in_masks)*predictions
            current_edge_logits = None
            predictions, decoder_hidden, _ = self.single_step_forward(encoder_inp, current_masks, current_graph_info, decoder_hidden, current_edge_logits, True)
            preds.append(predictions)
        return torch.stack(preds, dim=1)

    def copy_states(self, prior_state, decoder_state):
        if isinstance(prior_state, tuple) or isinstance(prior_state, list):
            current_prior_state = (prior_state[0].clone(), prior_state[1].clone())
        else:
            current_prior_state = prior_state.clone()
        if isinstance(decoder_state, tuple) or isinstance(decoder_state, list):
            current_decoder_state = (decoder_state[0].clone(), decoder_state[1].clone())
        else:
            current_decoder_state = decoder_state.clone()
        return current_prior_state, current_decoder_state

    def merge_hidden(self, hidden):
        if isinstance(hidden[0], tuple) or isinstance(hidden[0], list):
            result0 = torch.cat([x[0] for x in hidden], dim=0)
            result1 = torch.cat([x[1] for x in hidden], dim=0)
            return (result0, result1)
        else:
            return torch.cat(hidden, dim=0)

    def predict_future_fixedwindow(self, inputs, burn_in_steps, prediction_steps, batch_size):
        if self.fix_encoder_alignment:
            prior_logits, _, prior_hidden = self.encoder(inputs)
        else:
            prior_logits, _, prior_hidden = self.encoder(inputs[:, :-1])
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        for step in range(burn_in_steps-1):
            current_inputs = inputs[:, step]
            current_edge_logits = prior_logits[:, step]
            predictions, decoder_hidden, _ = self.single_step_forward(current_inputs, decoder_hidden, current_edge_logits, True)
        all_timestep_preds = []
        for window_ind in range(burn_in_steps - 1, inputs.size(1)-1, batch_size):
            current_batch_preds = []
            prior_states = []
            decoder_states = []
            for step in range(batch_size):
                if window_ind + step >= inputs.size(1):
                    break
                predictions = inputs[:, window_ind + step]
                current_edge_logits, prior_hidden = self.encoder.single_step_forward(predictions, prior_hidden)
                predictions, decoder_hidden, _ = self.single_step_forward(predictions, decoder_hidden, current_edge_logits, True)
                current_batch_preds.append(predictions)
                tmp_prior, tmp_decoder = self.copy_states(prior_hidden, decoder_hidden)
                prior_states.append(tmp_prior)
                decoder_states.append(tmp_decoder)
            batch_prior_hidden = self.merge_hidden(prior_states)
            batch_decoder_hidden = self.merge_hidden(decoder_states)
            current_batch_preds = torch.cat(current_batch_preds, 0)
            current_timestep_preds = [current_batch_preds]
            for step in range(prediction_steps - 1):
                current_batch_edge_logits, batch_prior_hidden = self.encoder.single_step_forward(current_batch_preds, batch_prior_hidden)
                current_batch_preds, batch_decoder_hidden, _ = self.single_step_forward(current_batch_preds, batch_decoder_hidden, current_batch_edge_logits, True)
                current_timestep_preds.append(current_batch_preds)
            all_timestep_preds.append(torch.stack(current_timestep_preds, dim=1))
        result =  torch.cat(all_timestep_preds, dim=0)
        return result.unsqueeze(0)

    def nll(self, preds, target, masks):
        if self.nll_loss_type == 'crossent':
            return self.nll_crossent(preds, target, masks)
        elif self.nll_loss_type == 'gaussian':
            return self.nll_gaussian(preds, target, masks)
        elif self.nll_loss_type == 'poisson':
            return self.nll_poisson(preds, target, masks)

    def nll_gaussian(self, preds, target, masks, add_const=False):
        neg_log_p = ((preds - target) ** 2 / (2 * self.prior_variance))*masks.unsqueeze(-1)
        const = 0.5 * np.log(2 * np.pi * self.prior_variance)
        #neg_log_p += const
        if self.normalize_nll_per_var:
            raise NotImplementedError()
        elif self.normalize_nll:
            return (neg_log_p.sum(-1) + const*masks).view(preds.size(0), -1).sum(dim=-1)/(masks.view(masks.size(0), -1).sum(dim=1)+1e-8)
        else:
            raise NotImplementedError()


    def nll_crossent(self, preds, target, masks):
        if self.normalize_nll:
            loss = nn.BCEWithLogitsLoss(reduction='none')(preds, target)
            return (loss*masks.unsqueeze(-1)).view(preds.size(0), -1).sum(dim=-1)/(masks.view(masks.size(0), -1).sum(dim=1))
        else:
            raise NotImplementedError()

    def nll_poisson(self, preds, target, masks):
        if self.normalize_nll:
            loss = nn.PoissonNLLLoss(reduction='none')(preds, target)
            return (loss*masks.unsqueeze(-1)).view(preds.size(0), -1).sum(dim=-1)/(masks.view(masks.size(0), -1).sum(dim=1))
        else:
            raise NotImplementedError()

    def kl_categorical_learned(self, preds, prior_logits):
        log_prior = nn.LogSoftmax(dim=-1)(prior_logits)
        kl_div = preds*(torch.log(preds + 1e-16) - log_prior)
        if self.normalize_kl:
            return kl_div.sum(-1).view(preds.size(0), -1).mean(dim=1)
        elif self.normalize_kl_per_var:
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class DynamicVarsDecoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        edge_types = params['num_edge_types']

        print('Using learned interaction net decoder.')

        self.num_layers = 4
        self.layers = EGNN_vel(markov=not (params['decoder_type'] ==
                                           'recurrent'))

        self.hidden_size = 64
        self.hidden_embedding = nn.Linear(1, self.hidden_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def reset_hidden_state(self, inputs):
      self.layers.reset_hidden_state(inputs)

    def forward(self, inputs, hidden, edges, node_masks, graph_info):
        # Input Size: [batch, num_vars, input_size]
        # Hidden Size: [batch, num_vars, rnn_hidden]
        # Edges size: [batch, current_num_edges, num_edge_types]

        max_num_vars = inputs.size(1)
        node_inds = node_masks.nonzero()[:, -1]

        # current_hidden = hidden[:, node_inds]
        current_inputs = inputs[:, node_inds]
        num_vars = current_inputs.size(1)

        if num_vars > 1:
            send_edges, recv_edges, edge2node_inds = graph_info
            send_edges, recv_edges, edge2node_inds = send_edges.cuda(non_blocking=True), recv_edges.cuda(non_blocking=True), edge2node_inds.cuda(non_blocking=True)

            pos_v, vel_v = torch.chunk(current_inputs, 2, dim=-1)

            hidden_v = torch.norm(vel_v, dim=-1, keepdim=True)

            hidden_v, pos_v, vel_v = self.layers(
                hidden_v, pos_v, (send_edges.cuda(),
                                  recv_edges.cuda()), vel_v)

            pred = torch.cat([pos_v, vel_v], -1)
        elif num_vars == 0:
            pred_all = torch.zeros(inputs.size(0), max_num_vars, inputs.size(-1), device=inputs.device)
            return pred_all, hidden
        else:
            agg_msgs = torch.zeros(current_inputs.size(0), num_vars, self.msg_out_shape, device=inputs.device)

        # hidden = hidden.clone()
        # hidden[:, node_inds] = current_hidden
        pred_all = torch.zeros(inputs.size(0), max_num_vars, inputs.size(-1), device=inputs.device)
        pred_all[0, node_inds] = pred

        return pred_all, None


class EGNN_vel(nn.Module):
    def __init__(self, in_node_nf=1, in_edge_nf=2, hidden_nf=64, device='cpu',
                 act_fn=activations.SiLU(), n_layers=4, coords_weight=1.0,
                 recurrent=True, norm_diff=False, tanh=False, markov=True):
        super(EGNN_vel, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.markov = markov
        print('markov', markov)
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
            import IPython; IPython.embed()
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
