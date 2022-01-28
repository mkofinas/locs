import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch_scatter import scatter

from locs.models import model_utils
from locs.models.model_utils import RefNRIMLP

from locs.models.global_to_local import Localizer
from locs.models.local_to_global import Globalizer
from locs.models.anisotropic_filter import AnisotropicEdgeFilter, MLPEdgeFilter


class LoCS(nn.Module):
    def __init__(self, params):
        super().__init__()
        # Model Params

        self.num_vars = params['num_vars']
        self.encoder = Encoder(params)
        decoder_type = params.get('decoder_type', None)
        if decoder_type == 'ref_mlp':
            self.decoder = MarkovDecoder(params)
        else:
            self.decoder = RecurrentDecoder(params)
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
        self.use_3d = params.get('use_3d', False)

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

    def single_step_forward(self, inputs, decoder_hidden, edge_logits,
                            hard_sample):
        old_shape = edge_logits.shape
        edges = model_utils.gumbel_softmax(
            edge_logits.reshape(-1, self.num_edge_types),
            tau=self.gumbel_temp,
            hard=hard_sample).view(old_shape)
        predictions, decoder_hidden = self.decoder(inputs, decoder_hidden, edges)
        return predictions, decoder_hidden, edges

    def calculate_loss(self, inputs, is_train=False, teacher_forcing=True, return_edges=False, return_logits=False, use_prior_logits=False):
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        num_time_steps = inputs.size(1)
        all_edges = []
        all_predictions = []
        all_priors = []
        hard_sample = (not is_train) or self.train_hard_sample
        prior_logits, posterior_logits, _ = self.encoder(inputs[:, :-1])
        if is_train:
            teacher_forcing_steps = self.teacher_forcing_steps
        else:
            teacher_forcing_steps = self.val_teacher_forcing_steps
        for step in range(num_time_steps-1):
            if (teacher_forcing and (teacher_forcing_steps == -1 or step < teacher_forcing_steps)) or step == 0:
                current_inputs = inputs[:, step]
            else:
                current_inputs = predictions
            if not use_prior_logits:
                current_p_logits = posterior_logits[:, step]
            else:
                current_p_logits = prior_logits[:, step]
            predictions, decoder_hidden, edges = self.single_step_forward(current_inputs, decoder_hidden, current_p_logits, hard_sample)
            all_predictions.append(predictions)
            all_edges.append(edges)
        all_predictions = torch.stack(all_predictions, dim=1)
        target = inputs[:, 1:, :, :]
        loss_nll = self.nll(all_predictions, target)
        prob = F.softmax(posterior_logits, dim=-1)
        loss_kl = self.kl_categorical_learned(prob, prior_logits)
        if self.add_uniform_prior:
            loss_kl = 0.5*loss_kl + 0.5*self.kl_categorical_avg(prob)
        # loss = loss_nll + self.kl_scheduler() * loss_kl
        loss = loss_nll + self.kl_coef*loss_kl
        loss = loss.mean()

        if return_edges:
            return loss, loss_nll, loss_kl, edges
        elif return_logits:
            return loss, loss_nll, loss_kl, posterior_logits, all_predictions
        else:
            return loss, loss_nll, loss_kl

    def predict_future(self, inputs, prediction_steps, return_edges=False, return_everything=False):
        burn_in_timesteps = inputs.size(1)
        decoder_hidden = self.decoder.get_initial_hidden(inputs)
        all_predictions = []
        all_edges = []
        prior_logits, _, prior_hidden = self.encoder(inputs[:, :-1])
        for step in range(burn_in_timesteps-1):
            current_inputs = inputs[:, step]
            current_edge_logits = prior_logits[:, step]
            predictions, decoder_hidden, edges = self.single_step_forward(current_inputs, decoder_hidden, current_edge_logits, True)
            if return_everything:
                all_edges.append(edges)
                all_predictions.append(predictions)
        predictions = inputs[:, burn_in_timesteps-1]
        for step in range(prediction_steps):
            current_edge_logits, prior_hidden = self.encoder.single_step_forward(predictions, prior_hidden)
            predictions, decoder_hidden, edges = self.single_step_forward(predictions, decoder_hidden, current_edge_logits, True)
            all_predictions.append(predictions)
            all_edges.append(edges)

        predictions = torch.stack(all_predictions, dim=1)
        if return_edges:
            edges = torch.stack(all_edges, dim=1)
            return predictions, edges
        else:
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


class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        num_vars = params['num_vars']
        self.num_edges = params['num_edge_types']
        self.sepaate_prior_encoder = params.get('separate_prior_encoder', False)
        no_bn = False
        dropout = params['encoder_dropout']
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.save_eval_memory = params.get('encoder_save_eval_memory', False)

        hidden_size = params['encoder_hidden']
        rnn_hidden_size = params['encoder_rnn_hidden']
        rnn_type = params['encoder_rnn_type']
        inp_size = params['input_size']
        self.mlp3 = RefNRIMLP(hidden_size, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.mlp4 = RefNRIMLP(hidden_size * 3, hidden_size, hidden_size, dropout, no_bn=no_bn)
        self.res1 = nn.Linear(inp_size, hidden_size)

        if rnn_hidden_size is None:
            rnn_hidden_size = hidden_size
        if rnn_type == 'lstm':
            self.forward_rnn = nn.LSTM(hidden_size, rnn_hidden_size, batch_first=True)
            self.reverse_rnn = nn.LSTM(hidden_size, rnn_hidden_size, batch_first=True)
        elif rnn_type == 'gru':
            self.forward_rnn = nn.GRU(hidden_size, rnn_hidden_size, batch_first=True)
            self.reverse_rnn = nn.GRU(hidden_size, rnn_hidden_size, batch_first=True)
        out_hidden_size = 2*rnn_hidden_size
        num_layers = params['encoder_mlp_num_layers']
        if num_layers == 1:
            self.encoder_fc_out = nn.Linear(out_hidden_size, self.num_edges)
        else:
            tmp_hidden_size = params['encoder_mlp_hidden']
            layers = [nn.Linear(out_hidden_size, tmp_hidden_size), nn.ELU(inplace=True)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(tmp_hidden_size, tmp_hidden_size))
                layers.append(nn.ELU(inplace=True))
            layers.append(nn.Linear(tmp_hidden_size, self.num_edges))
            self.encoder_fc_out = nn.Sequential(*layers)

        num_layers = params['prior_num_layers']
        if num_layers == 1:
            self.prior_fc_out = nn.Linear(rnn_hidden_size, self.num_edges)
        else:
            tmp_hidden_size = params['prior_hidden_size']
            layers = [nn.Linear(rnn_hidden_size, tmp_hidden_size), nn.ELU(inplace=True)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(tmp_hidden_size, tmp_hidden_size))
                layers.append(nn.ELU(inplace=True))
            layers.append(nn.Linear(tmp_hidden_size, self.num_edges))
            self.prior_fc_out = nn.Sequential(*layers)

        self.num_vars = num_vars
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]

        self.use_3d = params.get('use_3d', False)
        self.num_relative_features = 12 if self.use_3d else 7
        self.num_pos_features = 6 if self.use_3d else 3
        self.edge_filter = AnisotropicEdgeFilter(
            self.num_relative_features+inp_size, self.num_pos_features,
            hidden_size, hidden_size, hidden_size, do_prob=dropout, bn=not no_bn)

        self.localizer = Localizer(params)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge(self, node_embeddings):
        # Input size: [batch, num_objects, num_timesteps, embed_size]
        send_embed = node_embeddings[:, self.send_edges]
        recv_embed = node_embeddings[:, self.recv_edges]

        return torch.cat([send_embed, recv_embed], dim=-1)

    def edge2node(self, edge_embeddings):
        if len(edge_embeddings.shape) == 4:
            old_shape = edge_embeddings.shape
            tmp_embeddings = edge_embeddings.view(old_shape[0], old_shape[1], -1)
            incoming = scatter(tmp_embeddings, torch.from_numpy(self.recv_edges).to(edge_embeddings.device), dim=1, reduce='sum').view(old_shape[0], -1, old_shape[2], old_shape[3])
        else:
            incoming = scatter(edge_embeddings, torch.from_numpy(self.recv_edges).to(edge_embeddings.device), dim=1, reduce='sum')
        return incoming / (self.num_vars-1)

    def forward(self, inputs):
        assert self.training or not self.save_eval_memory
        # Inputs is shape [batch, num_timesteps, num_objects, input_size]
        x = inputs.transpose(2, 1).contiguous()
        # New shape: [batch, num_objects, num_timesteps, input_size]

        # NOTE: Features should be "properly" normalized
        rel_feat, _, edge_attr, edge_pos = self.localizer(x)

        # Anisotropic filter generation
        edge_attr = self.edge_filter(edge_attr, edge_pos)

        edge_attr_prev = edge_attr
        res_x = self.res1(rel_feat)
        x = self.edge2node(edge_attr) + res_x
        x = self.mlp3(x)
        x = self.node2edge(x)
        x = torch.cat((x, edge_attr_prev), dim=-1)  # Skip connection
        x = self.mlp4(x)

        # At this point, x should be [batch, num_edges, num_timesteps, hidden_size]
        # RNN aggregation
        old_shape = x.shape
        x = x.contiguous().view(-1, old_shape[2], old_shape[3])
        forward_x, prior_state = self.forward_rnn(x)
        reverse_x = self.reverse_rnn(x.flip(1))[0].flip(1)

        # x: [batch*num_edges, num_timesteps, hidden_size]
        prior_result = self.prior_fc_out(forward_x).view(*old_shape[:3], self.num_edges).transpose(1,2).contiguous()
        combined_x = torch.cat([forward_x, reverse_x], dim=-1)
        encoder_result = self.encoder_fc_out(combined_x).view(*old_shape[:3], self.num_edges).transpose(1,2).contiguous()
        return prior_result, encoder_result, prior_state

    def single_step_forward(self, inputs, prior_state):
        # Inputs is shape [batch, num_objects, input_size]
        rel_feat, _, edge_attr, edge_pos = self.localizer(inputs)

        # Anisotropic filter generation
        edge_attr = self.edge_filter(edge_attr, edge_pos)

        edge_attr_prev = edge_attr
        res_x = self.res1(rel_feat)
        x = self.edge2node(edge_attr) + res_x
        x = self.mlp3(x)
        x = self.node2edge(x)
        x = torch.cat((x, edge_attr_prev), dim=-1)  # Skip connection
        x = self.mlp4(x)

        old_shape = x.shape
        x = x.contiguous().view(-1, 1, old_shape[-1])
        old_prior_shape = prior_state[0].shape
        prior_state = (prior_state[0].view(1, old_prior_shape[0]*old_prior_shape[1], old_prior_shape[2]),
                       prior_state[1].view(1, old_prior_shape[0]*old_prior_shape[1], old_prior_shape[2]))

        x, prior_state = self.forward_rnn(x, prior_state)
        prior_result = self.prior_fc_out(x).view(old_shape[0], old_shape[1], self.num_edges)
        prior_state = (prior_state[0].view(old_prior_shape), prior_state[1].view(old_prior_shape))
        return prior_result, prior_state


class RecurrentDecoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.num_vars = num_vars = params['num_vars']
        input_size = params['input_size']
        self.gpu = params['gpu']
        n_hid = params['decoder_hidden']
        edge_types = params['num_edge_types']
        skip_first = params['skip_first']
        out_size = params['input_size']
        self.dropout_prob = params['decoder_dropout']
        self.edge_types = edge_types

        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2*n_hid, n_hid) for _ in range(edge_types)]
        )
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)]
        )
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(input_size, n_hid, bias=True)
        self.input_i = nn.Linear(input_size, n_hid, bias=True)
        self.input_n = nn.Linear(input_size, n_hid, bias=True)

        self.present_r = nn.Linear(n_hid, n_hid, bias=True)
        self.present_i = nn.Linear(n_hid, n_hid, bias=True)
        self.present_n = nn.Linear(n_hid, n_hid, bias=True)

        self.out_mlp = nn.Sequential(
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(n_hid, out_size),
        )

        print('Using learned recurrent interaction net decoder.')

        self.num_vars = num_vars
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]

        self.use_3d = params.get('use_3d', False)
        self.num_relative_features = 12 if self.use_3d else 7
        self.num_pos_features = 6 if self.use_3d else 3

        self.present_msg_fc1 = nn.ModuleList(
            [nn.Linear(self.num_relative_features+input_size, n_hid) for _ in range(edge_types)]
        )
        self.present_msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)]
        )
        # self.edge_filter = nn.ModuleList(
            # [AnisotropicEdgeFilter(
                # self.num_relative_features+input_size, self.num_pos_features,
                # n_hid, n_hid, n_hid, act='tanh', do_prob=self.dropout_prob, bn=False)
             # for _ in range(edge_types)]
        # )

        self.localizer = Localizer(params)
        self.globalizer = Globalizer(params)

    def get_initial_hidden(self, inputs):
        return torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape, device=inputs.device)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs, hidden, edges):
        # Input Size: [batch, num_vars, input_size]
        # Hidden Size: [batch, num_vars, rnn_hidden]
        # Edges size: [batch, num_edges, num_edge_types]
        dropout_prob = self.dropout_prob if self.training else 0.0

        # node2edge
        receivers = hidden[:, self.recv_edges, :]
        senders = hidden[:, self.send_edges, :]

        # pre_msg: [batch, num_edges, 2*msg_out]
        pre_msg = torch.cat([receivers, senders], dim=-1)

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                               self.msg_out_shape, device=inputs.device)

        start_idx = 1 if self.skip_first_edge_type else 0
        # Run separate MLP for every edge type
        # NOTE: to exclude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = torch.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=dropout_prob)
            msg = torch.tanh(self.msg_fc2[i](msg))
            msg = msg * edges[:, :, i:i+1]
            all_msgs += msg

        # This step sums all of the messages per node
        agg_msgs = scatter(all_msgs, torch.from_numpy(self.recv_edges).cuda(), dim=1, reduce='mean')
        agg_msgs = agg_msgs.contiguous()

        rel_feat, Rinv, edge_attr, edge_pos = self.localizer(inputs)

        present_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                   self.msg_out_shape, device=inputs.device)
        for i in range(start_idx, self.edge_types):
            # msg = self.edge_filter[i](edge_attr, edge_pos)
            # msg = F.relu(msg)
            msg = torch.relu(self.present_msg_fc1[i](edge_attr))
            msg = F.dropout(msg, p=dropout_prob)
            msg = torch.relu(self.present_msg_fc2[i](msg))
            msg = msg * edges[:, :, i:i + 1]
            present_msgs += msg

        present_agg_msgs = scatter(present_msgs, torch.from_numpy(self.recv_edges).cuda(), dim=1, reduce='mean')
        present_agg_msgs = present_agg_msgs.contiguous()

        # GRU-style gated aggregation
        inp_r = self.input_r(rel_feat) + self.present_r(present_agg_msgs)
        inp_i = self.input_i(rel_feat) + self.present_i(present_agg_msgs)
        inp_n = self.input_n(rel_feat) + self.present_n(present_agg_msgs)

        r = torch.sigmoid(inp_r + self.hidden_r(agg_msgs))
        i = torch.sigmoid(inp_i + self.hidden_i(agg_msgs))
        n = torch.tanh(inp_n + r*self.hidden_h(agg_msgs))
        hidden = (1 - i)*n + i*hidden

        # Output MLP
        pred = self.out_mlp(hidden)

        pred_global = self.globalizer(pred, Rinv)
        outputs = inputs + pred_global

        return outputs, hidden


class MarkovDecoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.num_vars = params['num_vars']
        self.edge_types = params['num_edge_types']
        n_hid = params['decoder_hidden']
        msg_hid = params['decoder_hidden']
        msg_out = msg_hid  # TODO: make this a param
        skip_first = params['skip_first']
        in_size = params['input_size']

        self.dropout_prob = params['decoder_dropout']
        self.skip_first_edge_type = skip_first

        out_size = in_size
        self.res1 = nn.Linear(in_size, msg_out)

        self.out_mlp = nn.Sequential(
            nn.Linear(msg_out, n_hid),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(n_hid, out_size),
        )

        print('Using learned interaction net decoder.')

        edges = np.ones(self.num_vars) - np.eye(self.num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]

        self.use_3d = params.get('use_3d', False)
        self.num_relative_features = 12 if self.use_3d else 7
        self.num_pos_features = 6 if self.use_3d else 3

        self.num_used_edge_types = (
            self.edge_types - 1 if self.skip_first_edge_type
            else self.edge_types)
        # self.edge_filter = AnisotropicEdgeFilter(
            # self.num_relative_features+in_size, self.num_pos_features,
            # msg_hid, msg_hid, msg_hid * self.num_used_edge_types,
            # act='tanh', do_prob=self.dropout_prob, bn=False)

        self.edge_filter = MLPEdgeFilter(
            self.num_relative_features+in_size, self.num_pos_features,
            msg_hid, msg_hid, msg_out * self.num_used_edge_types,
            do_prob=self.dropout_prob)

        self.localizer = Localizer(params)
        self.globalizer = Globalizer(params)

    def get_initial_hidden(self, inputs):
        return None

    def forward(self, inputs, hidden, edges):
        # single_timestep_inputs has shape
        # [batch_size, num_objects, input_size]
        start_idx = 1 if self.skip_first_edge_type else 0

        # single_timestep_rel_type has shape:
        # [batch_size, num_objects*(num_objects-1), num_edge_types]
        # Node2edge
        rel_feat, Rinv, edge_attr, edge_pos = self.localizer(inputs)

        edge_attr = self.edge_filter(edge_attr, edge_pos)
        all_msgs = torch.sum(
            edge_attr.view(*edge_attr.shape[:-1], -1, self.num_used_edge_types)
            * edges[..., start_idx:].unsqueeze(-2), -1)

        # Aggregate all msgs to receiver
        agg_msgs = scatter(all_msgs, torch.from_numpy(self.recv_edges).to(all_msgs.device), dim=1, reduce='mean')
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        res_x = self.res1(rel_feat)
        aug_inputs = agg_msgs + res_x

        # Output MLP
        pred = self.out_mlp(aug_inputs)

        pred = self.globalizer(pred, Rinv)

        # Predict position/velocity difference
        return inputs + pred, None
