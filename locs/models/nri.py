import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from locs.models import model_utils


class BaseNRI(nn.Module):
    def __init__(self, num_vars, encoder, decoder, params):
        super(BaseNRI, self).__init__()
        # Model Params
        self.num_vars = num_vars
        self.decoder = decoder
        self.encoder = encoder
        self.num_edge_types = params.get('num_edge_types')

        # Training params
        self.gumbel_temp = params.get('gumbel_temp')
        self.train_hard_sample = params.get('train_hard_sample')
        self.teacher_forcing_steps = params.get('teacher_forcing_steps', -1)
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

        self.normalize_kl = params.get('normalize_kl', False)
        self.normalize_kl_per_var = params.get('normalize_kl_per_var', False)
        self.normalize_nll = params.get('normalize_nll', False)
        self.normalize_nll_per_var = params.get('normalize_nll_per_var', False)
        self.kl_coef = params.get('kl_coef', 1.)
        self.nll_loss_type = params.get('nll_loss_type', 'crossent')
        self.prior_variance = params.get('prior_variance')
        self.timesteps = params.get('timesteps', 0)
        self.extra_context = params.get('embedder_time_bins', 0)
        self.burn_in_steps = params.get('train_burn_in_steps')
        self.no_prior = params.get('no_prior', False)
        self.val_teacher_forcing_steps = params.get('val_teacher_forcing_steps', -1)

    def calculate_loss(self, inputs, is_train=False, teacher_forcing=True, return_edges=False, return_logits=False):
        # Should be shape [batch, num_edges, edge_dim]
        encoder_results = self.encoder(inputs)
        logits = encoder_results['logits']
        old_shape = logits.shape
        hard_sample = (not is_train) or self.train_hard_sample
        edges = model_utils.gumbel_softmax(
            logits.view(-1, self.num_edge_types),
            tau=self.gumbel_temp,
            hard=hard_sample).view(old_shape)
        if not is_train and teacher_forcing:
            teacher_forcing_steps = self.val_teacher_forcing_steps
        else:
            teacher_forcing_steps = self.teacher_forcing_steps
        output = self.decoder(inputs[:, self.extra_context:-1], edges,
                              teacher_forcing=teacher_forcing,
                              teacher_forcing_steps=teacher_forcing_steps)
        if len(inputs.shape) == 4:
            target = inputs[:, self.extra_context+1:, :, :]
        else:
            target = inputs[:, self.extra_context+1:, :]
        loss_nll = self.nll(output, target)
        prob = F.softmax(logits, dim=-1)
        if self.no_prior:
            loss_kl = torch.cuda.FloatTensor([0.0])
        elif self.log_prior is not None:
            loss_kl = self.kl_categorical(prob)
        else:
            loss_kl = self.kl_categorical_uniform(prob)
        loss = loss_nll + self.kl_coef*loss_kl
        loss = loss.mean()
        if return_edges:
            return loss, loss_nll, loss_kl, edges
        elif return_logits:
            return loss, loss_nll, loss_kl, logits, output
        else:
            return loss, loss_nll, loss_kl


    def predict_future(self, data_encoder, data_decoder):
        raise NotImplementedError()

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

    def kl_categorical(self, preds, eps=1e-16):
        kl_div = preds*(torch.log(preds+eps) - self.log_prior)
        if self.normalize_kl:
            return kl_div.sum(-1).view(preds.size(0), -1).mean(dim=1)
        elif self.normalize_kl_per_var:
            return kl_div.sum() / (self.num_vars * preds.size(0))
        else:
            return kl_div.view(preds.size(0), -1).sum(dim=1)

    def kl_categorical_uniform(self, preds, eps=1e-16):
        kl_div = preds*(torch.log(preds + eps) + np.log(self.num_edge_types))
        if self.normalize_kl:
            return kl_div.sum(-1).view(preds.size(0), -1).mean()
        elif self.normalize_kl_per_var:
            return kl_div.sum() / (self.num_vars * preds.size(0))
        else:
            return kl_div.view(preds.size(0), -1).sum(dim=1)/self.num_edge_types

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class StaticNRI(BaseNRI):
    def predict_future(self, inputs, prediction_steps, return_edges=False, return_everything=False):
        encoder_dict = self.encoder(inputs)
        logits = encoder_dict['logits']
        old_shape = logits.shape
        edges = nn.functional.gumbel_softmax(
            logits.view(-1, self.num_edge_types),
            tau=self.gumbel_temp,
            hard=True).view(old_shape)
        tmp_predictions, decoder_state = self.decoder(inputs[:, :-1], edges, teacher_forcing=True, teacher_forcing_steps=-1, return_state=True)
        decoder_inputs = inputs[:, -1].unsqueeze(1)
        predictions = self.decoder(decoder_inputs, edges, prediction_steps=prediction_steps, teacher_forcing=False, state=decoder_state)
        if return_everything:
            predictions = torch.cat([tmp_predictions, predictions], dim=1)
        if return_edges:
            return predictions, edges
        else:
            return predictions


class DynamicNRI(BaseNRI):
    def predict_future(self, inputs, prediction_steps, return_edges=False, return_everything=False):
        encoder_dict = self.encoder(inputs)
        burn_in_logits = encoder_dict['logits']
        encoder_state = encoder_dict['state']
        old_shape = burn_in_logits.shape
        burn_in_edges = nn.functional.gumbel_softmax(
            burn_in_logits.view(-1, self.num_edge_types),
            tau=self.gumbel_temp,
            hard=True).view(old_shape)
        burn_in_predictions, decoder_state = self.decoder(inputs, burn_in_edges, teacher_forcing=True, teacher_forcing_steps=-1, return_state=True)
        prev_preds = burn_in_predictions[:, -1].unsqueeze(1)
        preds = [prev_preds]
        all_edges = [burn_in_edges]
        for step in range(prediction_steps-1):
            encoder_dict = self.encoder(prev_preds, encoder_state)
            logits = encoder_dict['logits']
            encoder_state = encoder_dict['state']
            old_shape = logits.shape
            edges = nn.functional.gumbel_softmax(
                logits.view(-1, self.num_edge_types),
                tau=self.gumbel_temp,
                hard=True).view(old_shape)
            if return_edges:
                all_edges.append(edges)
            prev_preds, decoder_state = self.decoder(prev_preds, edges, teacher_forcing=False, prediction_steps=1, return_state=True, state=decoder_state)
            preds.append(prev_preds)
        preds = torch.cat(preds, dim=1)
        if return_everything:
            preds = torch.cat([burn_in_predictions[:, :-1], preds], dim=1)
        if return_edges:
            return preds, torch.stack(all_edges, dim=1)
        else:
            return preds
