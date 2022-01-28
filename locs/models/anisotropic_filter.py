from torch import nn
import torch.nn.functional as F

from locs.models.activations import ACTIVATIONS


class AnisotropicEdgeFilter(nn.Module):
    def __init__(self, in_size, pos_size, hidden_size, dummy_size, out_size,
                 act='elu', **kwargs):
        super().__init__()
        self.num_relative_features = in_size
        self.out_size = out_size
        self._act = act
        self.edge_filter = nn.Sequential(
            nn.Linear(pos_size, hidden_size),
            ACTIVATIONS[act](),
            nn.Linear(hidden_size, self.num_relative_features * out_size),
        )
        self.init_weights()

    def init_weights(self):
        if self._act == 'elu':
            gain = nn.init.calculate_gain('relu')
        else:
            gain = nn.init.calculate_gain(self._act)
        nn.init.orthogonal_(self.edge_filter[0].weight, gain=gain)
        nn.init.orthogonal_(self.edge_filter[2].weight)

    def forward(self, edge_attr, edge_pos):
        edge_weight = self.edge_filter(edge_pos)
        edge_weight = edge_weight.reshape(
            edge_weight.shape[:-1] + tuple([self.num_relative_features, -1]))
        edge_attr = (edge_attr.unsqueeze(-2) @ edge_weight).squeeze(-2)
        return edge_attr


class MLPEdgeFilter(nn.Module):
    """2-layer MLP, follows same template as AnisotropicEdgeFilter"""
    def __init__(self, in_size, pos_size, hidden_size, bottleneck_size,
                 out_size, do_prob=0.0):
        super().__init__()
        self.num_relative_features = in_size
        self.out_size = out_size
        self.hidden_size = bottleneck_size

        self.lin1 = nn.Linear(self.num_relative_features, bottleneck_size)
        self.drop1 = nn.Dropout(p=do_prob)
        self.lin2 = nn.Linear(bottleneck_size, out_size)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, edge_attr, edge_pos):
        edge_attr = F.relu(self.lin1(edge_attr))
        edge_attr = self.drop1(edge_attr)
        edge_attr = F.relu(self.lin2(edge_attr))
        return edge_attr
