import numpy as np
import torch
import torch.nn as nn

from locs.utils.rotation_utilities import (
    canonicalize_inputs, create_edge_attr_pos_vel, create_3d_edge_attr_pos_vel,
    create_trans_edge_attr_pos_vel, create_trans_3d_edge_attr_pos_vel)


class Localizer(nn.Module):
    # Boolean Tuple Keys: trans_only & use_3d
    _global_to_local_fn = {
        (0, 0): create_edge_attr_pos_vel,
        (0, 1): create_3d_edge_attr_pos_vel,
        (1, 0): create_trans_edge_attr_pos_vel,
        (1, 1): create_trans_3d_edge_attr_pos_vel,
    }

    # Tuple Keys: use_3d & position_representation (cartesian vs polar)
    _edge_pos_idx_fn = {
        (0, 'cart'): [0, 1, 2],
        (0, 'polar'): [2, 3, 4],
        (1, 'cart'): [0, 1, 2, 3, 4, 5],
        (1, 'polar'): [3, 4, 5, 6, 7, 8],
    }

    def __init__(self, params):
        super().__init__()
        self.use_3d = params.get('use_3d', False)
        num_vars = params['num_vars']
        self.num_vars = num_vars
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]

        pos_representation = params['pos_representation']
        # NOTE: Default is polar
        if pos_representation not in ('cart', 'polar'):
            raise ValueError
        self.edge_pos_idx = self._edge_pos_idx_fn[
            (self.use_3d, pos_representation)]
        self.num_relative_features = 12 if self.use_3d else 7
        self.num_pos_features = 6 if self.use_3d else 3

        self.batched = params.get('batched', False)
        self.trans_only = params.get('trans_only', False)
        self._global_to_local = self._global_to_local_fn[
            (self.trans_only, self.use_3d)]

    def forward(self, x):
        rel_feat, Rinv = canonicalize_inputs(x, self.use_3d, self.trans_only)

        edge_attr = self._global_to_local(x, self.send_edges, self.recv_edges,
                                          batched=self.batched)
        edge_pos = edge_attr[..., self.edge_pos_idx]
        if self.batched:
            edge_attr = torch.cat([edge_attr, rel_feat[self.recv_edges]], -1)
        else:
            edge_attr = torch.cat([edge_attr, rel_feat[:, self.recv_edges]], -1)

        return rel_feat, Rinv, edge_attr, edge_pos
