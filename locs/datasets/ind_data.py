import os
import argparse
import math

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from locs.utils import data_utils


class IndData(Dataset):
    def __init__(self, data_path, mode, params):
        self.mode = mode
        if mode == 'train':
            path = os.path.join(data_path, 'processed_train_data')
        elif mode == 'valid':
            path = os.path.join(data_path, 'processed_val_data')
        elif mode == 'test':
            path = os.path.join(data_path, 'processed_test_data')
        self._data_path = data_path
        self.feats, self.masks = torch.load(path)
        self.vel_norm_norm = params['vel_norm_norm']
        self.normalize_data(data_path)
        self.train_data_len = params.get('train_data_len', -1)
        self.burn_in_masks = []
        self.node_inds = []
        self.graph_info = []
        max_burn_in_count = params['max_burn_in_count']
        if max_burn_in_count == -1:
            max_burn_in_count = 1000000000000000000

        if mode == 'test' and params['test_short_sequences']:
            num_full_steps = 50
            new_feats = sum([list(feat.split(num_full_steps)) for feat in self.feats], [])
            new_masks = sum([list(mask.split(num_full_steps)) for mask in self.masks], [])
            self._chunks_per_scene = [
                int(math.ceil(feat.size(0) / float(num_full_steps)))
                for feat in self.feats]
            non_empty_mask = [m.bool()[6:].any() for m in new_masks]
            new_feats = [f for idx, f in enumerate(new_feats) if non_empty_mask[idx]]
            new_masks = [m for idx, m in enumerate(new_masks) if non_empty_mask[idx]]
            self.feats = new_feats
            self.masks = new_masks
            self._road_images = torch.load(
                os.path.join(self._data_path, 'test_short_seq_roads.pt'))
        elif mode == 'test' and params['eval_sequentially']:
            num_full_steps = 16
            self._scene_indices = [
                int(math.ceil((feat.size(0)-num_full_steps) / 5.0))
                for feat in self.feats]
            new_feats = sum([[feat[i:i+num_full_steps] for i in range(0, feat.shape[0]-num_full_steps, 5)] for feat in self.feats], [])
            new_masks = sum([[mask[i:i+num_full_steps] for i in range(0, mask.shape[0]-num_full_steps, 5)] for mask in self.masks], [])

            non_empty_mask = [m.bool()[6:].any() for m in new_masks]
            new_feats = [f for idx, f in enumerate(new_feats) if non_empty_mask[idx]]
            new_masks = [m for idx, m in enumerate(new_masks) if non_empty_mask[idx]]

            self.feats = new_feats
            self.masks = new_masks
            self._road_images = torch.load(
                os.path.join(self._data_path, 'test_short_seq_roads.pt'))

        self._utm_conversion = torch.load(
            os.path.join(self._data_path, 'utm_conversion.pt'))
        graph_cache = {}
        print("Building graph info...")
        for mask in self.masks:
            burn_in_mask = torch.zeros_like(mask, dtype=torch.float32)
            burn_in_count = torch.zeros(mask.size(1), dtype=torch.float32)
            current_node_inds = []
            self.node_inds.append(current_node_inds)
            current_graph_info = []
            self.graph_info.append(current_graph_info)
            for step in range(len(mask)):
                burn_in_mask[step] = (mask[step] == 1).float()*(burn_in_count < max_burn_in_count).float()
                burn_in_count += burn_in_mask[step]
                current_node_inds.append(mask[step].nonzero()[:, -1])
                node_len = len(current_node_inds[-1])
                if node_len in graph_cache:
                    graph_info = graph_cache[node_len]
                else:
                    graph_info = get_graph_info(mask[step], node_len)
                current_graph_info.append(graph_info)
            self.burn_in_masks.append(burn_in_mask)
        self.expand_train = params.get('expand_train', False)
        if self.mode == 'train' and self.expand_train and self.train_data_len > 0:
            self.all_inds = []
            for ind in range(len(self.feats)):
                t_ind = 0
                while t_ind < len(self.feats[ind]):
                    self.all_inds.append((ind, t_ind))
                    t_ind += self.train_data_len
        else:
            self.expand_train = False

    def load_road_image(self, idx):
        img = Image.open(os.path.join(self._data_path, self._road_images[idx]))
        return img

    def get_image_conversion(self, idx):
        return self._utm_conversion[idx]

    def normalize_data(self, data_path):
        if self.vel_norm_norm:
            path = os.path.join(data_path, 'train_vel_norm_stats')
            _, vel_norm_max = torch.load(path)
            if isinstance(self.feats, torch.Tensor):
                self.feats = self.feats / vel_norm_max
            else:
                self.feats = [f / vel_norm_max for f in self.feats]
        else:
            path = os.path.join(data_path, 'train_data_stats')
            min_feats, max_feats = torch.load(path)
            min_feats = min_feats.view(1, 1, -1)
            max_feats = max_feats.view(1, 1, -1)
            for i in range(len(self.feats)):
                self.feats[i] = data_utils.normalize(self.feats[i], max_feats, min_feats)

    def unnormalize_data(self, feats):
        if self.vel_norm_norm:
            path = os.path.join(self._data_path, 'train_vel_norm_stats')
            _, vel_norm_max = torch.load(path)
            if isinstance(feats, torch.Tensor):
                unnorm_feats = feats * vel_norm_max
            else:
                unnorm_feats = torch.cat([f * vel_norm_max for f in feats])
            return unnorm_feats
        else:
            path = os.path.join(self._data_path, 'train_data_stats')
            min_feats, max_feats = torch.load(path)
            min_feats = min_feats.view(1, 1, -1)
            max_feats = max_feats.view(1, 1, -1)
            unnorm_feats = []
            for i in range(len(feats)):
                unnorm_feats.append(data_utils.unnormalize(feats[i], max_feats,
                                                        min_feats))
            return torch.cat(unnorm_feats)

    def __len__(self):
        if self.expand_train:
            return len(self.all_inds)
        else:
            return len(self.feats)

    def __getitem__(self, index):
        if self.expand_train:
            index, t_ind = self.all_inds[index]
            start_ind = np.random.randint(t_ind, t_ind + self.train_data_len)
            inputs = self.feats[index][start_ind:start_ind + self.train_data_len]
            masks = self.masks[index][start_ind:start_ind+self.train_data_len]
            burn_in_masks = self.burn_in_masks[index][start_ind:start_ind+self.train_data_len]
            node_inds = self.node_inds[index][start_ind:start_ind+self.train_data_len]
            graph_info = self.graph_info[index][start_ind:start_ind+self.train_data_len]
            if len(inputs) < self.train_data_len:
                inputs = self.feats[index][-self.train_data_len:]
                masks = self.masks[index][-self.train_data_len:]
                burn_in_masks = self.burn_in_masks[index][-self.train_data_len:]
                node_inds = self.node_inds[index][-self.train_data_len:]
                graph_info = self.graph_info[index][-self.train_data_len:]
        elif self.mode == 'train' and self.train_data_len > 0 and len(self.feats[index]) > self.train_data_len:
            start_ind = np.random.randint(0, len(self.feats[index])-self.train_data_len)
            inputs = self.feats[index][start_ind:start_ind+self.train_data_len]
            masks = self.masks[index][start_ind:start_ind+self.train_data_len]
            burn_in_masks = self.burn_in_masks[index][start_ind:start_ind+self.train_data_len]
            node_inds = self.node_inds[index][start_ind:start_ind+self.train_data_len]
            graph_info = self.graph_info[index][start_ind:start_ind+self.train_data_len]
        else:
            inputs = self.feats[index]
            masks = self.masks[index]
            burn_in_masks = self.burn_in_masks[index]
            node_inds = self.node_inds[index]
            graph_info = self.graph_info[index]
        return {'inputs':inputs, 'masks':masks, 'burn_in_masks':burn_in_masks,
                'node_inds':node_inds, 'graph_info':graph_info}

def get_graph_info(masks, num_vars, use_edge2node=True):
    if num_vars == 1:
        return None, None, None
    edges = torch.ones(num_vars, device=masks.device) - torch.eye(num_vars, device=masks.device)
    tmp = torch.where(edges)
    send_edges = tmp[0]
    recv_edges = tmp[1]
    tmp_inds = torch.tensor(list(range(num_vars)), device=masks.device, dtype=torch.long).unsqueeze_(1)
    if use_edge2node:
        edge2node_inds = (tmp_inds == recv_edges.unsqueeze(0)).nonzero()[:, 1].contiguous().view(-1, num_vars-1)
        return send_edges, recv_edges, edge2node_inds
    else:
        return send_edges, recv_edges

def ind_collate_fn(batch):
    inputs = [entry['inputs'] for entry in batch]
    masks = [entry['masks'] for entry in batch]
    burn_in_masks = [entry['burn_in_masks'] for entry in batch]
    node_inds = [entry['node_inds'] for entry in batch]
    graph_info = [entry['graph_info'] for entry in batch]
    max_len = max([inp.size(0) for inp in inputs])
    max_id = max([inp.size(1) for inp in inputs])
    final_inputs = torch.zeros(len(inputs), max_len, max_id, inputs[0].size(-1))
    final_masks = torch.zeros(len(inputs), max_len, max_id)
    final_burn_in_masks = torch.zeros(len(inputs), max_len, max_id)
    for ind, (inp, mask, burn_in_mask) in enumerate(zip(inputs, masks, burn_in_masks)):
        final_inputs[ind, :inp.size(0), :inp.size(1)] = inp
        final_masks[ind, :mask.size(0), :mask.size(1)] = mask
        final_burn_in_masks[ind, :mask.size(0), :mask.size(1)] = burn_in_mask
    return {'inputs': final_inputs, 'masks': final_masks, 'burn_in_masks': final_burn_in_masks,
            'node_inds':node_inds, 'graph_info':graph_info}


if __name__ == '__main__':
    import locs.datasets.utils.ind_data_utils as idu

    parser = argparse.ArgumentParser('Build ind datasets')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_train', type=int, default=19)
    parser.add_argument('--num_val', type=int, default=7)
    parser.add_argument('--downsample_factor', type=int, default=10)
    args = parser.parse_args()

    all_tracks, all_static, all_meta = idu.read_all_recordings_from_csv(args.data_dir)
    all_feats = []
    all_masks = []
    min_feats = np.array([100000000000000, 100000000000000, 100000000000000, 10000000000000])
    max_feats = np.array([-100000000000000, -100000000000000, -100000000000000, -10000000000000])
    for track_set_id,track_set in enumerate(all_tracks):
        num_tracks = len(track_set)
        max_frame = 0
        for track_info in all_static[track_set_id]:
            max_frame = max(max_frame, track_info['finalFrame'])
        print("%d: %d", track_set_id, max_frame)
        feats = np.zeros((max_frame+1, num_tracks, 4))
        masks = np.zeros((max_frame+1, num_tracks))
        for track_id, track in enumerate(track_set):
            frames = track['frame']
            feats[frames, track_id, 0] = track['xCenter']
            feats[frames, track_id, 1] = track['yCenter']
            feats[frames, track_id, 2] = track['xVelocity']
            feats[frames, track_id, 3] = track['yVelocity']
            masks[frames, track_id] = 1
            if track_set_id < args.num_train:
                min_feats[0] = min(min_feats[0], track['xCenter'].min())
                min_feats[1] = min(min_feats[1], track['yCenter'].min())
                min_feats[2] = min(min_feats[2], track['xVelocity'].min())
                min_feats[3] = min(min_feats[3], track['yVelocity'].min())
                max_feats[0] = max(max_feats[0], track['xCenter'].max())
                max_feats[1] = max(max_feats[1], track['yCenter'].max())
                max_feats[2] = max(max_feats[2], track['xVelocity'].max())
                max_feats[3] = max(max_feats[3], track['yVelocity'].max())
        all_feats.append(torch.FloatTensor(feats[::args.downsample_factor]))
        all_masks.append(torch.FloatTensor(masks[::args.downsample_factor]))
    train_feats = all_feats[:args.num_train]
    val_feats = all_feats[args.num_train:args.num_train+args.num_val]
    test_feats = all_feats[args.num_train+args.num_val:]
    train_masks = all_masks[:args.num_train]
    val_masks = all_masks[args.num_train:args.num_train+args.num_val]
    test_masks = all_masks[args.num_train+args.num_val:]
    train_path = os.path.join(args.output_dir, 'processed_train_data')
    torch.save([train_feats, train_masks], train_path)
    val_path = os.path.join(args.output_dir, 'processed_val_data')
    torch.save([val_feats, val_masks], val_path)
    test_path = os.path.join(args.output_dir, 'processed_test_data')
    torch.save([test_feats, test_masks], test_path)

    stats_path = os.path.join(args.output_dir, 'train_data_stats')
    torch.save([torch.FloatTensor(min_feats), torch.FloatTensor(max_feats)], stats_path)
