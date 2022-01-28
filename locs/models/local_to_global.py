import torch.nn as nn

from locs.utils.rotation_utilities import inv_rotation, inv_rotation3d


def _identity(x, *args, **kwargs):
    return x


class Globalizer(nn.Module):
    # Boolean Tuple Keys: trans_only & use_3d
    _local_to_global_fn = {
        (0, 0): inv_rotation,
        (0, 1): inv_rotation3d,
        (1, 0): _identity,
        (1, 1): _identity,
    }

    def __init__(self, params):
        super().__init__()
        self.use_3d = params.get('use_3d', False)
        self.trans_only = params.get('trans_only', False)

        self._local_to_global = self._local_to_global_fn[
            (self.trans_only, self.use_3d)]

    def forward(self, x, Rinv):
        y = self._local_to_global(x, Rinv)
        return y
