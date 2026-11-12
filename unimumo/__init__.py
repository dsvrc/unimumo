# PyTorch >= 2.6 changed torch.load's default to weights_only=True. UniMuMo's checkpoints
# (full.ckpt, Encodec/MusicGen packages, Lightning ckpts) store OmegaConf configs and numpy
# arrays, which that mode rejects. Restore the old default for every loader in this package.
# Only load checkpoints from sources you trust (here: the authors' HF repo and Meta's MusicGen).
import functools

import torch

if not getattr(torch.load, '_unimumo_patched', False):
    _torch_load = torch.load

    @functools.wraps(_torch_load)
    def _load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _torch_load(*args, **kwargs)

    _load._unimumo_patched = True
    torch.load = _load
