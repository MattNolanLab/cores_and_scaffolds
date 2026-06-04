from abc import ABC

from typing import Tuple, Union, List, Dict, Optional, Callable

import torch
from torch import nn
from torch.nn import Module, ModuleList
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence

from sample_factory.model.model_utils import ModelModule
from sample_factory.utils.typing import Config

from sample_factory.model.regularisation import L2

from einops import rearrange

try:
    from norse.torch.module import LSNNRecurrent, LSNNState, LIFState, LIFRecurrent, ConstantCurrentLIFEncoder, LILinearCell, LIState
    from norse.torch.functional import poisson_encode, decode
except ModuleNotFoundError as e:
    pass
    #print(e)
    #print('Norse was not installed. All SNN modules are disabled')


def apply_l2(cfg, module: nn.Module) -> nn.Module:
    if cfg.rnn_l2_weight_decay > 0.0:
        raise NotImplementedError
        print(f'Weight decay enabled {cfg.rnn_l2_weight_decay}. Applying to {module}')
        return L2(module, cfg.rnn_l2_weight_decay)
    else:
        #print(f'Weight decay not enabled. Skipping for {module}')
        return module


class ModelCore(ModelModule, ABC):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.core_output_size = -1  # to be overridden in derived classes

    def get_out_size(self) -> int:
        return self.core_output_size


class ModelCoreRNN(ModelCore):
    def __init__(self, cfg, input_size):
        super().__init__(cfg)

        self.cfg = cfg
        self.is_gru = False

        if cfg.rnn_type == "gru":
            self.core = nn.GRU(input_size, cfg.rnn_size, cfg.rnn_num_layers)
            self.is_gru = True
        elif cfg.rnn_type == "gru_nobias":
            self.core = nn.GRU(input_size, cfg.rnn_size, cfg.rnn_num_layers, bias=False)
            self.is_gru = True
        elif cfg.rnn_type == "lstm":
            self.core = nn.LSTM(input_size, cfg.rnn_size, cfg.rnn_num_layers)
        else:
            raise RuntimeError(f"Unknown RNN type {cfg.rnn_type}")

        self.core = apply_l2(cfg, self.core)
        self.core_output_size = cfg.rnn_size
        self.rnn_num_layers = cfg.rnn_num_layers

    def forward(self, head_output, rnn_states):
        is_seq = not torch.is_tensor(head_output) or head_output.ndim == 3
        if not is_seq:
            head_output = head_output.unsqueeze(0)

        if self.rnn_num_layers > 1:
            rnn_states = rnn_states.view(rnn_states.size(0), self.cfg.rnn_num_layers, -1)
            rnn_states = rnn_states.permute(1, 0, 2)
        else:
            rnn_states = rnn_states.unsqueeze(0)

        if self.is_gru:
            x, new_rnn_states = self.core(head_output, rnn_states.contiguous())
        else:
            h, c = torch.split(rnn_states, self.cfg.rnn_size, dim=2)
            x, (h, c) = self.core(head_output, (h.contiguous(), c.contiguous()))
            new_rnn_states = torch.cat((h, c), dim=2)

        if not is_seq:
            x = x.squeeze(0)

        if self.rnn_num_layers > 1:
            new_rnn_states = new_rnn_states.permute(1, 0, 2)
            new_rnn_states = new_rnn_states.reshape(new_rnn_states.size(0), -1)
        else:
            new_rnn_states = new_rnn_states.squeeze(0)

        return x, new_rnn_states


class ModelCoreVanillaRNN(ModelCore):
    def __init__(self, cfg, input_size):
        super().__init__(cfg)

        self.cfg = cfg

        if cfg.rnn_type == "rnn_relu":
            self.core = nn.RNN(input_size, cfg.rnn_size, cfg.rnn_num_layers, nonlinearity='relu', bias=False)
        elif cfg.rnn_type == "rnn_relu_bias":
            self.core = nn.RNN(input_size, cfg.rnn_size, cfg.rnn_num_layers, nonlinearity='relu', bias=True)
        elif cfg.rnn_type == "rnn_tanh":
            self.core = nn.RNN(input_size, cfg.rnn_size, cfg.rnn_num_layers, nonlinearity='tanh', bias=False)
        elif cfg.rnn_type == "rnn_tanh_bias":
            self.core = nn.RNN(input_size, cfg.rnn_size, cfg.rnn_num_layers, nonlinearity='tanh', bias=True)
        else:
            raise RuntimeError(f"Unknown RNN type {cfg.rnn_type}")

        self.core = apply_l2(cfg, self.core)

        self.core_output_size = cfg.rnn_size
        self.rnn_num_layers = cfg.rnn_num_layers

    def forward(self, head_output, rnn_states):
        is_seq = (not torch.is_tensor(head_output)) or head_output.ndim == 3
        if not is_seq:
            head_output = head_output.unsqueeze(0)

        if self.rnn_num_layers > 1:
            rnn_states = rnn_states.view(rnn_states.size(0), self.cfg.rnn_num_layers, -1)
            rnn_states = rnn_states.permute(1, 0, 2)
        else:
            rnn_states = rnn_states.unsqueeze(0)

        x, new_rnn_states = self.core(head_output, rnn_states.contiguous())

        if not is_seq:
            x = x.squeeze(0)

        if self.rnn_num_layers > 1:
            new_rnn_states = new_rnn_states.permute(1, 0, 2)
            new_rnn_states = new_rnn_states.reshape(new_rnn_states.size(0), -1)
        else:
            new_rnn_states = new_rnn_states.squeeze(0)

        return x, new_rnn_states


def _preprocess_x(head_output: Union[torch.Tensor, PackedSequence]) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
    """Preprocesses the head output (x). Handles unpacking PackedSequence"""
    if isinstance(head_output, PackedSequence):
        head_output, lengths = pad_packed_sequence(head_output)
    elif isinstance(head_output, torch.Tensor):
        head_output = head_output.unsqueeze(0)
        lengths = None
    else:
        raise RuntimeError

    return head_output, lengths


def _postprocess_x(x: torch.Tensor, lengths: Union[torch.Tensor, None]) -> Union[torch.Tensor, PackedSequence]:
    """Packs x back into a sequence if needed"""
    if lengths is None:
        return x.squeeze(0)
    else:
        return pack_padded_sequence(x, lengths, enforce_sorted=False)


class ModelCoreIdentity(ModelCore):
    """A noop core (no recurrency)."""

    def __init__(self, cfg, input_size):
        super().__init__(cfg)
        self.cfg = cfg
        self.core_output_size = input_size

    # noinspection PyMethodMayBeStatic
    def forward(self, head_output, fake_rnn_states):
        return head_output, fake_rnn_states


def default_make_core_func(cfg: Config, core_input_size: int) -> ModelCore:
    if cfg.use_rnn:
        if cfg.rnn_type in ['rnn_relu', 'rnn_tanh', 'rnn_relu_bias', 'rnn_tanh_bias']:
            core = ModelCoreVanillaRNN(cfg, core_input_size)
        else:
            raise NotImplementedError(f'{cfg.rnn_type} not valid')
    else:
        core = ModelCoreIdentity(cfg, core_input_size)

    if cfg.rnn_l2_weight_decay > 0:
        for name, module in core.named_children():
            if isinstance(module, nn.RNNBase):
                raise NotImplementedError('Weight decay was not applied to {param}')
            elif isinstance(module, L2):
                print(f'Weight decay applied to {name}-{module.module}')

    return core
