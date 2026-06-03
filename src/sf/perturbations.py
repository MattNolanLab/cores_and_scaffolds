import copy
from typing import List, Tuple

import torch
from einops import rearrange
from sample_factory.model.core import ModelCore

from sf.utils import Run


class TransientWeightModificationRNN:
    def __init__(self, rnn: torch.nn.RNNBase):
        self.rnn = rnn
        self.ih_weights = None
        self.hh_weights = None

    def __enter__(self):
        ih: torch.Tensor = self.rnn.weight_ih_l0.data
        self.ih_weights = copy.deepcopy(ih)

        hh: torch.Tensor = self.rnn.weight_hh_l0.data
        self.hh_weights = copy.deepcopy(hh)

        return ih, hh

    def __exit__(self, exc_type, exc_value, traceback):
        self.rnn.weight_ih_l0.data = copy.deepcopy(self.ih_weights)
        self.rnn.weight_hh_l0.data = copy.deepcopy(self.hh_weights)


class BasePerturbationWrapper(torch.nn.Module):
    def __init__(self, core: ModelCore, indices: torch.Tensor):
        super().__init__()

        # For thin compatibility
        self.core_output_size = core.core_output_size
        self.core: torch.nn.RNNBase = core.core
        self.cfg = core.cfg
        assert self.cfg.rnn_type in ['rnn_relu', 'rnn_relu_bias', 'rnn_tanh', 'rnn_tanh_bias']

        self.indices: torch.Tensor = indices
        self._validate_indices()  # TODO find some way to enforce this. Could easily delete and not know!

    def get_out_size(self) -> int:
        return self.core_output_size

    def forward(self, head_output, states) -> Tuple[torch.Tensor, torch.Tensor]:
        is_seq = (not torch.is_tensor(head_output)) or head_output.ndim == 3
        assert is_seq  # TODO: explanation
        # Only defined for [1, 1, x] (1 batch 1 timestep)
        assert head_output.shape[0] == head_output.shape[1] == 1
        assert states.shape[0] == states.shape[1] == 1

        output = self.forward_core(head_output, states)

        return output, output  # For 1 timestep, these are the same

    def forward_core(self, head_output, states) -> torch.Tensor:
        # E.g.
        # x, new_states = self.core(head_output, states)
        # return new_states  - note both are the same so it doesn't matter if return x..
        raise NotImplementedError('Perturbation not implemented')

    def _validate_indices(self):
        indices = self.indices
        assert type(indices) == torch.Tensor
        assert indices.ndim == 1
        assert not torch.any(indices < 0)
        assert not torch.any(indices > (self.get_out_size() - 1))
        assert len(torch.unique(indices)) == len(indices)
        if len(indices) == 0:
            print('Empty indices provided. User is probably running a control condition.')


class ScaleOutputWrapper(BasePerturbationWrapper):
    def __init__(self, core: ModelCore, indices: torch.Tensor, scaling_factor: float):
        super().__init__(core, indices)

        self.scaling_factor: float = scaling_factor
        if self.scaling_factor < 0:
            print(f'Warning: Scaling factor is negative ({self.scaling_factor})')

    def forward_core(self, head_output, states) -> torch.Tensor:
        new_states, new_states = self.core(head_output, states)

        new_states[0, 0, self.indices] *= self.scaling_factor

        return new_states


class RecordingScaleOutputWrapper(ScaleOutputWrapper):
    def __init__(self, core: ModelCore, indices: torch.Tensor, scaling_factor: float):
        super().__init__(core, indices, scaling_factor)

        self._pre_perturbation_states: List[torch.Tensor] = []

    def forward_core(self, head_output, states) -> torch.Tensor:
        # TODO refactor duplication with ScaleOutputWrapper
        new_states, new_states = self.core(head_output, states)
        self._pre_perturbation_states.append(copy.deepcopy(new_states))

        new_states[0, 0, self.indices] *= self.scaling_factor

        return new_states

    @property
    def pre_perturbation_states(self) -> torch.Tensor:
        states = torch.cat(self._pre_perturbation_states, dim=0)  # [5001, 1, 512]
        return rearrange(states, 't 1 n -> t n')

    def run_with_pre_states(self, run: Run) -> Run:
        return run.copy(deep=True, update={'states': self.pre_perturbation_states})
