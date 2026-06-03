from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

import numpy as np
import torch

from sf.analysis import prepare_input_for_obs
from sf.perturbations import TransientWeightModificationRNN
from sf.utils import Run, rank_truncate


@dataclass(frozen=True)
class DrivenTrajectory:
    states: torch.Tensor
    locations: np.ndarray
    velocities: np.ndarray
    ts: np.ndarray


class TruncationMode(Enum):
    FINAL = auto()  # Applies rank truncation at the t_t-1 -> t_t transition
    NONE = auto()  # Does not apply rank truncation


def run_over_inputs(
    data: Run,
    inputs: torch.Tensor,
    rank: int,
    truncation_mode: TruncationMode = TruncationMode.FINAL,
    initial_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    model = data.model

    device = inputs.device

    if truncation_mode == TruncationMode.FINAL:
        perturb_at_i: int = -1
    else:
        raise NotImplementedError

    if initial_state is None:
        rnn_states: torch.Tensor = torch.zeros([1, data.rnn.hidden_size], dtype=torch.float32, device=device)
    else:
        assert initial_state.shape == (1, data.rnn.hidden_size)
        rnn_states: torch.Tensor = initial_state

    states_history: List[torch.Tensor] = []

    truncated_whh = torch.clone(rank_truncate(torch.clone(data.rnn.weight_hh_l0), rank))
    ts = torch.arange(inputs.shape[0])
    for t in range(inputs.shape[0]):
        with torch.no_grad():
            with TransientWeightModificationRNN(data.rnn) as (w_ih, w_hh):
                if t in ts[perturb_at_i:]:  # Overly complicated
                    w_hh[:] = truncated_whh

                something_else, rnn_states = model.forward_core(inputs[t].unsqueeze(0), rnn_states)

        assert torch.all(something_else == rnn_states)  # 'Sanity double check'

        states_history.append(rnn_states)

    return torch.cat(states_history)


def simulate_driven_trajectory_simple(
    run: Run, base_v: float, before_final_v: float, outbound_distance: float, rank: int
) -> DrivenTrajectory:
    inputs_raw: List[torch.Tensor] = []

    # Blackbox
    velocities: List[float] = []
    pos: float = 0
    while pos < 30:
        v: float = base_v
        inputs_raw.append(
            prepare_input_for_obs(data=run, velocity=v, indicator=False, blackbox=True, beacon=False, context=3)
        )
        velocities.append(v)
        pos += v

    # Outbound zone
    for v in ([base_v] * int(outbound_distance // base_v)) + [before_final_v, base_v]:
        inputs_raw.append(
            prepare_input_for_obs(data=run, velocity=v, indicator=False, blackbox=False, beacon=False, context=3)
        )
        velocities.append(v)

    inputs: torch.Tensor = torch.concat(inputs_raw)

    states: torch.Tensor = run_over_inputs(run, inputs, rank)

    return DrivenTrajectory(
        states=states,
        locations=np.cumsum(np.array(velocities)),
        velocities=np.array(velocities),
        ts=np.arange(len(velocities)),
    )
