from typing import Callable, List, Literal, NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks
from tqdm.autonotebook import tqdm

from sf.analysis import MIDDLE_CONTEXT, plot_stops_base, prepare_input_for_obs
from sf.attribution import Action
from sf.utils import Run


def find_peak_indices(signal: np.ndarray, height: float = None, distance: int = 1) -> Tuple[np.ndarray, dict]:
    return find_peaks(signal, height=height, distance=distance)


def find_highest_peak(signal: np.ndarray) -> Tuple[int, float]:
    max_i: int = np.argmax(signal)
    return max_i, signal[max_i]

    peaks, properties = find_peak_indices(signal)
    if len(peaks) == 0:
        return -1, 0.0

    peak_heights = signal[peaks]
    highest_peak_idx = peaks[np.argmax(peak_heights)]
    return highest_peak_idx, signal[highest_peak_idx]


def _build_inputs(
    run: Run, velocity_sampler: Callable[[], float], context: int = MIDDLE_CONTEXT
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert 'Contextual' in run.env_name

    inputs: List[torch.Tensor] = []
    velocities: List[float] = []

    travelled: float = 0
    while travelled < 30:
        v = velocity_sampler()
        travelled += v
        velocities.append(v)
        inputs.append(
            prepare_input_for_obs(data=run, velocity=v, indicator=False, blackbox=True, beacon=False, context=context)
        )

    travelled: float = 0
    while travelled < 140:
        v = velocity_sampler()
        travelled += v
        velocities.append(v)
        inputs.append(
            prepare_input_for_obs(data=run, velocity=v, indicator=False, blackbox=False, beacon=False, context=context)
        )

    return torch.tensor(np.cumsum([0] + velocities[:-1])), torch.cat(inputs, dim=0)


def run_disrupted_simulation(
    rnn: nn.RNNBase, inputs: torch.Tensor, neuron_i: int, mode: Literal['bias', 'zero'] = 'zero'
) -> torch.Tensor:
    """Simulates network activity with recurrent contributions from neuron_i being zeroed"""
    device = inputs.device

    states_history: List[torch.Tensor] = []  # Bit lazy...

    # Initialise the hidden state
    rnn_states: torch.Tensor = torch.zeros([1, rnn.hidden_size], dtype=torch.float32, device=device)

    for t in range(inputs.shape[0]):
        with torch.no_grad():
            if mode == 'bias':
                new_value = rnn.bias_ih_l0[neuron_i] + rnn.bias_hh_l0[neuron_i]
            elif mode == 'zero':
                new_value = 0
            else:
                raise NotImplementedError
            rnn_states[:, neuron_i] = new_value

            rnn_states, _ = rnn(
                inputs[t].unsqueeze(0), rnn_states
            )  # returns [t, n], [1, n] - equivalent for 1 timestep

        states_history.append(rnn_states)

    return torch.cat(states_history)


def run_control_simulation(rnn: nn.RNNBase, inputs: torch.Tensor) -> torch.Tensor:
    """Simulates network activity with recurrent contributions from neuron_i being zeroed"""
    device = inputs.device

    states_history: List[torch.Tensor] = []  # Bit lazy...

    # Initialise the hidden state
    rnn_states: torch.Tensor = torch.zeros([1, rnn.hidden_size], dtype=torch.float32, device=device)

    for t in range(inputs.shape[0]):
        with torch.no_grad():
            rnn_states, _ = rnn(
                inputs[t].unsqueeze(0), rnn_states
            )  # returns [t, n], [1, n] - equivalent for 1 timestep

        states_history.append(rnn_states)

    return torch.cat(states_history)


def get_crossover_point(readout: torch.Tensor, default: int, crossover_threshold: float) -> int:
    probs: torch.Tensor = nn.functional.softmax(readout, dim=-1)  # [t, 2]

    crossover_mask: torch.Tensor = probs[:, Action.STOP] > crossover_threshold
    if not any(crossover_mask):
        score: int = default
    else:
        score: int = torch.where(crossover_mask)[0][0].item()
    return score


class DisruptionResults(NamedTuple):
    displacements: torch.Tensor
    heights: torch.Tensor
    delta_heights: torch.Tensor
    readouts: List[torch.Tensor]
    control_readout: torch.Tensor
    scores: torch.Tensor
    locations: torch.Tensor


def to_probs(readout: torch.Tensor) -> torch.Tensor:
    return nn.functional.softmax(readout, dim=-1)  # [t, 2]


def calculate_disruption_scores(
    run: Run,
    perturbation_mode: Literal['zero', 'bias'] = 'zero',
    preprocessing: Literal['none', 'softmax'] = 'softmax',
    velocity_sampler: Optional[Callable[[], float]] = lambda: 1,
) -> DisruptionResults:
    locations, inputs = _build_inputs(run, velocity_sampler=velocity_sampler)

    if preprocessing == 'none':

        def p(x):
            return x
    elif preprocessing == 'softmax':

        def p(x):
            return nn.functional.softmax(x, dim=-1)
    else:
        raise NotImplementedError

    readouts: List[torch.Tensor] = []  # List of [t, 2]
    for i in tqdm(range(run.rnn.hidden_size)):
        with torch.no_grad():
            states: torch.Tensor = run_disrupted_simulation(run.rnn, inputs, i, perturbation_mode)
            decoded: torch.Tensor = run.model.decoder(states)
            readout: torch.Tensor = run.model.action_parameterization(decoded)[0]
        readouts.append(readout)

    with torch.no_grad():
        control_readout: torch.Tensor = run.model.action_parameterization(
            run.model.decoder(run_control_simulation(run.rnn, inputs))
        )[0]

    control_i, control_height = find_highest_peak(p(control_readout)[:, Action.STOP].numpy())
    control_loc: float = locations[control_i].item()

    displacements: List[float] = [
        locations[find_highest_peak(p(r)[:, Action.STOP].numpy())[0]].item() - control_loc for r in readouts
    ]
    heights: List[float] = [find_highest_peak(p(r)[:, Action.STOP].numpy())[1] for r in readouts]
    delta_heights: List[float] = [find_highest_peak(p(r)[:, Action.STOP].numpy())[1] - control_height for r in readouts]

    return DisruptionResults(
        displacements=torch.tensor(displacements),
        heights=torch.tensor(heights),
        delta_heights=torch.tensor(delta_heights),
        readouts=readouts,
        control_readout=control_readout,
        scores=torch.tensor(displacements) * torch.tensor(heights),
        locations=locations,
    )


def plot_disruption_result(results: DisruptionResults, neuron_id: int, probs: bool = False) -> None:
    readout: torch.Tensor = results.readouts[neuron_id]
    control_readout: torch.Tensor = results.control_readout

    if probs:

        def t(x):
            return nn.functional.softmax(x, dim=-1)
    else:

        def t(x):
            return x

    xs: List[float] = results.locations.tolist()

    if probs:
        plt.plot(xs, t(readout)[:, Action.STOP], color='black')
        plt.plot(xs, t(control_readout)[:, Action.STOP], color='black', alpha=0.2)
        plt.ylim(0, 1)
    else:
        plt.plot(xs, t(control_readout)[:, Action.STOP], color='red', alpha=0.2)
        plt.plot(xs, t(control_readout)[:, Action.GO], color='green', alpha=0.2)
        plt.plot(xs, t(readout)[:, Action.STOP], color='red', label='Stop')
        plt.plot(xs, t(readout)[:, Action.GO], color='green', label='Go')

    plot_stops_base()

    # plt.title(f'Neuron {neuron_id}. Score: {results.scores[neuron_id]}')
    plt.ylabel('Stop prob.' if probs else 'Action output')


class RunDisplacementResults(NamedTuple):
    peak_locations: np.ndarray
    peak_heights: np.ndarray


@torch.no_grad()
def calculate_run_displacements(
    run: Run, trial_type: Literal['b', 'nb'], preprocessing: Literal['none', 'softmax'] = 'softmax'
) -> RunDisplacementResults:

    from sf.attribution import Action

    if preprocessing == 'none':

        def p(x):
            return x
    elif preprocessing == 'softmax':

        def p(x):
            return nn.functional.softmax(x, dim=-1)
    else:
        raise NotImplementedError

    decoded: torch.Tensor = run.model.decoder(run.states)
    readout: torch.Tensor = run.model.action_parameterization(decoded)[0]  # [t, 2]
    stop_probs: torch.Tensor = p(readout)[:, Action.STOP]  # [t]

    peak_locations: List[float] = []
    peak_heights: List[float] = []
    for trial_i, trial_mask in enumerate(run.iter_global_trial_masks()):
        assert len(np.unique(run.trial_types[trial_mask])) == 1
        if (
            run.trial_types[trial_mask][3] != trial_type
        ):  # TODO
            continue

        trial_stop_probs: np.ndarray = stop_probs[trial_mask].numpy()

        if not np.all(trial_stop_probs == 0):
            peak_i, peak_height = find_highest_peak(trial_stop_probs)
            peak_heights.append(peak_height)
            peak_locations.append(run.locations[trial_mask][peak_i].item())
        else:
            peak_heights.append(np.nan)
            peak_locations.append(np.nan)
            print(f'Warning: got nan result at global trial {trial_i} due to all zeroes. ')

    # valid_mask = np.isfinite(peak_locations)

    return RunDisplacementResults(
        np.array(peak_locations),  # [valid_mask],
        np.array(peak_heights),  # [valid_mask],
    )
