from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
import seaborn as sns
import torch
import torch.nn as nn

from sf import EXPERIMENTS
from sf.utils import Run, tqdm_joblib


@dataclass()
class FP:
    q_history: List[float]
    state_history: torch.Tensor
    recurrent_jacobian: torch.Tensor
    unsorted_jac_eigenvals: np.ndarray
    unsorted_jac_eigenvecs: np.ndarray

    def get_recurrent_jacobian_eigendecomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        eigenvals = self.unsorted_jac_eigenvals
        eigenvecs = self.unsorted_jac_eigenvecs

        eigenvals_real = np.real(eigenvals)

        eig_sorted_indices = np.argsort(eigenvals_real)[::-1]  # Flip because argsort returns in ascending order
        eigenvals_real_sorted = eigenvals_real[eig_sorted_indices]
        eigenvectors_real_sorted = np.real(eigenvecs[:, eig_sorted_indices])

        return eigenvals_real_sorted, eigenvectors_real_sorted

    @property
    def unstable_modes(self) -> List[np.ndarray]:
        if self.is_stable:
            raise ValueError('Tried to get unstable modes of a stable fixed point!')

        eigenvals, eigenvecs = self.get_recurrent_jacobian_eigendecomposition()

        modes: List[np.ndarray] = []
        for i, eigenval in enumerate(eigenvals):
            magnitude = 0.45
            if eigenval >= 1:
                xstar_plus = self.fp + (eigenvals[i] * magnitude) * eigenvecs[:, i]
                xstar_minus = self.fp - (eigenvals[i] * magnitude) * eigenvecs[:, i]
                xstar_mode = np.vstack([xstar_minus, self.fp, xstar_plus])

                modes.append(xstar_mode)

        return modes

    @property
    def n_unstable_modes(self) -> int:
        if self.is_stable:
            return 0
        else:
            return len(self.unstable_modes)

    @property
    def is_stable(self):
        eigenvals, _ = self.get_recurrent_jacobian_eigendecomposition()

        return np.all(np.real(eigenvals) <= 1)

    @property
    def fp(self) -> torch.Tensor:
        return self.state_history[-1]

    @property
    def fp_batched(self):
        return self.fp.unsqueeze(0)  # [128] -> [1, 128]

    @property
    def q(self) -> float:
        return self.q_history[-1]


def calculate_trial_speeds(data: Run) -> List[float]:
    speeds = []
    trial_states = data.states
    for t in range(1, trial_states.shape[0]):
        q = 0.5 * torch.sum((trial_states[t - 1] - trial_states[t]) ** 2)
        speeds.append(q.item())
    return speeds


def filter_fps(fps: List[FP], qtol=1e-12, plot=False) -> List[FP]:
    valid_fps = []

    qs = [fp.q for fp in fps]

    if plot:
        qs_numpy: np.ndarray = np.array(qs)
        qs_nonzero_numpy: np.ndarray = qs_numpy[qs_numpy != 0.0]

        if len(qs_numpy) != len(qs_nonzero_numpy):
            print(
                f'Warning: Some fixed points have a speed of 0.0!. {len(qs_nonzero_numpy)}/{len(qs_numpy)}. Skipping plotting these for visualisation because it will break the log scaled histogram.'
            )

        sns.histplot(qs_nonzero_numpy, stat='proportion', log_scale=True, color='blue')
        plt.axvline(x=qtol, c='black')
        plt.title(f'Speed threshold: {qtol}')

    for fp in fps:
        if fp.q < qtol:
            valid_fps.append(fp)
    return valid_fps


def filter_stable(fps: List[FP]) -> List[FP]:
    return [fp for fp in fps if fp.is_stable]


def add_noise(tensor: torch.Tensor, std: float) -> torch.Tensor:
    return torch.normal(0, std, size=tensor.shape, device=tensor.device) + tensor


def step_rnn(state_current: torch.Tensor, inputs: torch.Tensor, rnn: torch.nn.RNNBase) -> torch.Tensor:
    _, state_next = rnn(inputs, state_current)
    return state_next


def get_recurrent_jacobian(fp_state: torch.Tensor, rnn: torch.nn.RNNBase, inputs: torch.Tensor) -> torch.Tensor:
    _, jacobians = torch.autograd.functional.jacobian(rnn, (inputs, fp_state))
    input_jacobian, recurrent_jacobian = jacobians

    recurrent_jacobian = recurrent_jacobian.squeeze(0).squeeze(1).cpu().numpy()

    return recurrent_jacobian


def find_fp_from(
    initial_state: torch.Tensor,
    rnn: torch.nn.RNNBase,
    n_iters: int,
    noise_std: float,
    keep_optimisation_history: bool,
    inputs: torch.Tensor,
) -> FP:
    initial_state = add_noise(initial_state, std=noise_std)

    state: nn.Parameter = nn.Parameter(initial_state)

    optimiser: torch.optim.Optimizer = torch.optim.Adam([state], lr=0.1)
    scheduler: torch.optim.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=100)

    state_history: List[torch.Tensor] = [initial_state]
    q_history: List[float] = []

    rnn.flatten_parameters()
    rnn.train()

    for i in range(n_iters):
        state_next = step_rnn(state_current=state, inputs=inputs, rnn=rnn)

        q = 0.5 * torch.sum((state - state_next) ** 2, axis=1)

        q.backward()
        optimiser.step()
        optimiser.zero_grad()
        scheduler.step(q)

        # Keep the state history and speeds for visualisation
        if keep_optimisation_history:
            state_history.append(torch.clone(state).detach())  # TODO this should just be the parameter state

        q_history.append(q.detach().item())

    # Housekeeping - convert a list of tensors into a single tensor
    if keep_optimisation_history:
        state_history: torch.Tensor = torch.concat(state_history).detach().cpu()
    else:
        state_history: torch.Tensor = torch.concat([torch.clone(state)]).detach().cpu()

    recurrent_jacobian: torch.Tensor = get_recurrent_jacobian(state.detach(), rnn, inputs)

    rnn.eval()

    eigenvals, eigenvecs = np.linalg.eig(recurrent_jacobian)

    return FP(
        q_history=q_history,
        state_history=state_history,
        recurrent_jacobian=recurrent_jacobian,
        unsorted_jac_eigenvals=eigenvals,
        unsorted_jac_eigenvecs=eigenvecs,
    )


def find_fps(
    data: Run, n_inits: int, n_iters: int, noise_std: float, keep_optimisation_history: bool, inputs: torch.Tensor
) -> List[FP]:
    h_init_indexes: List[int] = np.random.randint(data.states.shape[0], size=n_inits)
    inits: List[torch.Tensor] = [
        data.states[i].unsqueeze(0) for i in h_init_indexes
    ]  # [15000, 128] -> [[1, 128], [1, 128]...]

    if inputs.shape[0] == 1:
        print('Input mode: using a single input')
        inputs: List[torch.Tensor] = [inputs for _ in h_init_indexes]  # [1, 128] -> [[1, 128], [1, 128]...]
    elif inputs.shape[0] == data.states.shape[0]:
        print('Input mode: matching input to corresponding state (for finding input-dependent fixed points)')
        inputs: List[torch.Tensor] = [
            inputs[i].unsqueeze(0) for i in h_init_indexes
        ]  # [15000, 128] -> [[1, 128], [1, 128]...]
    else:
        raise ValueError(f'Expected number of inputs ({inputs.shape}) to match {data.states.shape} or be 1')

    # Calls the find_fp_from function in parallel
    with tqdm_joblib(desc='FP finding', total=len(inits)):
        all_fps = joblib.Parallel(n_jobs=18)(
            joblib.delayed(find_fp_from)(init, data.rnn, n_iters, noise_std, keep_optimisation_history, inp)
            for init, inp in zip(inits, inputs)
        )
    return all_fps


DEFAULT_CACHE_NAME: str = 'fps_nb'


def find_fps_at_conditions(
    run: Run,
    conditions: List[str],
    fpf_n_inits: Optional[int] = 200,
    fpf_n_iters: Optional[int] = 5000,
    fpf_noise_std: Optional[float] = 0.05,
    base_context=None,
    cache_name=DEFAULT_CACHE_NAME,
) -> Dict[str, List[FP]]:
    from sf.analysis import prepare_input_for_obs

    if (run.milestone_path_override is not None) and cache_name == DEFAULT_CACHE_NAME:
        raise ValueError('You are finding fixed points at a milestone but did not provide a custom cache name')
    elif (run.milestone_path_override is None) and cache_name != DEFAULT_CACHE_NAME:
        print('Warning: you are using a cache path but using the default milestone path.')

    condition_generators: Dict[str, Callable[[], torch.Tensor]] = {
        #'blackbox_zerov': analysis.prepare_input_for_obs(data=run, velocity=0.0, indicator=False, blackbox=True, beacon=False, context=base_context),
        'blackbox_1v': lambda: prepare_input_for_obs(
            data=run, velocity=1.0, indicator=False, blackbox=True, beacon=False, context=base_context
        ),
        'blackbox_3v': lambda: prepare_input_for_obs(
            data=run, velocity=3.0, indicator=False, blackbox=True, beacon=False, context=base_context
        ),
        #'blackbox_2v': analysis.prepare_input_for_obs(data=run, velocity=2.0, indicator=False, blackbox=True, beacon=False, context=base_context),
        #    'middle_zerov': analysis.prepare_input_for_obs(data=run, velocity=0.0, indicator=False, blackbox=False, beacon=False, context=base_context),
        #    'middle_0p5v': analysis.prepare_input_for_obs(data=run, velocity=0.5, indicator=False, blackbox=False, beacon=False, context=base_context),
        #    'middle_1v': analysis.prepare_input_for_obs(data=run, velocity=1.0, indicator=False, blackbox=False, beacon=False, context=base_context),
        #    'middle_2v': analysis.prepare_input_for_obs(data=run, velocity=2.0, indicator=False, blackbox=False, beacon=False, context=base_context),
        'middle_5v': lambda: prepare_input_for_obs(
            data=run, velocity=5.0, indicator=False, blackbox=False, beacon=False, context=base_context
        ),
        'middle_4v': lambda: prepare_input_for_obs(
            data=run, velocity=4.0, indicator=False, blackbox=False, beacon=False, context=base_context
        ),
        'middle_3v': lambda: prepare_input_for_obs(
            data=run, velocity=3.0, indicator=False, blackbox=False, beacon=False, context=base_context
        ),
        'middle_2v': lambda: prepare_input_for_obs(
            data=run, velocity=2.0, indicator=False, blackbox=False, beacon=False, context=base_context
        ),
        'middle_1v': lambda: prepare_input_for_obs(
            data=run, velocity=1.0, indicator=False, blackbox=False, beacon=False, context=base_context
        ),
        'middle_0v': lambda: prepare_input_for_obs(
            data=run, velocity=0.0, indicator=False, blackbox=False, beacon=False, context=base_context
        ),
        **{
            f'middle_3v_ctx{ctx}': lambda: prepare_input_for_obs(
                data=run, velocity=3.0, indicator=False, blackbox=False, beacon=False, context=ctx
            )
            for ctx in range(8)
        },
        #    'middle_4v': analysis.prepare_input_for_obs(data=run, velocity=4.0, indicator=False, blackbox=False, beacon=False, context=base_context),
        #    'middle_5v': analysis.prepare_input_for_obs(data=run, velocity=5.0, indicator=False, blackbox=False, beacon=False, context=base_context),
        'indicator_0v': lambda: prepare_input_for_obs(
            data=run, velocity=0.0, indicator=True, blackbox=False, beacon=False, context=base_context
        ),
        'indicator_1v': lambda: prepare_input_for_obs(
            data=run, velocity=1.0, indicator=True, blackbox=False, beacon=False, context=base_context
        ),
        #    'beacon_zerov': analysis.prepare_input_for_obs(data=run, velocity=0.0, indicator=False, blackbox=False, beacon=True, context=base_context),
        #    'beacon_onev': analysis.prepare_input_for_obs(data=run, velocity=1.0, indicator=False, blackbox=False, beacon=True, context=base_context),
    }

    fps_at_conditions: Dict[str, List[FP]] = {}

    run_dir: Path = EXPERIMENTS / run.model_path / 'derivatives' / cache_name
    run_dir.mkdir(exist_ok=True, parents=True)

    for condition in conditions:
        print(f'Running condition {condition}')
        inputs: torch.Tensor = condition_generators[condition]()

        path: Path = run_dir / condition

        # TODO: use better caching method or disable entirely for reproduction. Unclear behaviour for user.
        if path.exists():
            print(f'{path} exists... Loading cached run')
            fps_at_condition: List[FP] = torch.load(path)
        else:
            fps_at_condition: List[FP] = find_fps(
                run,
                n_inits=fpf_n_inits,
                n_iters=fpf_n_iters,
                noise_std=fpf_noise_std,
                inputs=inputs,
                keep_optimisation_history=False,
            )
            torch.save(fps_at_condition, path)

        fps_at_conditions[condition] = fps_at_condition

    run.cpu()

    return fps_at_conditions
