import enum
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from sklearn.decomposition import PCA

import sf.analysis as analysis
from sf.analysis import Run
from sf.plots import make_panel_fig


class ClampMode(enum.Enum):
    PRE = 'pre'
    POST = 'post'
    BOTH = 'both'


def analyse_pc_clamp_multi(
    run: Run,
    values_by_pc: Dict[int, float],
    pca: PCA,
    plot: bool = True,
    clamp_mode: ClampMode = ClampMode.BOTH,
    n_frames: int = 5000,
) -> Run:
    """Clamp multiple PCs"""
    modified_rnn: MultiPCClampedRNN = MultiPCClampedRNN(values_by_pc, pca=pca, core=run.rnn, mode=clamp_mode)

    perturbed_run: Run = analysis.run_model(
        run.model_path, core_override=modified_rnn, env_name=run.env_name, n_frames=n_frames
    )[0]
    if plot:
        plot_results(run, perturbed_run)

    return perturbed_run


def reconstruct_vector_multi(vector: torch.Tensor, values_by_pc: Dict[int, float], pca: PCA) -> torch.Tensor:
    assert vector.ndim == 1

    pcs: np.ndarray = pca.transform(
        rearrange(vector, 'n_neurons -> 1 n_neurons')
    )  # [n_components, 1] (reshape to handle single sample)
    # print(f'Overwriting PC{pc_i} {pcs[0, pc_i]} -> {pc_value:.1f}')

    if len(values_by_pc) == 0:
        raise ValueError

    for pc_i, pc_value in values_by_pc.items():
        pcs[0, pc_i] = pc_value

    reconstructed = pca.inverse_transform(pcs)  # [1, 512]
    return torch.Tensor(rearrange(reconstructed, '1 n_neurons -> n_neurons'))


class MultiPCClampedRNN(nn.Module):
    def __init__(
        self, values_by_pc: Dict[int, float], pca: PCA, core: nn.RNN, mode: ClampMode = ClampMode.BOTH
    ) -> None:
        super().__init__()

        self.core = core

        self.pca: PCA = pca
        self.values_by_pc: Dict[int, float] = values_by_pc

        self.mode: ClampMode = mode

    def forward(self, head_output, rnn_states) -> Tuple[torch.Tensor, torch.Tensor]:
        # Head output: [1, 1, 4]
        # RNN states: [1, 1, 512]
        if self.mode in [ClampMode.BOTH, ClampMode.PRE]:
            rnn_states[0, 0, :] = reconstruct_vector_multi(rnn_states[0, 0, :], self.values_by_pc, self.pca)

        x, new_rnn_states = self.core(head_output, rnn_states)

        # outputs: [1, 1, 512]
        assert torch.all(x == new_rnn_states)  # Sanity check

        if self.mode in [ClampMode.BOTH, ClampMode.POST]:
            x[0, 0, :] = reconstruct_vector_multi(x[0, 0, :], self.values_by_pc, self.pca)

        new_rnn_states = x

        # if self.pc_i == 0:
        #    print(self.pca.transform(x[0, 0].reshape(1, -1)).T[0])

        return x, new_rnn_states


def reconstruct_vector(vector: torch.Tensor, pc_i: int, pc_value: float, pca: PCA) -> torch.Tensor:
    assert vector.ndim == 1

    pcs: np.ndarray = pca.transform(
        rearrange(vector, 'n_neurons -> 1 n_neurons')
    )  # [n_components, 1] (reshape to handle single sample)
    # print(f'Overwriting PC{pc_i} {pcs[0, pc_i]} -> {pc_value:.1f}')

    if pc_i != -1:  # Reconstruct without modification  # TODO: make this clearer
        pcs[0, pc_i] = pc_value

    reconstructed = pca.inverse_transform(pcs)  # [1, 512]
    return torch.Tensor(rearrange(reconstructed, '1 n_neurons -> n_neurons'))


class ClampedRNN(nn.Module):
    def __init__(self, pc_i: int, pc_value: float, pca: PCA, core: nn.RNN) -> None:
        super().__init__()

        self.core = core

        self.pca: PCA = pca
        self.pc_i: int = pc_i
        self.pc_value: float = pc_value

    def forward(self, head_output, rnn_states) -> Tuple[torch.Tensor, torch.Tensor]:
        # Head output: [1, 1, 4]
        # RNN states: [1, 1, 512]
        rnn_states[0, 0, :] = reconstruct_vector(rnn_states[0, 0, :], self.pc_i, self.pc_value, self.pca)
        x, new_rnn_states = self.core(head_output, rnn_states)

        # outputs: [1, 1, 512]
        assert torch.all(x == new_rnn_states)  # Sanity check

        x[0, 0, :] = reconstruct_vector(x[0, 0, :], self.pc_i, self.pc_value, self.pca)
        new_rnn_states = x

        # if self.pc_i == 0:
        #    print(self.pca.transform(x[0, 0].reshape(1, -1)).T[0])

        return x, new_rnn_states


def analyse_pc_inclusions(run: Run, n_pcs: int, plot: bool = True, env_name: Optional[str] = None) -> Run:
    """Given 'n_pcs' principal components, what is the activity like?"""
    pca: PCA = analysis.fit_pca(run, n_components=n_pcs)

    modified_rnn: ClampedRNN = ClampedRNN(pc_i=-1, pc_value=0.0, pca=pca, core=run.rnn)

    perturbed_run: Run = analysis.run_model(
        run.model_path,
        core_override=modified_rnn,
        env_name=run.env_name if env_name is None else env_name,
        n_frames=5000,
    )[0]
    if plot:
        plot_results(run, perturbed_run)

    return perturbed_run


def analyse_pc_clamp(
    run: Run, pc_i: int, pc_value: float = 0, n_pcs: int = 15, plot: bool = True, env_name: Optional[str] = None
) -> Run:
    """Clamp a given PC to a fixed value"""
    pca: PCA = analysis.fit_pca(run, n_components=n_pcs)

    modified_rnn: ClampedRNN = ClampedRNN(pc_i=pc_i, pc_value=pc_value, pca=pca, core=run.rnn)

    perturbed_run: Run = analysis.run_model(
        run.model_path,
        core_override=modified_rnn,
        env_name=run.env_name if env_name is None else env_name,
        n_frames=5000,
    )[0]
    if plot:
        plot_results(run, perturbed_run)

    return perturbed_run


def plot_results(original_run: Run, perturbed_run: Run) -> None:
    pca = analysis.fit_pca(original_run, n_components=3)

    n_pcs_to_plot: int = 3
    n_cols: int = 2 + n_pcs_to_plot

    make_panel_fig(cols=n_cols)

    plt.subplot(1, n_cols, 1)
    analysis.plot_pca_mpl_2d(original_run, pca=pca)
    analysis.plot_trajectory_2d(
        perturbed_run,
        trajectory_states=perturbed_run.states[
            torch.logical_and(perturbed_run.episodes == 0, perturbed_run.trials == 0)
        ],
        pca=pca,
    )

    for i in range(3):
        plt.subplot(1, n_cols, i + 2)
        z: np.ndarray = pca.transform(perturbed_run.states).T[i]
        plt.title(f'PC{i}')
        plt.ylabel(f'PC{i} output')
        plt.xlabel('Speed')
        plt.scatter(perturbed_run.velocities_offset, z)

    plt.subplot(1, n_cols, n_cols)
    analysis.plot_stops(perturbed_run)
    plt.title(f'Reward: NB: {perturbed_run.nonbeaconed_reward_avg:.1f}, B: {perturbed_run.beaconed_reward_avg:.1f}')
