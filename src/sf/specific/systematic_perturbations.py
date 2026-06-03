"""
Analysis for figure 3 - perturbations of neurons by various criteria
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from tqdm.autonotebook import tqdm

import sf.analysis as analysis
import sf.utils as utils
from sf.attribution import encoding_attributions
from sf.perturbations import ScaleOutputWrapper
from sf.utils import Run

CONDITION_COLOURS = {
    'PC1': '#1f77b4',
    'PC2': '#ff7f0e',
    'PC3': '#2ca02c',
    'Random': '#d62728',
    'Speed Attr.': '#9467bd',
    'Location r': '#8c564b',
    'Speed r': '#e377c2',
    'Disruption': 'purple',
    'Displacement': 'orange',
}


CONDITION_LABELS = {
    'pca0': 'PC1',
    'pca1': 'PC2',
    'pca2': 'PC3',
    'random': 'Random',
    'speed_attribution': 'Speed Attr.',
    'location_correlation': 'Location r',
    'speed_correlation': 'Speed r',
    'disruption': 'Disruption',
    'displacement': 'Displacement',
}


def average_random_seeds(df: pd.DataFrame, on: str) -> pd.DataFrame:
    # Filter out rows containing 'random' in indices
    random_df = df[df['indices'].str.startswith('+random_', na=False)].copy()

    # Group by number of perturbed neurons and calculate means
    grouped_zeroing_df = (
        random_df.groupby([on, 'descending'])
        .agg(
            {
                'nb_stops': 'mean',
                'nb_displacement_mean': 'mean',
                'b_displacement_mean': 'mean',
                'nb_displacement_sd': 'mean',
                'b_displacement_sd': 'mean',
                'b_stops': 'mean',
                'mean_nb_stop': 'mean',
                'mean_first_nb_stop': 'mean',
                'mean_first_b_stop': 'mean',
                'nb_reward_perc': 'mean',
                'b_reward_perc': 'mean',
                'n_frames': 'mean',
            }
        )
        .reset_index()
    )
    # TODO we're leaking the logic of indices[1:], descending etc, here.... outside of the loading functions
    grouped_zeroing_df['descending'] = True
    grouped_zeroing_df['indices'] = '+random'
    grouped_zeroing_df['condition'] = 'random'
    ascending_df = grouped_zeroing_df.copy()
    ascending_df['descending'] = False
    ascending_df['indices'] = '-random'
    ascending_df['condition'] = 'random'

    df_norandom = df[~df['indices'].str.contains('random', na=False)]
    return pd.concat([grouped_zeroing_df, ascending_df, df_norandom], ignore_index=True)


def load_zeroing_df(model_path: Path) -> pd.DataFrame:
    data: List[Dict[str, Dict[int, Run]]] = [
        torch.load(p) for p in (model_path / 'derivatives/single').glob('zeroing*.pkl')
    ]
    data: Dict[str, Dict[int, Run]] = utils.merge_dicts(data)

    records: List[Dict[str, Any]] = []

    for indices, runs in data.items():
        for n, run in runs.items():
            records.append(
                {
                    'indices': indices,
                    'descending': {'+': True, '-': False}[indices[0]],  # +pca -> positive. -pca -> negative
                    'condition': indices[1:],
                    'n_perturbed': n,
                    'nb_stops': run.nb_stops,
                    'b_stops': run.b_stops,
                    'nb_displacement_mean': np.mean(run.get_displacements('nb').peak_locations),
                    'b_displacement_mean': np.mean(run.get_displacements('b').peak_locations),
                    'nb_displacement_sd': np.std(run.get_displacements('nb').peak_locations),
                    'b_displacement_sd': np.std(run.get_displacements('b').peak_locations),
                    'mean_nb_stop': run.average_nb_stop_location,
                    'mean_first_nb_stop': run.get_mean_first_stop()[1],
                    'mean_first_b_stop': run.get_mean_first_stop()[0],
                    'nb_reward_perc': run.reward_avg_by_trial('nb') * 2,
                    'b_reward_perc': run.reward_avg_by_trial('b') * 2,
                    'n_frames': run.n_timesteps,
                }
            )

    df = pd.DataFrame(records)
    df = average_random_seeds(df, on='n_perturbed')  # Converts random_0, random_1, etc to random
    df['condition'] = df['condition'].map(CONDITION_LABELS)

    return df


def load_scaling_df(model_path: Path) -> pd.DataFrame:
    data: List[Dict[str, Dict[float, Run | None]]] = [
        torch.load(p) for p in (model_path / 'derivatives/single').glob('scaling*.pkl')
    ]
    data: Dict[str, Dict[float, Run | None]] = utils.merge_dicts(data)

    records: List[Dict[str, Any]] = []
    for indices, runs in tqdm(data.items()):
        for s, run in runs.items():
            if run is None:
                records.append(
                    {
                        'indices': indices,
                        'descending': {'+': True, '-': False}[indices[0]],  # +pca -> positive. -pca -> negative
                        'condition': indices[1:],
                        'scaling_factor': s,
                        'nb_displacement_mean': np.nan,
                        'b_displacement_mean': np.nan,
                        'nb_displacement_sd': np.nan,
                        'b_displacement_sd': np.nan,
                        'nb_stops': np.nan,
                        'b_stops': np.nan,
                        'mean_nb_stop': np.nan,
                        'mean_first_nb_stop': np.nan,
                        'mean_first_b_stop': np.nan,
                        'nb_reward_perc': np.nan,
                        'b_reward_perc': np.nan,
                        'n_frames': np.nan,
                    }
                )

            else:
                records.append(
                    {
                        'indices': indices,
                        'descending': {'+': True, '-': False}[indices[0]],  # +pca -> positive. -pca -> negative
                        'condition': indices[1:],
                        'scaling_factor': s,
                        'nb_stops': run.nb_stops,
                        'b_stops': run.b_stops,
                        'nb_displacement_mean': np.mean(run.get_displacements('nb').peak_locations),
                        'b_displacement_mean': np.mean(run.get_displacements('b').peak_locations),
                        'nb_displacement_sd': np.std(run.get_displacements('nb').peak_locations),
                        'b_displacement_sd': np.std(run.get_displacements('b').peak_locations),
                        'mean_nb_stop': run.average_nb_stop_location,
                        'mean_first_nb_stop': run.get_mean_first_stop()[1],
                        'mean_first_b_stop': run.get_mean_first_stop()[0],
                        'nb_reward_perc': run.reward_avg_by_trial('nb') * 2,
                        'b_reward_perc': run.reward_avg_by_trial('b') * 2,
                        'n_frames': run.n_timesteps,
                    }
                )
    df = pd.DataFrame(records)
    df = average_random_seeds(df, on='scaling_factor')
    df['condition'] = df['condition'].map(CONDITION_LABELS)
    return df


# @utils.cache_derivative('indices')
def get_indices(run: Run) -> Dict[str, torch.Tensor]:
    n_timesteps_required: int = 5_000
    if run.n_timesteps < n_timesteps_required:
        raise ValueError(f'Timesteps should be > {n_timesteps_required} for accurate metrics')

    return {
        **{f'pca{pc}': torch.Tensor(a_pca_contribution(run)[pc]) for pc in range(3)},
        **{f'random_{r}': torch.Tensor(utils.get_random_control_indices(run, run.rnn.hidden_size)) for r in range(10)},
        'location_correlation': torch.Tensor(a_location_correlations(run)),
        'speed_attribution': torch.Tensor(velocity_attributions_middle(run)),
        'speed_correlation': torch.Tensor(a_velocity_correlations(run)),
    }


def velocity_attributions_middle(run: Run) -> np.ndarray:
    ctxs_input = [0] * 12
    ctxs_input[3] = 1
    return encoding_attributions(
        baselines=torch.Tensor([[0, 0, 0, 0] + [0] * 12]),
        inputs=torch.Tensor([[1, 0, 0, 0] + ctxs_input]),
        run=run,
        n_steps=50,
    )[:, 0]


def run_perturbations(
    run: Run,
    scaling_factor: float,
    indices: torch.Tensor,
    descending: bool,
    ns: Tuple[int, ...],  # N perturbed
) -> Dict[int, Run]:
    """Runs perturbations with iteratively ascending/descending n sorted indices"""

    runs: Dict[int, Run] = {}
    for n in tqdm(ns):
        core = ScaleOutputWrapper(
            run.model.core, indices=utils.get_top_indices(indices, n, descending), scaling_factor=scaling_factor
        )

        runs[n] = analysis.run_model(run.model_path, core_override=core)[0]

    return runs


def run_perturbations_scaling(
    run: Run, indices: torch.Tensor, n_perturb: int, scaling_factors: Tuple[float, ...], descending: bool
) -> Dict[float, Run | None]:
    """Runs perturbations with iteratively descending n sorted indices"""

    runs: Dict[float, Run | None] = {}
    for f in tqdm(scaling_factors):
        core = ScaleOutputWrapper(
            run.model.core, indices=utils.get_top_indices(indices, n_perturb, descending), scaling_factor=f
        )

        try:
            runs[f] = analysis.run_model(run.model_path, core_override=core, n_frames=20_000)[0]
        except RuntimeError as e:
            if 'probability tensor contains either `inf`, `nan` or element < 0' in str(e):
                print(f'Encountered probability tensor error for {f}, {e}')
                runs[f] = None
            else:
                raise e

    return runs


"""
Perturbation criteria
"""


def a_pca_contribution(run: Run, n_components: int = 3) -> np.ndarray:
    arr = analysis.fit_pca(run, n_components=n_components).components_

    assert arr.shape == (n_components, run.rnn.hidden_size)

    return arr


def a_velocity_correlations(run: Run) -> np.ndarray:
    def correlation_analysis(behavior, neural_responses):
        correlation_coefficient, p_value = pearsonr(behavior, neural_responses)
        return correlation_coefficient, p_value

    def get_outbound_region(run: Run, neuron_i: int) -> Tuple[np.ndarray, np.ndarray]:
        location_mask: torch.Tensor = torch.logical_and(run.locations > 31, run.locations < 89)
        action_mask: torch.Tensor = torch.logical_not(run.stops)
        mask = torch.logical_and(location_mask, action_mask)

        velocities: np.ndarray = run.velocities_offset[mask].numpy()
        states: np.ndarray = run.states[mask, neuron_i].numpy()

        return velocities, states

    velocity_correlations: List[float] = []
    for neuron_i in range(run.rnn.hidden_size):
        velocities, states = get_outbound_region(run, neuron_i)
        correlation_coefficient, p_value = correlation_analysis(velocities, states)
        velocity_correlations.append(correlation_coefficient)

    arr = np.array(velocity_correlations)
    if np.any(np.isnan(arr)):
        print('Warning - some NaN correlation coefficients. Setting to 0.')
    arr[np.isnan(arr)] = 0

    return arr


def a_location_correlations(run: Run, rz_end: int = 93, suppress_warnings: bool = False) -> np.ndarray:
    if (rz_end % 10) != 0:
        if not suppress_warnings:
            print(f'RZ end is {rz_end} - did you forget to offset this by 1?')

    def correlation_analysis(behavior, neural_responses):
        correlation_coefficient, p_value = pearsonr(behavior, neural_responses)
        return correlation_coefficient, p_value

    def get_outbound_region_locations(run: Run, neuron_i: int) -> Tuple[np.ndarray, np.ndarray]:
        location_mask: torch.Tensor = torch.logical_and(run.locations_offset > 35, run.locations_offset < (rz_end - 5))
        action_mask: torch.Tensor = torch.logical_not(run.stops)
        mask = torch.logical_and(location_mask, action_mask)

        locations: np.ndarray = run.locations_offset[mask].numpy()
        states: np.ndarray = run.states[mask, neuron_i].numpy()

        return locations, states

    location_correlations: List[float] = []
    for neuron_i in range(run.rnn.hidden_size):
        locations, states = get_outbound_region_locations(run, neuron_i)
        correlation_coefficient, p_value = correlation_analysis(locations, states)
        location_correlations.append(correlation_coefficient)

    arr = np.array(location_correlations)
    if np.any(np.isnan(arr)):
        if not suppress_warnings:
            print('Warning - some NaN correlation coefficients. Setting to 0.')
    arr[np.isnan(arr)] = 0

    return arr
