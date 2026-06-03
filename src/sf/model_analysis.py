import json
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import sf.plots as plots
from sf.utils import Run, get_closest_i, get_sorted_milestones


class ModelStatus(Enum):
    SUCCESS = auto()
    INTERRUPTED = auto()
    NAN_ERROR = auto()


def compact_name(model_name: str) -> str:
    params: Dict[str, str] = extract_params(model_name)

    n_neurons: str = params['rnn_size']
    mappings: Dict[str, Callable[[str], str]] = {
        'rnn_type': lambda f: f'{n_neurons}-' + {'rnn_relu_bias': 'ReLU', 'rnn_tanh_bias': 'Tanh'}[f],
        'encoder_mlp_layers': lambda f: {'': '0', '512': '1', '512_512_512_512': '4'}[f] + 'E',
        'decoder_mlp_layers': lambda f: {'': '0', '512': '1', '512_512_512_512': '4'}[f] + 'D',
        'seed': lambda f: 'S' + f,
    }

    t: str = ' '.join(mappings[param](value) for param, value in params.items() if param in mappings)

    return t


def extract_params(m: str) -> Dict[str, Any]:
    assert isinstance(m, str)

    items: List[str] = m.strip('__').split('____')[1:]  # Ignore the first because it's the group name (w12_sweep)

    params: Dict[str, Any] = {pair[0]: pair[1] for pair in [i.split('==') for i in items]}

    return params


def model_status(model_path: Path) -> ModelStatus:
    log_file: Path = model_path / 'sf_log.txt'
    if not log_file.exists():
        return ModelStatus.INTERRUPTED

    with open(log_file) as f:
        log: str = f.read()

    if 'RuntimeError: probability tensor contains either `inf`, `nan` or element < 0' in log:
        return ModelStatus.NAN_ERROR
    elif 'InferenceWorker_p0-w0 profile tree view' in log:
        return ModelStatus.SUCCESS
    elif ('Error:' in log) or ('Exception: ' in log):
        raise NotImplementedError(f'Unhandled error in log {model_path}')
    else:
        return ModelStatus.INTERRUPTED


def count_completions(model_path: Path) -> int:
    with open(model_path / 'sf_log.txt') as f:
        log: str = f.read()

    lines: List[str] = log.split('\n')
    completion_pattern: str = 'InferenceWorker_p0-w0 profile tree view'
    return sum([completion_pattern in line for line in lines])


def read_eventfile(model_path: Path, tag) -> pd.DataFrame:
    event_accumulator = EventAccumulator(str(model_path / '.summary/0'))
    event_accumulator.Reload()

    events = event_accumulator.Scalars(tag)

    x: List[int] = [e.step for e in events]
    value: List[int] = [e.value for e in events]

    return pd.DataFrame.from_dict(dict(x=x, value=value))


def get_training_curve(model: Path) -> pd.DataFrame:
    df: pd.DataFrame = read_eventfile(model, 'reward/reward')

    print(
        f'Training curve has {len(np.unique(df.x))}/{len(df.x)} unique values. Picking lowest value for duplicate x-values'
    )

    df = df.loc[df.groupby('x')['value'].idxmin()]

    return df


def get_model_env(model: Path):
    config_path: Path = model / 'config.json'
    with open(config_path) as f:
        config: Dict = json.load(f)
    return config['env']


def get_training_curve_across_episodes(model: Path, trial_multiplier: int = 6) -> pd.DataFrame:
    df: pd.DataFrame = read_eventfile(model, 'reward/reward_across_episodes')
    df['trial'] = df['x'] * trial_multiplier

    return df


def plot_training_curve(model: Path, across_episodes=False, color='black', alpha: float = 1.0) -> None:
    if across_episodes:
        df: pd.DataFrame = get_training_curve_across_episodes(model)
        sns.lineplot(df, x='trial', y='value', color=color, alpha=alpha)
        plt.xlabel('Episode')
    else:
        df: pd.DataFrame = get_training_curve(model)
        sns.lineplot(df, x='x', y='value', color=color, alpha=alpha)
        plt.xlabel('Training step')
    plt.ylabel('Reward')


def plot_granular_training_curve(
    model: Path,
    episodes_to_trials_multiplier: Optional[int] = None,
    across_episodes=False,
    alpha: float = 1.0,
    plot_nb: bool = True,
    plot_b: bool = True,
    b_colour: str = 'blue',
    nb_colour: str = 'black',
) -> None:
    if across_episodes:
        if episodes_to_trials_multiplier is None:
            raise ValueError('Specify a multiplier to convert episodes to trial numbers')
        elif (episodes_to_trials_multiplier < 1) or not (isinstance(episodes_to_trials_multiplier, int)):
            raise ValueError('Multiplier should be nonzero positive integer')
        reward_data: Dict = get_granular_reward_data(model)
    else:
        raise NotImplementedError

    if across_episodes:
        if plot_b:
            plt.plot(
                reward_data['episode'] * episodes_to_trials_multiplier,
                np.array(reward_data['b']) * 2,
                '-',
                color=b_colour,
                label='Beaconed',
                alpha=alpha,
            )
        if plot_nb:
            plt.plot(
                reward_data['episode'] * episodes_to_trials_multiplier,
                np.array(reward_data['nb']) * 2,
                '-',
                color=nb_colour,
                label='Non-beaconed',
                alpha=alpha,
            )
    else:
        if plot_nb:
            plt.plot(
                reward_data['timestep'],
                np.array(reward_data['nb']) * 2,
                '-',
                color=nb_colour,
                label='Non-beaconed',
                alpha=alpha,
            )
        if plot_b:
            plt.plot(
                reward_data['timestep'],
                np.array(reward_data['b']) * 2,
                '-',
                color=b_colour,
                label='Beaconed',
                alpha=alpha,
            )


def get_milestone_index_closest_to_episode(episode: int, model: Path) -> int:
    milestones: List[Path] = get_sorted_milestones(model)
    milestone_timesteps: List[int] = [steps_at_milestone(m) for m in milestones]

    episode_to_timestep_df: pd.DataFrame = read_eventfile(model, 'episode/episode')
    i = get_closest_i(episode_to_timestep_df['value'].to_numpy(), episode, verbose=True)
    timestep = episode_to_timestep_df.loc[i, 'x']

    milestone_i = get_closest_i(np.array(milestone_timesteps), timestep, verbose=True)
    return milestone_i


def steps_at_milestone(milestone_path: Path) -> int:
    milestone_training_steps: int = int(milestone_path.with_suffix('').name.split('_')[2])
    return milestone_training_steps


def get_granular_reward_data(model_path: Path) -> Dict[str, np.ndarray]:
    bnb_df = read_eventfile(model_path, 'policy_stats/avg_all_across_episodes')

    offset: int = 3

    nb_df: pd.DataFrame = read_eventfile(model_path, 'policy_stats/avg_nb_across_episodes')
    b_df: pd.DataFrame = read_eventfile(model_path, 'policy_stats/avg_b_across_episodes')

    assert np.allclose(bnb_df['x'].to_numpy(), b_df['x'].to_numpy())
    assert np.allclose(bnb_df['x'].to_numpy(), nb_df['x'].to_numpy())

    return {
        'nb': nb_df['value'].to_numpy()[offset:],
        'b': b_df['value'].to_numpy()[offset:],
        'bnb': bnb_df['value'].to_numpy()[offset:],
        'episode': bnb_df['x'].to_numpy()[
            offset:
        ],  # Doesn't matter which df we use for the x axis. They should be the same
    }


def training_curve_with_episodes(run: Run, selected_episodes: List[int]) -> List[int]:
    assert selected_episodes[-1] == -1

    selected_milestone_indices: List[int] = [
        get_milestone_index_closest_to_episode(e, run.model_path) if e != -1 else -1 for e in selected_episodes
    ]
    episodes_to_trials_multiplier: int = 6

    plot_granular_training_curve(run.model_path, episodes_to_trials_multiplier, across_episodes=True)

    vline_colour: str = '#cb181d'
    for e in selected_episodes[:-1]:
        plt.axvline(e * episodes_to_trials_multiplier, color=vline_colour)

    xlim: float = plt.gca().get_xlim()[1]
    plt.axvline(xlim * 0.95, color=vline_colour)

    plots.set_xaxis_engineering_labels()
    plots.set_xaxis_n_ticks(4)

    plt.xlabel('Training trials')
    plt.ylabel('% of total reward')

    return selected_milestone_indices
