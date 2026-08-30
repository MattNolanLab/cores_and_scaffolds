import contextlib
import enum
import itertools
import json
import logging
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Union

import gym
import joblib
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from gym import envs
from pydantic import BaseModel, validator
from sample_factory.enjoy import enjoy_with_data
from sample_factory.envs.env_utils import register_env
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.model.core import ModelCore
from sample_factory.model.decoder import MlpDecoder
from sf_examples.train_gym_env import parse_custom_args
from tqdm.autonotebook import tqdm

envs_by_speed: Dict[str, str] = {
    'Normal': 'LinearNavigationContextual12Static3-v2',
    'Rnd Slow': 'LinearNavigationContextual12Static3Speedslow-v2',
    'Single Nb Rnd Slow': 'LinearNavigationSingleNbContextual12Static3Speedslow-v2',
    'Rnd Fast': 'LinearNavigationContextual12Static3Speedfast-v2',
    'Half zero': 'LinearNavigationContextual12Static3Speedhalfzero-v2',
    **{
        name: f'LinearNavigationContextual12Static3Speed{speed_str}-v2'
        for speed_str, name in [
            (f'constant_{str(s).replace(".", "p")}', f'Constant {s}') for s in [1.0, 1.5, 2.5, 3.5, 4.5]
        ]
    },
}


class Run(BaseModel):
    states: torch.Tensor
    inputs: torch.Tensor
    locations: torch.Tensor
    episodes: torch.Tensor
    stops: torch.BoolTensor
    velocities: torch.Tensor
    timesteps: torch.Tensor
    trials: torch.Tensor
    trial_types: np.ndarray
    rnn: torch.nn.RNNBase
    model: ActorCritic
    rewards: torch.Tensor
    reward_avg: float
    model_path: Path
    video_path: Optional[Path]
    milestone_path_override: Optional[Path]
    env_name: str

    def __str__(self):
        return f'{self.env_name} {len(self.timesteps)}'

    def __repr__(self):
        return f'{self.env_name} {len(self.timesteps)}'

    def cpu(self):
        self.rnn.cpu()
        self.states = self.states.cpu()
        self.inputs = self.inputs.cpu()
        self.model = self.model.cpu()

    def cuda(self):
        self.rnn.cuda()
        self.states = self.states.cuda()
        self.inputs = self.inputs.cuda()
        self.model = self.model.cuda()

    @property
    def velocities_offset(self):
        return torch.roll(self.velocities, 1)

    @property
    def locations_offset(self):
        return torch.roll(self.locations, 1)

    @property
    def velocities_offset_2(self):
        return torch.roll(self.velocities, 2)

    def trial_type_mask(self, trial_type: Literal['b', 'nb']) -> torch.BoolTensor:
        return torch.Tensor(self.trial_types == trial_type).bool()

    def reward_avg_by_trial(self, trial_type: Literal['b', 'nb']) -> float:
        trial_type_mask: torch.Tensor = self.trial_type_mask(trial_type)

        rewards = []
        for episode in range(
            self.episodes.max().int()
        ):  # subtle: we do all but the last, to get complete episodes only
            episode_mask: torch.Tensor = self.episodes == episode
            r = sum(self.rewards[torch.logical_and(episode_mask, trial_type_mask)])

            rewards.append(r)

        return np.mean(rewards)

    @property
    def average_nb_stop_location(self) -> float:
        return torch.mean(self.locations[torch.logical_and(self.trial_type_mask('nb'), self.stops)]).item()

    def get_first_stop_locations(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        first_stops_b: List[float] = []
        first_stops_nb: List[float] = []
        first_stops_p: List[float] = []
        for episode in range(
            self.episodes.max().int()
        ):  # Ensures we don't include the last (possibly incomplete) episode
            for trial in range(
                self.trials.max().int() + 1
            ):  # Handles edge case where probe simulations only have 1 trial
                episode_trial_mask = get_episode_trial_mask(self, episode=episode, trial=trial)

                stops: torch.Tensor = self.stops[episode_trial_mask]
                locations: torch.Tensor = self.locations[episode_trial_mask]
                if any(stops):
                    stop_index = torch.where(stops)[0][0]
                    stop_location = locations[stop_index]

                    if self.trial_types[episode_trial_mask][0] == 'b':
                        first_stops_b.append(stop_location)
                    elif self.trial_types[episode_trial_mask][0] == 'nb':
                        first_stops_nb.append(stop_location)
                    elif self.trial_types[episode_trial_mask][0] == 'p':
                        first_stops_p.append(stop_location)
                    else:
                        raise NotImplementedError
                else:
                    pass

        return np.array(first_stops_b), np.array(first_stops_nb), np.array(first_stops_p)

    def iter_global_trial_masks(self) -> Iterator[torch.Tensor]:
        """Yields boolean masks for each episode x trial combination"""
        for episode in range(
            self.episodes.max().int()
        ):  # Ensures we don't include the last (possibly incomplete) episode
            for trial in range(
                self.trials.max().int() + 1
            ):  # Handles edge case where probe simulations only have 1 trial
                yield get_episode_trial_mask(self, episode=episode, trial=trial)

    def get_uncued_first_stop_locations(self) -> np.ndarray:
        return np.array(
            self.get_first_stop_locations()[1]
        )  # TODO backward compat because old version of func returns List[float]

    def get_mean_first_stop(self) -> Tuple[float, float, float]:
        a, b, c = self.get_first_stop_locations()
        return np.mean(a), np.mean(b), np.mean(c)

    @property
    def beaconed_reward_avg(self) -> float:
        return self.reward_avg_by_trial('b')

    @property
    def nonbeaconed_reward_avg(self) -> float:
        return self.reward_avg_by_trial('nb')

    @property
    def nb_stops(self) -> int:
        return torch.sum(self.stops[self.trial_type_mask('nb')]).item()

    @property
    def b_stops(self) -> int:
        return torch.sum(self.stops[self.trial_type_mask('b')]).item()

    @property
    def n_timesteps(self) -> int:
        return self.states.shape[0]

    def get_displacements(self, trial_type: Literal['b', 'nb']) -> 'RunDisplacementResults':
        from sf.attribution.disruption import calculate_run_displacements

        return calculate_run_displacements(self, trial_type)

    @property
    def rewarded_nb_trials_percentage(self) -> float:
        run: Run = self

        trial_outcomes: List[int] = []
        # TODO I'm sure there is a nice way to do this with boolean indexing
        for episode in run.episodes.unique():
            for trial in (run.trials[run.episodes == episode]).unique():
                mask: torch.Tensor = torch.logical_and(run.episodes == episode, run.trials == trial)
                _trial_types: np.ndarray = np.unique(run.trial_types[mask])
                assert len(np.unique(_trial_types)) == 1

                trial_type: str = _trial_types[0]
                assert run.trials[mask].unique() <= 6, (
                    'TODO: looking for high reward but this might be low if we have many trials!'
                )

                if trial_type == 'nb':
                    was_rewarded: int = int(run.rewards[mask].max() > 10.0)
                    trial_outcomes.append(was_rewarded)

        return float(np.mean(trial_outcomes) * 100)

    @validator('states')
    @classmethod
    def validate_states_shape(cls, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 2:
            raise ValueError(f'states has an incorrect number of dimensions (not 2): {states.shape}')
        return states

    @validator('video_path')
    @classmethod
    def validate_video_exists(cls, video_path: Optional[Path]) -> Path:
        if video_path is not None:
            assert video_path.exists(), 'Video path does not exist'
        return video_path

    class Config:
        arbitrary_types_allowed = True
        validate_all = True
        validate_assignments = True


def get_sorted_milestones(model_path: Path) -> List[Path]:
    if not model_path.exists():
        raise ValueError(f'Model path does not exist!\n{model_path}')

    return sorted(list((model_path / 'checkpoint_p0' / 'milestones').glob('checkpoint*.pth')))


def get_latest_checkpoint(model_path: Path) -> Path:
    """Excluding milestones!"""
    if not model_path.exists():
        raise ValueError(f'Model path does not exist!\n{model_path}')

    sorted_checkpoints: List[Path] = sorted(list((model_path / 'checkpoint_p0').glob('checkpoint_*_*.pth')))
    latest_checkpoint: Path = sorted_checkpoints[-1]
    print(
        f'Got checkpoints for {model_path.name}: {[c.name for c in sorted_checkpoints]}. Returning {latest_checkpoint.name}'
    )

    return latest_checkpoint


def extract_rnn(core: ModelCore) -> torch.nn.RNNBase:
    if isinstance(core, torch.nn.RNNBase):
        c = core
    else:
        c = core.core

    assert isinstance(c, torch.nn.RNNBase)
    return c


def run_model(
    model_path: Path,
    n_frames: int = 5_000,
    env_name: Optional[str] = None,
    load_checkpoint_kind: Literal['latest', 'best'] = 'latest',
    verbose: bool = False,
    milestone_path_override: Optional[Path] = None,
    milestone_index_override: Optional[int] = None,
    device: Literal['gpu', 'cpu'] = 'cpu',
    core_override: Optional[ModelCore | nn.Module] = None,
    decoder_override: Optional[MlpDecoder] = None,
    distribution_linear_override: Optional[torch.nn.Linear] = None,
    video_name: Optional[str] = None,
) -> List[Run]:
    register_custom_envs()

    from sf import EXPERIMENTS

    if str(EXPERIMENTS) not in str(model_path):
        print(f'Substituting {model_path} -> ', end='')
        model_path = EXPERIMENTS / model_path
        print(model_path)

    with open(model_path / 'config.json') as f:
        config = json.load(f)

    if not verbose:
        log = logging.getLogger('rl')
        log.setLevel(logging.CRITICAL)

    if env_name is None:
        env_name: str = config['env']
        if verbose:
            print(f'No env provided. Detected {env_name} from config')

    # Construct the argument variables
    argv: List[str] = [
        f'--env={env_name}',
        f'--max_num_frames={n_frames}',
        f'--train_dir={model_path.parent}',
        f'--experiment={model_path.name}',
        f'--device={device}',
        f'--load_checkpoint_kind={load_checkpoint_kind}',
    ]

    video_path: Optional[Path]
    if video_name is not None:
        argv.append('--save_video')
        argv.append(f'--video_name={video_name}')
        video_path = model_path / f'{video_name}.mp4'
    else:
        video_path = None

    cfg = parse_custom_args(evaluation=True, argv=argv)
    cfg.env = env_name

    cfg.no_render = True

    if milestone_path_override is not None:
        assert milestone_index_override is None
    elif milestone_index_override is not None:
        assert milestone_path_override is None
        milestones: List[Path] = get_sorted_milestones(model_path)
        milestone_path_override = milestones[milestone_index_override]
        print(f'Loaded milestone {milestone_index_override} out of {len(milestones)} milestones')

    status, avg_reward, data, model = enjoy_with_data(
        cfg,
        milestone_path_override=milestone_path_override,
        core_override=core_override,
        distribution_linear_override=distribution_linear_override,
        decoder_override=decoder_override,
    )

    if verbose:
        print(model)

    separate_actor_critic: bool = not config['actor_critic_share_weights']

    non_state_data: Dict[str, Any] = dict(
        locations=torch.Tensor([s['location'] for s in data['info_step']]),
        episodes=torch.Tensor([s['episode'] for s in data['info_step']]),
        stops=torch.Tensor([s['stopped'] for s in data['info_step']]).bool(),
        velocities=torch.Tensor([s['velocity'] for s in data['info_step']]),
        timesteps=torch.Tensor([s['timestep'] for s in data['info_step']]),
        trials=torch.Tensor([s['trial'] for s in data['info_step']]),
        trial_types=np.array([s['trial_type'] for s in data['info_step']], dtype=str),
        rewards=torch.Tensor(data['rewards']),
        reward_avg=avg_reward,
        model=model,
        model_path=model_path,
        video_path=video_path,
        milestone_path_override=milestone_path_override,
        env_name=env_name,
    )

    if separate_actor_critic:
        actor_states, critic_states = torch.cat(data['rnn_states']).chunk(2, dim=1)
        actor_inputs, critic_inputs = torch.cat(data['rnn_inputs']).chunk(2, dim=1)
        run_data = [
            Run(
                states=actor_states, inputs=actor_inputs, rnn=extract_rnn(model.actor_core.core).cpu(), **non_state_data
            ),
            Run(
                states=critic_states,
                inputs=critic_inputs,
                rnn=extract_rnn(model.critic_core.core).cpu(),
                **non_state_data,
            ),
        ]
    else:
        run_data = [
            Run(
                states=torch.cat(data['rnn_states']),
                inputs=torch.cat(data['rnn_inputs']),
                rnn=extract_rnn(model.core.core).cpu(),
                **non_state_data,
            )
        ]

    if not verbose:
        log.setLevel(logging.DEBUG)

    return run_data


class Action(enum.IntEnum):
    STOP = 0
    GO = 1


def make_nonvisual_gym_func(full_env_name, cfg=None, env_config=None, render_mode: Optional[str] = None):
    env = gym.make(full_env_name, render_mode=render_mode)
    return env


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, 'Error: Operation, ' + str(op) + ', must be dim3'
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[self.op[0]], obs_shape[self.op[1]], obs_shape[self.op[2]]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


def make_visual_gym_func(full_env_name, cfg=None, env_config=None, render_mode: Optional[str] = None):
    env = gym.make(full_env_name, render_mode=render_mode)
    obs_shape = env.observation_space.shape
    if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
        env = TransposeImage(env, op=[2, 0, 1])

    return env


def register_custom_envs():
    for env_name in envs.registry.keys():
        if ('LinearNavigation' in env_name) or ('SpatialMemory' in env_name) or ('FlipFlop' in env_name):
            register_env(env_name, make_visual_gym_func if 'Visual' in env_name else make_nonvisual_gym_func)


@contextlib.contextmanager
def tqdm_joblib(*args, **kwargs):
    """Context manager to patch joblib to report into tqdm progress bar
    given as argument"""

    tqdm_object = tqdm(*args, **kwargs)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def chunk_iter(iterable: Iterable, n: int = 2) -> Iterable[Iterable]:
    """Iterate chunkwise over an iterable, returning n elements in each chunk"""
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args)


def is_monotonic_increasing(a: np.array) -> bool:
    return np.all(np.diff(a) >= 0)


def get_symmetric_top(tensor: torch.Tensor, n: int) -> torch.Tensor:
    """Returns top n (where n is composed half of smallest half of largest)"""
    assert (n % 2) == 0

    a = torch.Tensor(tensor)
    _, smallest = torch.topk(a, n // 2, largest=False)
    _, largest = torch.topk(a, n // 2, largest=True)

    return torch.cat([smallest, largest])


def get_random_control_indices(run: Run, n: int) -> torch.Tensor:
    n_hidden = run.rnn.hidden_size
    indices = torch.Tensor(np.random.choice(range(n_hidden), size=n, replace=False)).int()

    return indices


def get_top_indices(arr: torch.Tensor, n: int, positive: bool) -> torch.Tensor:
    """Positive TRUE -> aka sort descending, returning the corresponding n indices for arr"""
    arr = arr.numpy()
    assert arr.ndim == 1
    assert n <= len(arr)

    sorted_indices: np.ndarray = np.argsort(arr)

    if positive:
        sorted_indices: np.ndarray = sorted_indices[
            ::-1
        ].copy()  # https://stackoverflow.com/questions/72550211/valueerror-at-least-one-stride-in-the-given-numpy-array-is-negative-and-tensor

    return torch.Tensor(sorted_indices[:n]).int()


def reverse_tensor(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.ndim == 1

    return torch.flip(tensor, dims=[0])


def get_closest_i(tensor: Union[torch.Tensor, np.ndarray], v, verbose: bool = False) -> int:
    """Returns the index of the value closest to v"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.numpy()

    diffs = np.abs(tensor - v)
    index = np.argmin(diffs)

    if verbose:
        print(f'Index matching delta: {diffs[index]}')

    return index


def matrix_to_df(m: torch.Tensor, value_name: str, row_name: str, col_name: str) -> pl.DataFrame:
    """Converts a matrix to a tidy dataframe"""
    m: torch.Tensor = m.numpy()

    values = m.flatten()

    rows, cols = np.indices(m.shape)

    df = pl.DataFrame({value_name: values, row_name: rows.flatten(), col_name: cols.flatten()})

    return df


def get_episode_trial_mask(run: Run, episode: int, trial: int) -> torch.Tensor:
    episode_mask: torch.BoolTensor = run.episodes == episode
    trial_mask: torch.BoolTensor = run.trials == trial

    return torch.logical_and(episode_mask, trial_mask)


def cache_derivative(item_name: str) -> Callable:
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            run: Run = args[0]
            assert len(args) == 1
            assert len(kwargs) == 0

            derivatives_folder: Path = run.model_path / 'derivatives'
            derivatives_folder.mkdir(exist_ok=True)

            item_path: Path = derivatives_folder / item_name
            if item_path.exists():
                print(f'Cached derivative exists, loading: {item_path}')
                return torch.load(item_path)
            else:
                print(f'Cached derivative does not exist: {item_path}')
                item = f(*args, **kwargs)
                print(f'Saving derivative to: {item_path}')
                torch.save(item, item_path)
                return item

        return wrapper

    return decorator


def disable_jit():
    torch.jit._state.disable()


def merge_dicts(ds: List[Dict[Any, Any]]) -> Dict[Any, Any]:
    new: Dict[Any, Any] = {}
    for d in ds:
        new.update(d)
    return new


def rank_truncate(m: torch.Tensor, k: int) -> torch.Tensor:
    """Truncates the rank to k using svd"""
    U, S, Vh = torch.linalg.svd(m, full_matrices=False)
    m2 = sum([torch.outer(U[:, i], Vh[i]) * S[i] for i in range(k)])
    m1 = (U[:, :k] * S[:k]) @ Vh[:k]
    # print(m1)
    # print(m2)

    torch.testing.assert_close(m2, m1)
    return m1


def get_singular_modes(m: torch.Tensor) -> List[torch.Tensor]:
    m = m.detach()
    U, S, Vh = torch.linalg.svd(m, full_matrices=False)

    return [torch.outer(U[:, i], Vh[i, :]) * S[i] for i in range(m.shape[0])]
