from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

import cebra
import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns
import torch
from einops import rearrange
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.actor_critic import ActorCritic
from sklearn.decomposition import PCA
from tqdm.autonotebook import tqdm

import sf.plots as plots
from sf import EXPERIMENTS
from sf.fixedpoints import FP, calculate_trial_speeds, filter_fps

# from cluster_analysis import * # TODO
from sf.plots import strip_plot_axes
from sf.utils import Run, run_model

sns.set_style('white')

BASE_FIGURE_PATH: Path = Path('/code/images')  # TODO refactor

DEFAULT_RZ_MIN: int = 95
DEFAULT_RZ_MAX: int = 105

REWARD_THRESHOLD: float = 95

MIDDLE_CONTEXT: int = 3


def get_run(model_path: Path) -> Run:
    return torch.load(model_path / 'derivatives/run.pkl')


def get_hrun(model_path: Path) -> Run:
    return torch.load(model_path / 'derivatives/run_high.pkl')


def get_valid_models() -> List[Path]:
    """Returns trained models that have adequate performance"""
    return [m for m in tqdm(get_all_models()) if torch.load(m / 'derivatives/run.pkl').reward_avg > REWARD_THRESHOLD]


def get_exemplar_model() -> Path:
    return get_curriculum_exemplar().pre


def get_all_models() -> List[Path]:
    return list((EXPERIMENTS / 'exp_001/models').glob('w90_papersweep*'))


def get_fig_dir(s: str) -> Path:
    figure_path = BASE_FIGURE_PATH / s
    figure_path.mkdir(exist_ok=True, parents=True)
    print(f'Figure path configured: {figure_path}')
    return figure_path


class ModelCurriculumPaths:
    def __init__(self, model_id: str):
        base: Path = Path(EXPERIMENTS / 'exp_001/models')

        self.pre: Path = base / f'{model_id}_pre'
        # TODO: refactor - legacy data structure that used to contain other model paths


def get_curriculum_exemplar() -> ModelCurriculumPaths:
    return ModelCurriculumPaths('w64_repro_seed__0')


def get_curriculum_exemplar_params() -> Dict[str, Any]:
    return {
        'n_perturb': 8,
        'episodes': [0, 1500, -1],  # -1 dummy value
    }


def fit_pca(data: Run, n_components: int = 3):
    return PCA(n_components=n_components).fit(data.states)


def fit_cebra(
    data: Run,
    max_iterations=10_000,
    output_dimensions=3,
    behaviour: Optional[torch.Tensor] = None,
    decode_states: bool = False,
):
    X: torch.Tensor = data.states[:, :]
    if decode_states:
        X = data.model.decoder(X).detach()

    if behaviour is None:
        behaviour = data.locations_offset
        print('No behaviour tensor was provided. Using CEBRA-behaviour across locations')

    X_behaviour: torch.Tensor = rearrange(behaviour, 't -> t 1')

    model = cebra.CEBRA(
        model_architecture='offset10-model-mse',
        batch_size=512,
        learning_rate=3e-4,
        temperature=1,
        output_dimension=output_dimensions,
        max_iterations=max_iterations,
        distance='euclidean',
        conditional='time_delta',
        device='cuda_if_available',
        verbose=True,
        time_offsets=1,
    )  # type: ignore
    model.fit(X, X_behaviour)

    return model


class SubPal:
    def __init__(self, low, high, cmap) -> None:
        self.low = low
        self.high = high
        self.normaliser = plt.Normalize(vmin=low, vmax=high)  # type: ignore

        self.cmap = cmap

    def in_bounds(self, value: float) -> bool:
        return self.low <= value <= self.high

    def map(self, value: float) -> Tuple:
        assert self.in_bounds(value)

        return self.cmap(self.normaliser(value))


def map_to_subpal(location: float, sub_palettes: List[SubPal]) -> Union[Tuple, str]:
    for pal in sub_palettes:
        if pal.in_bounds(location):
            return pal.map(location)

    raise ValueError


def make_sub_palettes(rz_min: int, rz_max: int) -> List[SubPal]:
    sub_palettes: List[SubPal] = [  # Organised by priority
        SubPal(0, 30, sns.color_palette('Greys_r', as_cmap=True)),
        SubPal(170, 200, sns.color_palette('Greys', as_cmap=True)),
        SubPal(rz_min, rz_max, sns.color_palette('Greens_r', as_cmap=True)),
        SubPal(30, rz_min, sns.color_palette('Blues_r', as_cmap=True)),
        SubPal(rz_max, 170, sns.color_palette('Reds', as_cmap=True)),
    ]

    return sub_palettes


def locations_to_colours(locations: Iterable, rz_min: int = DEFAULT_RZ_MIN, rz_max: int = DEFAULT_RZ_MAX) -> List[str]:
    colours: List[str] = []
    sub_palettes: List[SubPal] = make_sub_palettes(rz_min, rz_max)

    for l in locations:
        colours.append(map_to_subpal(np.clip(l, 0, 200), sub_palettes))

    return colours


def make_legend(rz_min=DEFAULT_RZ_MIN, rz_max=DEFAULT_RZ_MAX):
    plot_stops_base(rz_min=rz_min, rz_max=rz_max)
    step = 0.10
    for l in np.linspace(0 + step, 200, num=200):
        plt.axvspan(l - step, l, 0.75, 1.0, color=map_to_subpal(l, make_sub_palettes(rz_min, rz_max)))
    strip_plot_axes()


def plot_pca_mpl_2d(
    data: Run,
    pca=None,
    alpha=1.0,
    color: Optional[str] = None,
    rz_min: float = DEFAULT_RZ_MIN,
    rz_max: float = DEFAULT_RZ_MAX,
    colours_fn: Callable = locations_to_colours,
    n: int = -1,
    label: Optional[str] = None,
    zorder: int = -1,
):
    if pca is None:
        pca = PCA(n_components=2).fit(data.states)

    if color is None:
        color = colours_fn(data.locations[:n], rz_min=rz_min, rz_max=rz_max)

    x, y, *_ = pca.transform(data.states[:n:]).T

    sns.scatterplot(
        x=x, y=y, s=50.0, zorder=zorder, color=color, alpha=alpha, edgecolor='none', label=label, legend=False
    )


def plot_fps_2d(data: Run, fps: List[FP], stable_color: str = 'blue'):
    pca = PCA(n_components=2).fit(data.states)

    if len(fps) > 0:
        plt.scatter(
            *pca.transform(np.concatenate([fp.fp_batched for fp in fps])).T,
            s=300,
            marker='*',
            edgecolors='black',
            c=[stable_color if fp.is_stable else 'purple' for fp in fps],
        )
    else:
        print('Warning: no fixed points provided to fp plotting function')


def plot_unstable_modes(data: Run, fps: List[FP], qtol: Optional[float] = None, plot_point: Optional[bool] = True):
    if qtol is not None:
        fps = [fp for fp in fps if fp.q < qtol]

    pca = PCA(n_components=3).fit(data.states)

    unstable_fps = [fp for fp in fps if not fp.is_stable]

    ax = plt.gca()
    for fp in unstable_fps:
        if plot_point:
            ax.scatter3D(*pca.transform(fp.fp_batched).T, color='purple', marker='*', s=75)
        for unstable_mode in fp.unstable_modes:
            ax.plot(*pca.transform(unstable_mode).T, color='purple')


def plot_unstable_modes_2d(data: Run, fps: List[FP], qtol: Optional[float] = None, plot_point: Optional[bool] = True):
    if qtol is not None:
        fps = [fp for fp in fps if fp.q < qtol]

    pca = PCA(n_components=2).fit(data.states)

    unstable_fps = [fp for fp in fps if not fp.is_stable]

    ax = plt.gca()
    for fp in unstable_fps:
        if plot_point:
            ax.scatter(*pca.transform(fp.fp_batched).T, color='#00000000', marker='*', edgecolors='black', s=300)
        for unstable_mode in fp.unstable_modes:
            ax.plot(*pca.transform(unstable_mode).T, color='purple')


def _plot_fps_with_qtol(
    data: Run, fps: List[FP], qtol: float, mode: Literal['interactive', '3d', '2d'] = 'interactive', qtol_plot=True
):
    valid_fps = filter_fps(fps, qtol=qtol, plot=qtol_plot)

    if mode == 'interactive':
        raise NotImplementedError
    elif mode == '3d':
        raise NotImplementedError
    elif mode == '2d':
        plot_fps_2d(data, valid_fps)


def plot_qtol_hist(data: Run, fps: List[FP], qtol: float):
    valid_fps = filter_fps(fps, qtol=qtol, plot=True)
    speeds = calculate_trial_speeds(data)
    sns.histplot(x=speeds, log_scale=True, color='red', stat='proportion')
    print(f'Threshold: {qtol} -> Fixed points {len(valid_fps)}/{len(fps)}')


def plot_fps_interactive_with_qtol(data: Run, fps: List[FP], qtol: float, qtol_plot=True):
    _plot_fps_with_qtol(data, fps, qtol, mode='interactive', qtol_plot=qtol_plot)


def plot_fps_2d_with_qtol(data: Run, fps: List[FP], qtol: float, qtol_plot=True):
    _plot_fps_with_qtol(data, fps, qtol, mode='2d', qtol_plot=qtol_plot)


def plot_fps_with_qtol(data: Run, fps: List[FP], qtol: float):
    _plot_fps_with_qtol(data, fps, qtol, mode='3d')


def plot_trajectory_2d(
    data: Run,
    trajectory_states: np.ndarray,
    color='red',
    plot_steps=True,
    zorder: Optional[int] = 10,
    pca: Optional = None,
    lw=2,
    s: int = 30,
):
    if pca is None:
        pca = PCA(n_components=2).fit(data.states)

    ax = plt.gca()

    x, y, *_ = pca.transform(trajectory_states).T

    ax.plot(x, y, color=color, linewidth=lw, zorder=zorder)
    if plot_steps:
        ax.scatter(x, y, color=color, edgecolors=color, s=s, alpha=1, zorder=zorder)


def plot_stop_distributions(
    data: Run,
    rz_min: int = DEFAULT_RZ_MIN,
    rz_max: int = DEFAULT_RZ_MAX,
    minimal=True,
    below_stops=True,
    p_colour: str = '#00000000',
    b_colour: str = 'blue',
    nb_colour: str = 'black',
    trial_types: List[str] = ['nb', 'b'],
    override_max_count: Optional[int] = None,  # Used for combined plots
    divide_y_by: float = 1.0,
    plot_base: bool = True,
    bin_width=4,
    lw=3,
):
    if below_stops:
        plt.figure(figsize=(3.3, 2.5 / 3))

    if plot_base:
        plot_stops_base(rz_min, rz_max)

    type_to_color: Dict[str, str] = {'nb': nb_colour, 'b': b_colour, 'p': p_colour}

    all_bin_counts: List[int] = []  # Keep track of this to calculate max count

    for trial_type in trial_types:
        colour: str = type_to_color[trial_type]

        stop_locations: torch.Tensor = data.locations[data.stops]
        trial_type_mask: torch.Tensor = data.trial_types[data.stops] == trial_type
        bin_width = bin_width
        bins = np.arange(0, 201, bin_width)
        stop_bins = np.digitize(stop_locations[trial_type_mask], bins)

        bin_counts, bin_edges = np.histogram(stop_bins, bins=np.arange(len(bins) + 1))

        plt.step(
            bin_edges[:-1] * bin_width, bin_counts / divide_y_by, where='pre', color=colour, alpha=0.7, linewidth=lw
        )

        all_bin_counts.extend(bin_counts)

    max_count = max(all_bin_counts) if override_max_count is None else override_max_count

    if override_max_count:
        plt.ylim(0, max_count)

    print(max_count)

    if minimal:
        plt.xticks([])
        plt.xlabel('')
        plt.ylabel('')
    else:
        plt.xlabel('Location')
        plt.ylabel('Density')

    plt.gca().tick_params(left=True)


def get_max_stops_for_dist(runs: Iterable[Run], trial_types: Tuple[str] = ('nb', 'b')) -> int:
    all_bin_counts: List[int] = []

    for run in runs:
        for trial_type in trial_types:
            # TODO duplication from plot_stop_distributions
            stop_locations: torch.Tensor = run.locations[run.stops]
            trial_type_mask: torch.Tensor = run.trial_types[run.stops] == trial_type
            bin_width = 4
            bins = np.arange(0, 201, bin_width)
            stop_bins = np.digitize(stop_locations[trial_type_mask], bins)

            bin_counts, bin_edges = np.histogram(stop_bins, bins=np.arange(len(bins) + 1))

            all_bin_counts.extend(bin_counts)

    return max(all_bin_counts)


def plot_agent_track_location(location: float, y: float = 1.5):
    """Plots a location line on a track figure

    Intended for explanatory figs, e.g. manifold trajectory plots with this plot underneath
    """
    plots.make_figure(1, 1 / 3)
    plot_stops_base(60 + 30, 80 + 30)
    strip_plot_axes()

    plt.scatter(x=location, y=y, marker='>')

    plt.ylim(0, 3)


def plot_stops_base(
    rz_min: Optional[int] = DEFAULT_RZ_MIN,
    rz_max: Optional[int] = DEFAULT_RZ_MAX,
    set_xticks: Optional[bool] = True,
    blackbox_colour: Optional[str] = None,
    rz_colour: str = 'green',
):
    blackbox_colour: str = blackbox_colour or 'black'

    ax = plt.gca()
    ax.axvspan(0, 30, alpha=0.15, color=blackbox_colour)

    blackbox2_end: int = 60 + 60 + 20 + 30 + 30
    ax.axvspan(60 + 60 + 20 + 30, blackbox2_end, alpha=0.15, color=blackbox_colour)

    ax.axvspan(rz_min, rz_max, alpha=0.15, color=rz_colour)

    if set_xticks:
        plots.set_xticks_track()

    plt.xlabel('Location')
    plt.xlim(0, 200)


def plot_stops_base_vert(
    rz_min: Optional[int] = DEFAULT_RZ_MIN,
    rz_max: Optional[int] = DEFAULT_RZ_MAX,
    set_xticks: Optional[bool] = False,
    blackbox_colour: Optional[str] = None,
    rz_colour: str = 'green',
):
    blackbox_colour: str = blackbox_colour or 'black'

    ax = plt.gca()
    ax.axhspan(0, 30, alpha=0.15, color=blackbox_colour)

    blackbox2_end: int = 60 + 60 + 20 + 30 + 30
    ax.axhspan(60 + 60 + 20 + 30, blackbox2_end, alpha=0.15, color=blackbox_colour)

    ax.axhspan(rz_min, rz_max, alpha=0.15, color=rz_colour)

    if set_xticks:
        plots.set_xticks_track()

    plt.ylim(0, 200)


def plot_stops(
    data: Run,
    rz_min: int = DEFAULT_RZ_MIN,
    rz_max: int = DEFAULT_RZ_MAX,
    nb_colour: str = 'black',
    b_colour: str = 'blue',
    p_colour: str = 'red',
    blackbox_colour_override: Optional[str] = None,
    minimal=True,
):
    plot_stops_base(rz_min, rz_max, blackbox_colour=blackbox_colour_override)

    global_trials: List[str] = [f'{e}_{t}' for e, t in zip(data.episodes[data.stops], data.trials[data.stops])]

    type_to_color: Dict[str, str] = {'nb': nb_colour, 'b': b_colour, 'p': p_colour}
    colours: List[str] = [type_to_color[trial_type] for trial_type in data.trial_types[data.stops]]
    plt.scatter(data.locations[data.stops], global_trials, c=colours, marker='|')

    if minimal:
        strip_plot_axes()
    else:
        plt.ylabel('Trial')
        plt.yticks([], '')


def bin_single_neuron_activity_by_location(
    neuron_activity: torch.Tensor, locations: torch.Tensor, n_bins: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    bin_means, bin_edges, bin_number = scipy.stats.binned_statistic(
        np.clip(locations.numpy(), a_min=0, a_max=200), neuron_activity.numpy(), bins=n_bins
    )

    return bin_means, bin_edges


def bin_neurons_by_locations(states: torch.Tensor, locations: torch.Tensor, n_bins: int = 200) -> np.ndarray:
    all_bin_means: List[np.ndarray] = []

    n_neurons: int = states.shape[1]  # [time, neurons]
    for neuron_id in range(n_neurons):
        bin_means, _ = bin_single_neuron_activity_by_location(states[:, neuron_id], locations, n_bins)
        all_bin_means.append(bin_means)

    return rearrange(all_bin_means, 'neuron location -> neuron location')


def get_rz_mins_maxes() -> Tuple[Tuple[float], Tuple[float]]:
    N_CONTEXTS: int = 8
    RZ_LENGTH: int = gym.make('LinearNavigationContextual-v2').rz_length
    context_rzs: Dict[int, int] = gym.make('LinearNavigationContextual-v2').reward_locations_by_context
    assert RZ_LENGTH == 10
    RZ_MINS: Tuple[float] = tuple([float(rz - (RZ_LENGTH / 2)) for rz in context_rzs.values()][:N_CONTEXTS])
    RZ_MAXES: Tuple[float] = tuple([float(rz + (RZ_LENGTH / 2)) for rz in context_rzs.values()][:N_CONTEXTS])
    return RZ_MINS, RZ_MAXES


def plot_neurons_combined(
    data: Run,
    neuron_ids: Iterable,
    plot_mean=False,
    rz_min: int = DEFAULT_RZ_MIN,
    rz_max: int = DEFAULT_RZ_MAX,
    highlight_id: Optional[int] = None,
    lw: float = 5,
    mean_lw: float = 5,
    marker: str = '',
    neuron_colour: str = 'black',
    mean_colour: str = 'red',
    alpha: float = 0.5,
    trial_type_filter: Optional[str] = None,
    plot_base: bool = True,
    means_only: bool = False,
):
    states: torch.Tensor = data.states
    locations: torch.Tensor = data.locations

    if trial_type_filter is not None:
        mask: torch.Tensor = torch.tensor(data.trial_types == trial_type_filter)
        states = states[mask]
        locations = locations[mask]

    if plot_base:
        plot_stops_base(rz_min=rz_min, rz_max=rz_max)

    activity_means: List[np.ndarray] = []  # Used to handle mean plotting
    for neuron_id in neuron_ids:
        neuron_activity: torch.Tensor = states[:, neuron_id]

        bin_means, bin_edges = bin_single_neuron_activity_by_location(neuron_activity, locations)

        if not means_only:
            if (highlight_id is not None) and (neuron_id == highlight_id):
                plt.plot(bin_edges[:-1], bin_means, color='red', alpha=0.9, linestyle='-', marker=marker)
            else:
                plt.plot(
                    bin_edges[:-1],
                    bin_means,
                    color=neuron_colour,
                    alpha=alpha,
                    linestyle='-',
                    zorder=1,
                    lw=lw,
                    marker=marker,
                )

        activity_means.append(bin_means)

    if plot_mean:
        plt.plot(bin_edges[:-1], np.array(activity_means).mean(axis=0), lw=mean_lw, color=mean_colour)


def plot_neuron_speed_tuning(data: Run, neuron_id: int, middle_track=True):
    if middle_track:
        middle_track_mask: torch.Tensor = torch.logical_and(data.locations > 30, data.locations < 170)
    else:
        middle_track_mask: torch.Tensor = torch.ones_like(data.locations).bool()

    xs = data.velocities_offset[middle_track_mask]
    ys = data.states[:, neuron_id][middle_track_mask]
    plt.scatter(xs, ys, alpha=0.10)

    plt.ylabel('Neuron activity')
    plt.xlabel('Velocity')


def prepare_input_for_obs(
    data: Run,
    velocity: float,
    indicator: bool,
    blackbox: Union[bool, float],
    beacon: bool,
    context: Optional[int] = None,
) -> torch.Tensor:
    """Given a custom observation, preprocess the observation ready for input into the RNN"""

    # np.array([velocity, 1 if indicator_timer > 0 else 0, int(in_blackbox), int(show_beacon)],
    #                dtype=self.OBS_DTYPE)
    model: ActorCritic = data.model

    if context is not None:
        n_contexts: int = 12
        context_onehot: List[bool] = [False] * n_contexts
        context_onehot[context] = True
    else:
        context_onehot = []

    if not isinstance(blackbox, bool):
        print(f'Got {type(blackbox)} for blackbox. User must be performing input interpolation.')

    raw_obs: torch.Tensor = torch.Tensor(
        [velocity, 1 if indicator else 0, float(blackbox), int(beacon)] + context_onehot
    ).float()

    obs: Dict[str, torch.Tensor] = {'obs': rearrange(raw_obs, 'obs_shape -> 1 obs_shape')}  # Obs expects a batch
    normalized_obs = prepare_and_normalize_obs(model, obs)

    inputs: torch.Tensor = model.forward_head(normalized_obs)

    return inputs


def run_over_inputs(data: Run, inputs: torch.Tensor, initial_state: Optional[torch.Tensor] = None) -> torch.Tensor:
    model: ActorCritic = data.model

    device = inputs.device

    if initial_state is None:
        rnn_states: torch.Tensor = torch.zeros([1, data.rnn.hidden_size], dtype=torch.float32, device=device)
    else:
        assert initial_state.shape == (1, data.rnn.hidden_size)
        rnn_states: torch.Tensor = initial_state

    states_history: List[torch.Tensor] = []

    for t in range(inputs.shape[0]):
        with torch.no_grad():
            something_else, rnn_states = model.forward_core(inputs[t].unsqueeze(0), rnn_states)
        assert torch.all(something_else == rnn_states)

        states_history.append(rnn_states)

    return torch.cat(states_history)


def save_figure(path: Path, transparent=True, allow_non_pdf: bool = False, **kwargs):
    print(f'Saving figure to {path}. Making dir if it does not already exist.')
    path.parent.mkdir(exist_ok=True, parents=True)
    if not isinstance(path, Path):
        raise ValueError('Path was not a path! Did you mean to call save_figure_name instead?')

    valid_exts: List[str]
    if ('_plot' in path.name) or allow_non_pdf:
        valid_exts = ['.pdf', '.png']
    else:
        valid_exts = ['.pdf']
    assert any([ext in path.name for ext in valid_exts])

    plt.savefig(path, transparent=transparent, bbox_inches='tight', **kwargs)


def save_figure_name(name: str):
    if '.png' not in name:
        name += '.png'

    save_figure(BASE_FIGURE_PATH / name)
