# %%
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns
import torch
from matplotlib.colors import TwoSlopeNorm
from rastermap import Rastermap
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

import sf.analysis as analysis
import sf.cebra_analysis as cebra_analysis
import sf.fixedpoints as fixedpoints
import sf.model_analysis as model_analysis
import sf.plots as plots
from sf.analysis import get_fig_dir
from sf.notebook_setup import MIDDLE_CONTEXT
from sf.utils import Run

PARAMS: Dict[str, Any] = analysis.get_curriculum_exemplar_params()

# %%
FIGURE_PATH: Path = get_fig_dir('dynamics')
analysis.make_paper_theme()

# %%
curriculum: analysis.ModelCurriculumPaths = analysis.get_curriculum_exemplar()
runs: List[Run] = analysis.get_contexts_runs(curriculum.pre)
RUN: Run = runs[MIDDLE_CONTEXT]


MILESTONE_INDICES: List[int] = model_analysis.training_curve_with_episodes(RUN, PARAMS['episodes'])


# %%
def _get_runs() -> List[Run]:
    return [
        analysis.run_model(RUN.model_path, env_name=RUN.env_name, milestone_index_override=i)[0]
        for i in MILESTONE_INDICES
    ]


RUNS_ACROSS_TRAINING: List[Run] = _get_runs()
RUNS_ACROSS_TRAINING_TEST: List[Run] = _get_runs()


# %%
def scale_labels(multiplier: float = 1.6) -> None:
    plt.tick_params(axis='both', labelsize=multiplier * plt.gca().get_xticklabels()[0].get_size())


# %% [markdown]
# # Results

# PCs across the track


def plot_pcs_across_track(run: Run = RUN) -> None:
    pca: PCA = analysis.fit_pca(run, n_components=15)
    states: np.ndarray = pca.transform(run.states.numpy()).T

    panel = plots.PanelFigure(cols=3, rows=5)
    for i, pc in enumerate(states):
        panel()

        plt.title(f'PC{i + 1}')
        analysis.plot_stops_base()
        plt.scatter(run.locations_offset, pc)
        plt.xlabel('')
        plt.xticks([])

        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

    plt.tight_layout()


plot_pcs_across_track()
analysis.save_figure(FIGURE_PATH / 'pcs_across_track.pdf')
plt.close()


# %% [markdown]
# ## Location manifolds across training

# %%
analysis.make_legend()
analysis.save_figure(FIGURE_PATH / 'legend.pdf')
plt.close()

# %%
for i, run in enumerate(RUNS_ACROSS_TRAINING):
    analysis.plot_pca_mpl_2d(run)
    plots.despine_topright()
    plots.strip_plot_axes()
    analysis.save_figure(FIGURE_PATH / f'dynamics_{i}.pdf')
    plt.close()

# %% [markdown]
# ## B/NB manifolds


# %%
def plot_bnb_manifold(run: Run = RUNS_ACROSS_TRAINING[-1]):
    analysis.plot_pca_mpl_2d(
        run, color=['blue' if t == 'b' else 'black' for t in run.trial_types][:-1], alpha=0.5
    )  # TODO deal with off-by-one edge case
    plots.despine_topright()
    plots.strip_plot_axes()
    analysis.save_figure(FIGURE_PATH / 'dynamics_bnb.pdf')
    plt.close()


plot_bnb_manifold()

# %% [markdown]
# ## Information


# %%
def calculate_mutual_information(x, y) -> float:
    x = x.reshape(-1, 1)
    nats: float = mutual_info_regression(x, y, discrete_features=False)[0]

    bits: float = nats / np.log(2)

    return bits


def plot_information(run: Run):
    plots.make_figure(w=0.75)

    n_components = 15
    pca = analysis.fit_pca(run, n_components=n_components)
    component = list(range(1, n_components + 1))

    outbound_mask = torch.logical_and(run.locations_offset < 89, run.locations_offset > 31)

    mi_scores = [
        calculate_mutual_information(pc[outbound_mask], run.locations_offset[outbound_mask])
        for pc in pca.transform(run.states).T
    ]
    plt.plot(component, mi_scores, color='#E41A1C', marker='o', lw=2.2)

    mi_scores = [
        calculate_mutual_information(pc[outbound_mask], run.velocities_offset[outbound_mask])
        for pc in pca.transform(run.states).T
    ]
    plt.plot(component, mi_scores, color='#377EB8', marker='o', lw=2.2)

    scale_labels()

    plt.ylabel('')
    plt.title('')


for i, r in enumerate(RUNS_ACROSS_TRAINING):
    plot_information(r)
    plt.ylim(0, 4.0)  # TODO: refactor to use sharey...
    analysis.save_figure(FIGURE_PATH / f'information_{i}.pdf')
    plt.close()

# %% [markdown]
# ## Location decoding

# %% [markdown]
#

# %%
for i, (run_train, run_test) in enumerate(zip(RUNS_ACROSS_TRAINING, RUNS_ACROSS_TRAINING_TEST)):
    plt.figure(figsize=(3.3 / 1.50, 2.5))

    reducer: PCA = analysis.fit_pca(run_train, n_components=2)
    decoder = cebra_analysis.fit_location_decoder(run_train, reducer)

    pca_untrained_r2 = cebra_analysis.plot_decoding(run_test, reducer, decoder)
    scale_labels()
    plt.xlim(0, 450)
    plt.ylim(0, 200)
    plt.title('')
    analysis.save_figure(FIGURE_PATH / f'decoding_{i}.pdf')
    plt.close()

    cebra_analysis.plot_decoding_scatter(run_test, reducer, decoder)
    scale_labels()
    analysis.save_figure(FIGURE_PATH / f'decoding_scatter_{i}.pdf')
    plt.close()

# %% [markdown]
# ## PCA
#
# ### Loadings matrix


# %%
def plot_loadings():
    n_components: int = 15

    loadings: np.ndarray = analysis.fit_pca(RUN, n_components=n_components).components_.T

    abs_max: float = np.max(np.abs(loadings))

    plt.figure(dpi=600)
    g = sns.clustermap(
        loadings,
        figsize=(5, 7),
        cmap='bwr',
        center=0,
        vmin=-abs_max,
        vmax=+abs_max,
        xticklabels=list(range(1, n_components + 1)),
        col_cluster=False,
        row_cluster=True,
    )
    g.ax_heatmap.set_yticklabels([])
    g.ax_heatmap.yaxis.set_ticks([])

    analysis.save_figure(FIGURE_PATH / 'clustered_loadings.pdf')
    plt.close()


analysis.make_paper_theme()
plot_loadings()


# %%
# ### Rastermap
def fit_rastermap(run: Run) -> Rastermap:
    model = Rastermap(n_PCs=200, n_clusters=100, locality=0.75, time_lag_window=50, time_bin=1, bin_size=1)
    episode: int = 0
    episode_mask: torch.Tensor = run.episodes == episode
    states: np.ndarray = run.states[episode_mask].numpy().T
    states: np.ndarray = scipy.stats.zscore(states, axis=1)

    model = model.fit(states)

    return model


RASTERMAP_MODEL = fit_rastermap(RUN)

# %%
HIGH_SAMPLED_RUN: Run = analysis.run_model(RUN.model_path, n_frames=20_000)[0]


# %%
def plot_rastermap(
    run: Run = RUN,
    model: Rastermap = RASTERMAP_MODEL,
    loadings: np.ndarray = analysis.fit_pca(RUN, n_components=15).components_,
) -> None:
    activity: np.ndarray = analysis.bin_neurons_by_locations(run.states, run.locations_offset)
    activity: np.ndarray = scipy.stats.zscore(activity, axis=1)

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    ax.imshow(activity[model.isort, :], vmin=0, vmax=1.5, cmap='Reds')

    analysis.plot_stops_base()
    plt.xticks([])
    plt.xlabel('')
    plt.yticks([])

    start: float = 0.59
    for i in range(15):
        width: float = 0.01
        gap: float = 0.012
        ax2 = fig.add_axes([start + i * gap, 0.1, width, 0.8])
        norm = TwoSlopeNorm(vmin=loadings[i].min(), vcenter=0, vmax=loadings[i].max())
        ax2.imshow(loadings[i].reshape(-1, 1)[RASTERMAP_MODEL.isort], cmap='bwr', aspect='auto', norm=norm)
        # ax2.set_title(f'{i+1}')
        ax2.axis('off')

        pos1 = ax.get_position()
        pos2 = ax2.get_position()
        ax2.set_position([pos2.x0, pos1.y0, pos2.width, pos1.height])


plot_rastermap(HIGH_SAMPLED_RUN, RASTERMAP_MODEL)

analysis.save_figure(FIGURE_PATH / 'rastermap.pdf')
plt.close()


# %% [markdown]
# ### Scree


# %%
def plot_scree_across_training():
    stage_colours: Iterable = sns.color_palette('Set1', n_colors=len(RUNS_ACROSS_TRAINING))
    for r, stage, colour in zip(
        RUNS_ACROSS_TRAINING, ['Untrained', 'Partially trained', 'Fully trained'], stage_colours
    ):
        n_components: int = 15
        pca = analysis.fit_pca(r, n_components=15)

        plt.xticks([1, 5, 10, 15])
        explained_variance: np.ndarray = np.cumsum(pca.explained_variance_ratio_)
        component: np.ndarray = np.array(range(n_components)) + 1
        plt.plot(component, explained_variance, color=colour, label=stage, marker='o')

        plt.title('')
        plt.xlabel('')
        plt.ylabel('')

        plots.despine_topright()
        plt.legend(frameon=False)

    analysis.save_figure(FIGURE_PATH / 'scree.pdf')
    plt.close()


analysis.make_paper_theme()
plot_scree_across_training()


# %% [markdown]
# ## Trajectory progression
#
# ### Setup: Fixed point finding
#

# %%
FPS_AT_CONDITIONS: Dict[str, List[analysis.FP]] = fixedpoints.find_fps_at_conditions(
    RUN, ['middle_3v', 'blackbox_3v', 'indicator_0v', 'middle_0v'], cache_name='1k_test0v', base_context=MIDDLE_CONTEXT
)


# %%
def explore_fps() -> None:
    panel = plots.PanelFigure(cols=2, rows=len(FPS_AT_CONDITIONS), dpi=70)
    qtol: float = 1e-3
    for condition, fps in FPS_AT_CONDITIONS.items():
        panel()
        analysis.plot_qtol_hist(RUN, fps, qtol=qtol)

        panel()
        plt.title(condition)
        analysis.plot_pca_mpl_2d(RUN)
        analysis.plot_fps_2d(RUN, analysis.filter_fps(fps, qtol=qtol))


explore_fps()

# %%
# Plotting fps at 0v


def plot_fps_0v():
    fps: List[analysis.FP] = FPS_AT_CONDITIONS['middle_0v']
    qtol: float = 1e-4

    analysis.plot_pca_mpl_2d(RUN)
    analysis.plot_unstable_modes_2d(RUN, analysis.filter_fps(fps, qtol=qtol), plot_point=True)
    plots.despine_topright()
    analysis.strip_plot_axes()

    analysis.save_figure(FIGURE_PATH / 'fps_middle0v_unstable.pdf')
    plt.close()

    analysis.plot_pca_mpl_2d(RUN)
    analysis.plot_fps_2d(RUN, analysis.filter_fps(fixedpoints.filter_stable(fps), qtol=qtol), stable_color='#00000000')
    plots.despine_topright()
    analysis.strip_plot_axes()

    analysis.save_figure(FIGURE_PATH / 'fps_middle0v_stable.pdf')
    plt.close()


plot_fps_0v()

# %% [markdown]
#
#
# ### Back to results


# %%
# TODO: refactor trajectory construction
def _redistribute_final(increments: List[float]) -> List[float]:
    increments: np.ndarray = np.array(increments)

    mean_except_last: float = sum(increments[:-1] / len(increments[:-1]))
    increments[-1] += mean_except_last

    increments[:-1] -= increments[:-1] / len(increments[:-1])

    increments: List[float] = list(increments)
    random.shuffle(increments)
    return increments


def _generate_random_increments(n_timesteps, total_distance):
    assert n_timesteps > 0
    assert total_distance > 0

    random_parts = [random.random() for _ in range(n_timesteps - 1)]
    total_random_part = sum(random_parts)
    normalized_random_parts = [total_distance * (part / total_random_part) for part in random_parts]

    increments = normalized_random_parts.copy()

    final_increment = total_distance - sum(increments)
    increments.append(final_increment)

    return increments


def generate_increments(n_timesteps, total_distance):
    valid = False
    while not valid:
        increments = _generate_random_increments(n_timesteps, total_distance)
        if increments[-1] > 0:
            valid = True
    return increments


def get_random_velocity(run: Run) -> float:
    # Random = randomly samnpled during the trial
    return np.random.choice(run.velocities)


@dataclass
class TrajectoryData:
    metadata: List[str]
    fp_conditions: List[str]
    timestep_fps: List[List[analysis.FP]]
    trajectory: torch.Tensor
    locations: List[float]
    stopped: List[bool]


def construct_trajectory(run: Run, fps_at_conditions: Dict[str, List[analysis.FP]]) -> TrajectoryData:
    inputs_raw: List[torch.Tensor] = []
    metadata: List[str] = []
    fp_conditions: List[str] = []
    locations: List[float] = []
    stopped: List[bool] = []

    for v in generate_increments(15, 45):
        inputs_raw.append(
            analysis.prepare_input_for_obs(
                data=run, velocity=v, indicator=False, blackbox=True, beacon=False, context=MIDDLE_CONTEXT
            )
        )
        metadata.append('Moving through black box. Velocity = 3')
        fp_conditions.append('blackbox_3v')
        stopped.append(False)

        if len(locations) == 0:
            locations.append(0)
        else:
            locations.append(locations[-1] + v)

    for v in generate_increments(23, 3 * 23):
        inputs_raw.append(
            analysis.prepare_input_for_obs(
                data=run, velocity=v, indicator=False, blackbox=False, beacon=False, context=MIDDLE_CONTEXT
            )
        )
        metadata.append('Moving through middle track. Velocity = 3')
        fp_conditions.append('middle_3v')
        locations.append(locations[-1] + v)
        stopped.append(False)

    v = 0.5
    inputs_raw.append(
        analysis.prepare_input_for_obs(
            data=run, velocity=v, indicator=False, blackbox=False, beacon=False, context=MIDDLE_CONTEXT
        )
    )
    metadata.append('Moving through in rz: indicator = 0')
    fp_conditions.append('middle_3v')
    locations.append(locations[-1] + v)
    stopped.append(False)

    inputs_raw.append(
        analysis.prepare_input_for_obs(
            data=run, velocity=0, indicator=True, blackbox=False, beacon=False, context=MIDDLE_CONTEXT
        )
    )
    metadata.append('Stopped in RZ: indicator = 1')
    fp_conditions.append('indicator_0v')  # should be an indicator 0v
    locations.append(locations[-1] + 0)
    stopped.append(True)

    inputs_raw.append(
        analysis.prepare_input_for_obs(
            data=run, velocity=3, indicator=False, blackbox=False, beacon=False, context=MIDDLE_CONTEXT
        )
    )
    metadata.append('Moving through rz: ')
    fp_conditions.append('middle_3v')
    locations.append(locations[-1] + 3)
    stopped.append(False)

    # for i in range(20):
    for v in generate_increments(20, 3 * 20):
        inputs_raw.append(
            analysis.prepare_input_for_obs(
                data=run, velocity=v, indicator=False, blackbox=False, beacon=False, context=MIDDLE_CONTEXT
            )
        )
        metadata.append('Moving through middle track (after reward zone). Velocity = 3')
        fp_conditions.append('middle_3v')
        locations.append(locations[-1] + v)
        stopped.append(False)

    for v in generate_increments(15, 3 * 15):
        inputs_raw.append(
            analysis.prepare_input_for_obs(
                data=run, velocity=v, indicator=False, blackbox=True, beacon=False, context=MIDDLE_CONTEXT
            )
        )
        metadata.append('Moving through blackbox again. Velocity = 3')
        fp_conditions.append('blackbox_3v')
        locations.append(locations[-1] + v)
        stopped.append(False)

    qtol = 1e-12
    timestep_fps: List[List[analysis.FP]] = [
        analysis.filter_fps(fps_at_conditions[condition], qtol=qtol) for condition in fp_conditions
    ]
    trajectory: torch.Tensor = analysis.run_over_inputs(run, torch.concat(inputs_raw))

    return TrajectoryData(
        metadata=metadata,
        fp_conditions=fp_conditions,
        timestep_fps=timestep_fps,
        trajectory=trajectory,
        locations=locations,
        stopped=stopped,
    )


def loop_trajectory_figs(run: Run, trajectory_data: TrajectoryData, figs_dir: Path):

    metadata = trajectory_data.metadata
    fp_conditions = trajectory_data.fp_conditions
    trajectory = trajectory_data.trajectory
    locations = trajectory_data.locations

    to_save = [17, 32, 58, 65]

    offset = 3

    xlim, ylim, _zlim = None, None, None
    fig_i: int = 0
    for i, m in enumerate(metadata[1:]):
        i = i + 1
        prev_m: str = metadata[i - 1]

        if i < offset:
            continue

        # i = 10 for black box
        if (
            (prev_m != m) or (i == 10) or (i in to_save) or (i == len(fp_conditions) - 2)
        ):  # minus two because we start at 1:
            analysis.plot_pca_mpl_2d(RUN, alpha=0.10)

            analysis.plot_trajectory_2d(
                run, trajectory_states=trajectory[offset:i].numpy(), plot_steps=False, color='#FFD92F'
            )
            plots.despine_topright()
            plots.strip_plot_axes()

            fig_i += 1

            print(m, prev_m)
            if xlim is None:
                xlim = plt.gca().get_xlim()
                ylim = plt.gca().get_ylim()
            else:
                plt.gca().set_xlim(xlim)
                plt.gca().set_ylim(ylim)
            analysis.save_figure(figs_dir / f'trajectory_condition_{fig_i}.pdf')
            plt.close()

            analysis.plot_agent_track_location(locations[i - 1], y=0.5)  # off by one (input roll)

            analysis.save_figure(figs_dir / f'agent_location_{fig_i}.pdf')
            print(locations[i - 1])
            plt.close()


def save_trajectory_figs():
    trajectory_data: TrajectoryData = construct_trajectory(RUN, FPS_AT_CONDITIONS)

    figs_dir: Path = FIGURE_PATH / 'trajectory'
    figs_dir.mkdir(exist_ok=True)

    loop_trajectory_figs(RUN, trajectory_data, figs_dir)


save_trajectory_figs()
