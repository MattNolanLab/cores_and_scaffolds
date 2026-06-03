from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from einops import rearrange
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA

import sf
import sf.analysis as analysis
import sf.cebra_analysis as cebra_analysis
import sf.plots as plots
import sf.specific.disruption_perturbations
import sf.style as style
import sf.subspace as subspace
import sf.utils as utils
from sf import EXPERIMENTS
from sf.analysis import get_fig_dir
from sf.notebook_setup import setup_notebook
from sf.utils import Run

setup_notebook()
analysis.make_paper_theme()

figure_path: Path = get_fig_dir('rank_main_truncation')

plots.set_font_size(9)

original_run = analysis.run_model(analysis.get_exemplar_model())[0]

speed: str = 'Rnd Slow'


# %%
_, singular_vals, _ = torch.linalg.svd(original_run.rnn.weight_hh_l0.detach(), full_matrices=False)
plt.figure(figsize=plots.A5_L * [1 / 6, 1 / 4])
sns.lineplot(x=np.arange(1, len(singular_vals) + 1), y=singular_vals, marker='o', markeredgecolor='none', alpha=1.0)
plt.xticks([1, 256, 512])
plt.ylabel('Singular value')
plt.xlabel('Mode')
plt.tight_layout()

analysis.save_figure(figure_path / 'singular_values.pdf')
plt.show()


# %%
def _zscore(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mu: torch.Tensor = torch.mean(x)
    sd: torch.Tensor = torch.std(x)
    return (x - mu) / (sd + eps)


def _bin_across_location(
    values: torch.Tensor, locations: torch.Tensor, n_bins: int = 200
) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_means, bin_edges = analysis.bin_single_neuron_activity_by_location(values, locations, n_bins=n_bins)
    xs = torch.tensor(bin_edges[:-1])
    ys = torch.tensor(bin_means)
    return xs, ys


mappings = subspace.calculate_mapping_matrices(original_run)
latents: torch.Tensor = (mappings['hxs_to_latent'] @ original_run.states.T).T  # [T, n_latents]

pca: PCA = analysis.fit_pca(original_run, n_components=2)
pcs_np: np.ndarray = pca.transform(original_run.states)  # [T, 2]

mask_nb: torch.BoolTensor = original_run.trial_types == 'nb'

loc_nb: torch.Tensor = original_run.locations[mask_nb]

pc1_nb: torch.Tensor = _zscore(torch.tensor(pcs_np[mask_nb, 0]))
pc2_nb: torch.Tensor = _zscore(torch.tensor(pcs_np[mask_nb, 1]))
lat1_nb: torch.Tensor = _zscore(latents[mask_nb, 0])
lat2_nb: torch.Tensor = _zscore(latents[mask_nb, 1])

xs1_pc, ys1_pc = _bin_across_location(pc1_nb, loc_nb)
xs1_lat, ys1_lat = _bin_across_location(lat1_nb, loc_nb)
xs2_pc, ys2_pc = _bin_across_location(pc2_nb, loc_nb)
xs2_lat, ys2_lat = _bin_across_location(lat2_nb, loc_nb)

c_pc = sns.color_palette('Blues', n_colors=9)[6]
c_latent = sns.color_palette('Reds', n_colors=9)[6]

fig = plt.figure(figsize=plots.A5_L * [1 / 3, 1 / 4])
gs = plt.GridSpec(nrows=1, ncols=2)

ax1 = fig.add_subplot(gs[0, 0])
plt.sca(ax1)
plt.title('Axis 1')
analysis.plot_stops_base()
plt.plot(xs1_pc, ys1_pc, color=c_pc, lw=2.0, label='PC')
plt.plot(xs1_lat, ys1_lat, color=c_latent, lw=2.0, label='u')
plt.xlabel('Location')

ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
plt.sca(ax2)
plt.title('Axis 2')
analysis.plot_stops_base()
plt.plot(xs2_pc, ys2_pc, color=c_pc, lw=2.0, label='PC')
plt.plot(xs2_lat, ys2_lat, color=c_latent, lw=2.0, label='m')
plt.xlabel('Location')

plt.gca().tick_params(axis='y', labelleft=False)

plots.supylabel('Activity along axis\n(z-scored)', x=0.10, ha='center')

# Legend outside (to the right)
handles = [
    plt.Line2D([0], [0], color=c_pc, lw=2.0, label='PC'),
    plt.Line2D([0], [0], color=c_latent, lw=2.0, label='u'),
]
fig.legend(
    handles=handles, loc='center left', bbox_to_anchor=(0.83, 0.5), frameon=False, handlelength=0.5, handletextpad=0.25
)

plots.tight_layout(right=0.9)

analysis.save_figure(figure_path / 'pcs_vs_latent.pdf')
plt.show()

# %%
fig = plt.figure(figsize=plots.A5_L * [1 / 4, 1 / 2])

ax_top = fig.add_subplot(2, 1, 1)
plt.sca(ax_top)
plt.title(' ')  # dummy for alignment
analysis.plot_stops_base()
plt.plot(xs1_pc, ys1_pc, color=c_pc, lw=2.0, label='PC')
plt.plot(xs1_lat, ys1_lat, color=c_latent, lw=2.0, label='m')
plt.ylabel('Dimension 1')
plt.xlabel('')
plt.gca().tick_params(axis='x', labelbottom=False)  # keep ticks but hide labels

ax_bottom = fig.add_subplot(2, 1, 2, sharex=ax_top)
plt.sca(ax_bottom)
analysis.plot_stops_base()
plt.plot(xs2_pc, ys2_pc, color=c_pc, lw=2.0, label='PC')
plt.plot(xs2_lat, ys2_lat, color=c_latent, lw=2.0, label='m')
plt.ylabel('Dimension 2')
plt.xlabel('Location')

plots.supylabel('Activity along dimension (z-scored)', x=0.1)

handles = [
    plt.Line2D([0], [0], color=c_pc, lw=2.0, label='PC'),
    plt.Line2D([0], [0], color=c_latent, lw=2.0, label='m'),
]


plots.tight_layout()

fig.legend(
    handles=handles,
    loc='lower center',
    ncol=2,
    frameon=False,
    bbox_to_anchor=(plots.get_midpoint_x(), -0.04),
    handlelength=0.5,
    handletextpad=0.25,
)

analysis.save_figure(figure_path / 'pcs_vs_latent_singlecol.pdf')
plt.show()


# %%
latent_indices: np.ndarray = np.array([0, 1, 2, 3, 4])
n_lat = len(latent_indices)
n_pc = n_lat

mappings = subspace.calculate_mapping_matrices(original_run)
hxs_to_latent: torch.Tensor = mappings['hxs_to_latent']

latent_indices = latent_indices[latent_indices < hxs_to_latent.shape[0]]

latent_loadings: np.ndarray = hxs_to_latent[latent_indices].detach().cpu().numpy()

pca: PCA = analysis.fit_pca(original_run, n_components=max(n_pc, 2))
pc_loadings: np.ndarray = pca.components_[:n_pc]

L = latent_loadings
P = pc_loadings
Ln = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-12)
Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
cos_sim = np.abs(Ln @ Pn.T)  # [n_lat, n_pc]

# Labels exactly match the indices order
yticks = [f'{i + 1}' for i in latent_indices.tolist()]
xticks = [f'{j + 1}' for j in range(n_pc)]

fig = plt.figure(figsize=plots.A5_L * [1 / 4, 1 / 4])
plt.sca(plt.gca())
sns.heatmap(
    cos_sim.T, cmap='viridis', vmin=0.0, vmax=1.0, square=True, cbar=True, xticklabels=xticks, yticklabels=yticks
)
plt.ylabel('PC loading')
plt.xlabel('u')

plots.tight_layout()

analysis.save_figure(figure_path / 'loadings_pcs_vs_latent.pdf')
plt.show()

# %%
# Latent loadings (rows are selected latents, in given order)
latent_loadings: np.ndarray = hxs_to_latent[latent_indices].detach().cpu().numpy()  # [n_lat, n_neurons]

# PC loadings (rows are PCs by index, starting at 0)
pca: PCA = analysis.fit_pca(original_run, n_components=max(n_pc, 2))
pc_loadings: np.ndarray = pca.components_[:n_pc]  # [n_pc, n_neurons]

fig = plt.figure(figsize=plots.A5_L * [1 / 5, 1 / 4])
plt.sca(plt.gca())
sns.scatterplot(x=latent_loadings[1], y=pc_loadings[1], alpha=0.8)
plt.xlabel('u$_2$')
plt.ylabel('PC$_2$')

plots.tight_layout()

analysis.save_figure(figure_path / 'u_vs_pc_scatter.pdf')
plt.show()


# %%

ks_by_speed_i: List[List[int]] = [[224, 192, 128, 32], [224, 192, 128, 32], [4, 8, 128, 256]]
for speed_i, s in enumerate(['Rnd Slow', 'Normal', 'Rnd Fast']):  # Order is a bit important because of sharing axes
    df: pd.DataFrame = torch.load(original_run.model_path / 'derivatives/rank/df.pkl')
    df = df[df['mode'].str.startswith('normal')]
    df = df[df['speed'] == s]

    plt.figure(figsize=plots.A5_L * [1 / 4, 1 / 3])
    sns.lineplot(
        data=df,
        x=df['k'],
        y='nb_reward_perc',
        marker='o',
        markeredgecolor='none',
        alpha=0.7,
        color='black',
        label='Uncued',
    )
    sns.lineplot(
        data=df, x=df['k'], y='b_reward_perc', marker='o', markeredgecolor='none', alpha=0.7, color='blue', label='Cued'
    )

    plt.xlabel('Truncated rank')
    plt.ylabel('Reward %')
    plt.xscale('log', base=2)
    ticks = [4, 32, 256]
    plt.xticks(ticks, [str(t) for t in ticks])

    # plt.legend(frameon=False, bbox_to_anchor=(1.00, 0.75), loc='upper left', fontsize=7)
    plots.tight_layout()

    plt.legend(frameon=False, bbox_to_anchor=(0.50, 1.15), loc='upper center', fontsize=7, ncol=2, columnspacing=0.7)

    plt.ylim(-25, 105)
    plt.yticks([0, 50, 100])
    analysis.save_figure(figure_path / f'truncated_rank_{s}.pdf')

    plt.show()

    ks: List[int] = ks_by_speed_i[speed_i]
    plt.figure(figsize=plots.A5_L * [1 / 5, 1 / 3])
    gs = plt.GridSpec(nrows=4, ncols=1)

    for i, (_, row) in enumerate(df[df['k'].isin(ks)].iterrows()):
        plots.asp(gs[i])
        k = row['k']
        run: Run = torch.load(EXPERIMENTS / Path(row['record_path']))['run']
        max_stops: int = analysis.get_max_stops_for_dist([run], trial_types=('nb',))
        analysis.plot_stop_distributions(
            run, trial_types=['nb'], divide_y_by=max_stops, below_stops=False, plot_base=True
        )

        plots.strip_plot_axes()
        plt.ylim(0, 1)

        plt.gca().annotate(f'{k}', xy=(1.04, 0.5), xycoords='axes fraction', va='center', ha='left', rotation=0)

        if i == 3:
            plt.xticks([0, 100, 200])
            plt.xlabel('Location')

    plt.gcf().text(s='Truncated rank', x=0.91, y=0.6, va='center', ha='left', rotation=-90)

    plots.supylabel('Stops', x=0.09)
    plots.tight_layout(right=0.97)

    analysis.save_figure(figure_path / f'truncated_rank_behaviour_{s}.pdf')

    plt.show()


# %%

df: pd.DataFrame = torch.load(original_run.model_path / 'derivatives/rank/df.pkl')
df = df[df['mode'].str.startswith('normal')]

reducer: PCA = analysis.fit_pca(original_run, n_components=14)
decoder = cebra_analysis.fit_location_decoder(original_run, cebra_model=reducer)

speed_to_label: Dict[str, str] = {'Rnd Slow': 'Standard * 0.5', 'Normal': 'Standard', 'Rnd Fast': 'Standard * 2.0'}

speeds_to_plot: List[str] = ['Rnd Slow', 'Normal', 'Rnd Fast']
pal = sns.color_palette('Set1', n_colors=len(speeds_to_plot))

plt.figure(figsize=plots.A4_L * [1 / 4, 1 / 4])

for speed, colour in zip(speeds_to_plot, pal):
    df_speed = df[df['speed'] == speed].sort_values('k')

    ks: List[int] = []
    r2s: List[float] = []

    for _, row in df_speed.iterrows():
        record: Dict[str, Any] = torch.load(EXPERIMENTS / row['record_path'])
        run: Run | None = record['run']
        if run is None:
            continue

        embedding: np.ndarray = reducer.transform(run.states)
        r2: float = cebra_analysis.calculate_location_r2(run, torch.tensor(embedding), decoder)

        ks.append(int(row['k']))
        r2s.append(r2)

    plt.plot(ks, r2s, color=colour, marker='o', markeredgecolor='none', label=speed_to_label[speed])

plt.xscale('log', base=2)
plt.xticks([4, 32, 256], [4, 32, 256])
plt.xlabel('Truncated rank')
plt.ylabel('location R$^2$')

plt.legend(frameon=False)
plots.tight_layout()
plt.show()

# %%


df: pd.DataFrame = torch.load(original_run.model_path / 'derivatives/rank/df.pkl')
df = df[df['mode'].str.startswith('normal')]

speed_to_label: Dict[str, str] = {'Rnd Slow': 'Standard * 0.5', 'Normal': 'Standard', 'Rnd Fast': 'Standard * 2.0'}
speeds_to_plot: List[str] = ['Rnd Slow', 'Normal', 'Rnd Fast']
pal = sns.color_palette('Set1', n_colors=len(speeds_to_plot))

min_nb_stops: int = 10

plt.figure(figsize=plots.A4_L * [1 / 4, 1 / 4])

# background track shading in y (location) coordinates
analysis.plot_stops_base_vert()
plt.ylim(0, 200)

for speed, colour in zip(speeds_to_plot, pal):
    df_speed = df[df['speed'] == speed].sort_values('k')

    ks: List[int] = []
    mean_nb_stops: List[float] = []

    for _, row in df_speed.iterrows():
        record: Dict[str, Any] = torch.load(EXPERIMENTS / row['record_path'])
        run: Run | None = record['run']
        if run is None:
            continue

        nb_stop_mask: torch.Tensor = torch.tensor(run.trial_types == 'nb') & run.stops
        n_nb_stops: int = int(nb_stop_mask.sum().item())
        if n_nb_stops < min_nb_stops:
            continue

        mean_stop_loc: float = float(run.locations[nb_stop_mask].mean().item())

        ks.append(int(row['k']))
        mean_nb_stops.append(mean_stop_loc)

    plt.plot(
        ks, mean_nb_stops, color=colour, marker='o', markeredgecolor='none', linewidth=2, label=speed_to_label[speed]
    )

plt.xscale('log', base=2)
plt.xticks([4, 32, 256], [4, 32, 256])
plt.xlabel('Truncated rank')
plt.ylabel('Mean uncued stop location')

plt.legend(frameon=False)
plots.tight_layout()
plt.show()


# %%


def plot_truncation_avgfirststop():
    df: pd.DataFrame = torch.load(original_run.model_path / 'derivatives/rank/df.pkl')
    df = df[df['mode'].str.startswith('normal')]

    speeds_to_plot: List[str] = ['Rnd Fast', 'Normal', 'Rnd Slow']  # needs to match below!
    pal = [style.SPEEDS_TO_COLOUR[s] for s in ['fast', 'standard', 'slow']]

    min_nb_stops: int = 10

    # plt.figure(figsize=plots.A4_L * [1/4, 1/4])

    analysis.plot_stops_base_vert()
    plt.ylim(0, 200)

    for speed, colour in zip(speeds_to_plot, pal):
        df_speed = df[df['speed'] == speed].sort_values('k')

        ks: List[int] = []
        means: List[float] = []
        stds: List[float] = []

        for _, row in df_speed.iterrows():
            record: Dict[str, Any] = torch.load(EXPERIMENTS / row['record_path'])
            run: Run | None = record['run']
            if run is None:
                continue

            stop_locs: torch.Tensor = torch.tensor(run.get_first_stop_locations()[1])

            if len(stop_locs) < min_nb_stops:
                print(f'Skipped {speed}, {row.k} because num stops below threshold')
                continue

            mean_loc: float = float(stop_locs.mean().item())
            std_loc: float = float(stop_locs.std(unbiased=True).item())

            ks.append(int(row['k']))
            means.append(mean_loc)
            stds.append(std_loc)

        ks_np = np.array(ks, dtype=float)
        means_np = np.array(means, dtype=float)
        # stds_np = np.array(stds, dtype=float)

        plt.plot(ks_np, means_np, color=colour, marker='o', markeredgecolor='none', linewidth=2, zorder=3)
        # plt.fill_between(
        #     ks_np,
        #     means_np - stds_np,
        #     means_np + stds_np,
        #     color=colour,
        #     alpha=0.18,
        #     linewidth=0,
        #     zorder=2,
        # )

    plt.xscale('log', base=2)
    plt.xticks([4, 32, 256], [4, 32, 256])
    plt.xlabel('Truncated rank')
    plt.ylabel('Avg. first uncued stop')


def plot_truncation_reward(trial_type: Literal['b', 'nb']):
    df: pd.DataFrame = torch.load(original_run.model_path / 'derivatives/rank/df.pkl')
    df = df[df['mode'].str.startswith('normal')]

    speeds_to_plot: List[str] = ['Rnd Fast', 'Normal', 'Rnd Slow']  # needs to match below!
    pal = [style.SPEEDS_TO_COLOUR[s] for s in ['fast', 'standard', 'slow']]

    for speed, colour in zip(speeds_to_plot, pal):
        df_speed = df[df['speed'] == speed].sort_values('k')

        plt.plot(
            df_speed['k'],
            df_speed[f'{trial_type}_reward_perc'],
            color=colour,
            marker='o',
            markeredgecolor='none',
            linewidth=2,
            zorder=3,
        )
        # plt.fill_between(
        #     ks_np,
        #     means_np - stds_np,
        #     means_np + stds_np,
        #     color=colour,
        #     alpha=0.18,
        #     linewidth=0,
        #     zorder=2,
        # )

    plt.xscale('log', base=2)
    plt.xticks([4, 32, 256], [4, 32, 256])
    plt.xlabel('Truncated rank')
    plt.ylabel('% reward')
    plt.yticks([0, 50, 100], [0, 50, 100])
    plt.ylim(-25, 105)

    # analysis.save_figure(figure_path / 'first_uncued_stop_normal.pdf')


def plot_truncation_r2():
    df: pd.DataFrame = torch.load(original_run.model_path / 'derivatives/rank/df.pkl')
    df = df[df['mode'].str.startswith('normal')]

    speeds_to_plot: List[str] = ['Rnd Fast', 'Normal', 'Rnd Slow']  # needs to match below!
    pal = [style.SPEEDS_TO_COLOUR[s] for s in ['fast', 'standard', 'slow']]

    reducer = analysis.fit_pca(original_run, n_components=15)
    decoder = cebra_analysis.fit_location_decoder(original_run, reducer)

    for speed, colour in zip(speeds_to_plot, pal):
        df_speed = df[df['speed'] == speed].sort_values('k')

        ks: List[int] = []
        r2s: List[float] = []

        for _, row in df_speed.iterrows():
            record: Dict[str, Any] = torch.load(EXPERIMENTS / row['record_path'])
            run: Run | None = record['run']
            if run is None:
                continue

            r2 = cebra_analysis.calculate_location_r2(run, torch.tensor(reducer.transform(run.states)), decoder=decoder)

            ks.append(int(row['k']))
            r2s.append(r2)

        ks_np = np.array(ks, dtype=float)

        plt.plot(ks_np, r2s, color=colour, marker='o', markeredgecolor='none', linewidth=2, zorder=3)

    plt.xscale('log', base=2)
    plt.xticks([4, 32, 256], [4, 32, 256])
    plt.xlabel('Truncated rank')
    plt.ylabel('Location R$^2$')


plt.figure(figsize=plots.A5_L * [1 / 4, 1 / 3])
plot_truncation_avgfirststop()
plt.tight_layout()
analysis.save_figure(figure_path / 'rank_truncation_panel_firststop.pdf')
plt.show()


plt.figure(figsize=plots.A5_L * [1.25 / 4, 1 / 3])
gs = plt.GridSpec(ncols=2, nrows=1)
plots.asp(gs[0])
plot_truncation_reward('b')
plt.ylabel('% reward')

plots.asp(gs[1])
plot_truncation_reward('nb')
plt.yticks([])
plt.ylabel('')
plots.tight_layout()
analysis.save_figure(figure_path / 'rank_truncation_panel_reward.pdf')
plt.show()


plt.figure(figsize=plots.A5_L * [1 / 4, 1 / 3])
plot_truncation_r2()
plots.tight_layout()
analysis.save_figure(figure_path / 'rank_truncation_panel_r2.pdf')
plt.show()


# %%

df: pd.DataFrame = torch.load(original_run.model_path / 'derivatives/rank/df.pkl')
df = df[df['mode'].str.startswith('normal')]

green = '#4daf4a'
red = '#e41a1c'


speed_to_label: Dict[str, str] = {'Rnd Slow': 'S * 0.5', 'Normal': 'S', 'Rnd Fast': 'S * 2.0'}
speeds_to_plot: List[str] = ['Rnd Slow', 'Normal', 'Rnd Fast']
pal = sns.color_palette('Set1', n_colors=len(speeds_to_plot))

# min_nb_stops: int = 10

# plt.figure(figsize=plots.A4_L * [1/4, 1/4])
# plt.figure(figsize=plots.A5_L * [1 / 3, 1 / 3])  # with legend


def plot_displacement_supplemental_truncation():

    plt.figure(figsize=plots.A5_L * [3 / 5, 1 / 3])
    gs = plots.GridSpec(ncols=2, nrows=1)

    for col_i, trial_type in enumerate(['b', 'nb']):
        plots.asp(gs[col_i])
        analysis.plot_stops_base_vert()
        plt.ylim(0, 200)

        for speed, colour in zip(
            speeds_to_plot,
            [style.SPEEDS_TO_COLOUR['slow'], style.SPEEDS_TO_COLOUR['standard'], style.SPEEDS_TO_COLOUR['fast']],
        ):
            print(speed)
            df_speed = df[df['speed'] == speed].sort_values('k')

            ks: List[int] = []
            means: List[float] = []
            stds: List[float] = []

            for _, row in df_speed.iterrows():
                record: Dict[str, Any] = torch.load(EXPERIMENTS / row['record_path'])
                run: Run | None = record['run']
                if run is None:
                    continue

                stop_locs: torch.Tensor = torch.tensor(run.get_displacements(trial_type).peak_locations)

                mean_loc: float = float(stop_locs.mean().item())
                std_loc: float = float(stop_locs.std().item())

                ks.append(int(row['k']))
                means.append(mean_loc)
                stds.append(std_loc)

            ks_np = np.array(ks, dtype=float)
            means_np = np.array(means, dtype=float)
            stds_np = np.array(stds, dtype=float)

            plt.plot(
                ks_np,
                means_np,
                color=colour,
                marker='o',
                markeredgecolor='none',
                linewidth=2,
                label=speed_to_label[speed],
                zorder=3,
            )
            plt.fill_between(
                ks_np, means_np - stds_np, means_np + stds_np, color=colour, alpha=0.3, linewidth=0, zorder=2
            )

        # plt.ylim(90, 105)
        plt.xscale('log', base=2)
        plt.xticks([4, 32, 256], [4, 32, 256])
        plt.xlim(1, 520)
        plt.xlabel('Truncated rank')

        plt.title({'b': 'Cued', 'nb': 'Uncued'}[trial_type])

        if gs.is_first_col(col_i):
            plt.ylabel('Stop readout peak')
            plt.yticks([0, 100, 200])
        else:
            plt.ylabel('')
            plt.yticks([])

        # plt.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
        plots.tight_layout(right=0.99)

    analysis.save_figure(figure_path / 'rank_truncation_displacement_plot.png')

    plt.show()


plot_displacement_supplemental_truncation()

# %%


df: pd.DataFrame = torch.load(original_run.model_path / 'derivatives/rank/df.pkl')
df = df[df['mode'].str.startswith('normal')]

green = '#4daf4a'
red = '#e41a1c'


speed_to_label: Dict[str, str] = {'Rnd Slow': 'S * 0.5', 'Normal': 'S', 'Rnd Fast': 'S * 2.0'}
speeds_to_plot: List[str] = ['Rnd Slow', 'Normal', 'Rnd Fast']
pal = sns.color_palette('Set1', n_colors=len(speeds_to_plot))

# min_nb_stops: int = 10

# plt.figure(figsize=plots.A4_L * [1/4, 1/4])
# plt.figure(figsize=plots.A5_L * [1 / 3, 1 / 3])  # with legend


for trial_type in ['b', 'nb']:
    fig = plt.figure(figsize=plots.A5_L * [1 / 5, 1 / 3])

    analysis.plot_stops_base_vert()
    plt.ylim(0, 200)

    for speed, colour in zip(speeds_to_plot, [red, 'black', green]):
        print(speed)
        df_speed = df[df['speed'] == speed].sort_values('k')

        ks: List[int] = []
        means: List[float] = []
        stds: List[float] = []

        for _, row in df_speed.iterrows():
            record: Dict[str, Any] = torch.load(EXPERIMENTS / row['record_path'])
            run: Run | None = record['run']

            if run is None:
                continue

            stop_locs: torch.Tensor = torch.tensor(run.get_first_stop_locations()[{'b': 0, 'nb': 1}[trial_type]])

            if len(stop_locs) <= 3:
                print(f'Skipped {speed}, {row.k} because num stops below threshold')
                continue

            mean_loc: float = float(stop_locs.mean().item())
            std_loc: float = float(stop_locs.std().item())

            ks.append(int(row['k']))
            means.append(mean_loc)
            stds.append(std_loc)

        ks_np = np.array(ks, dtype=float)
        means_np = np.array(means, dtype=float)
        stds_np = np.array(stds, dtype=float)

        plt.plot(
            ks_np,
            means_np,
            color=colour,
            marker='o',
            markeredgecolor='none',
            linewidth=2,
            label=speed_to_label[speed],
            zorder=3,
        )
        plt.fill_between(ks_np, means_np - stds_np, means_np + stds_np, color=colour, alpha=0.3, linewidth=0, zorder=2)

    # plt.ylim(90, 105)
    plt.xscale('log', base=2)
    plt.xticks([4, 32, 256], [4, 32, 256])
    plt.xlim(1, 530)
    plt.xlabel('Truncated rank')
    plt.ylabel('Avg. first stop')

    # plt.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plots.tight_layout(right=0.99)

    analysis.save_figure(figure_path / f'{trial_type}_firststop.pdf')

    plt.show()


# %%
argsorted_metrics: Dict[str, torch.Tensor] = sf.specific.disruption_perturbations.get_argsorted_metrics(original_run)

# %%
fig = plt.figure(figsize=plots.A5_L * [1 / 3, 1 / 3], constrained_layout=True)
gs = plt.GridSpec(nrows=2, ncols=3, height_ratios=[1, 0.05])

modes: List[torch.Tensor] = utils.get_singular_modes(original_run.rnn.weight_hh_l0)

for i, mode_i in enumerate([0, 1, 2]):
    plots.asp(gs[0, i])
    plt.title(f'Mode {mode_i + 1}')
    m: NDArray = modes[i][argsorted_metrics['displacement'], :][:, argsorted_metrics['displacement']].clone().numpy()
    m = gaussian_filter(m, sigma=4.0, mode='nearest')
    sns.heatmap(m, center=0, cmap='bwr', cbar=False, square=True)

    plt.yticks([1, 256, 512])
    plt.xticks([1, 256, 512])

# centered colorbar under the middle heatmap
plots.asp(gs[1, 1])
norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
sm = plt.cm.ScalarMappable(norm=norm, cmap='bwr')
cb = plt.colorbar(sm, cax=plt.gca(), orientation='horizontal', ticks=[-1, 0, 1])
cb.set_ticklabels(['Neg', '0', 'Pos'])
cb.set_label('Synaptic weight')

plots.supylabel('Target neuron', x=0.05, y=0.71)
plots.supxlabel('Source neuron', y=0.28)

analysis.save_figure(figure_path / 'heatmap_modes.png', dpi=300, allow_non_pdf=True)  # v. large file if pdf! (~15 MB)
plt.show()

# %%
modes: List[torch.Tensor] = utils.get_singular_modes(original_run.rnn.weight_hh_l0)
u, s, v = torch.linalg.svd(original_run.rnn.weight_hh_l0.detach(), full_matrices=False)

for i, mode_i in enumerate([0, 1, 2, 99, 511]):
    fig = plt.figure(figsize=plots.A5_L * [1 / 7, 1 / 4], constrained_layout=True)
    plt.title(f'{mode_i + 1}')
    m: NDArray = (
        modes[mode_i][argsorted_metrics['displacement'], :][:, argsorted_metrics['displacement']].clone().numpy()
    )
    m = gaussian_filter(m, sigma=4.0, mode='nearest')
    sns.heatmap(m, center=0, cmap='bwr', cbar=False, square=True)

    plt.yticks([])
    plt.xticks([])

    plt.tight_layout()
    analysis.save_figure(figure_path / f'heatmap_modes_{mode_i}.png', dpi=300, allow_non_pdf=True)
    plt.show()


# %%
# Rank subtraction
df: pd.DataFrame = torch.load(original_run.model_path / 'derivatives/rank/df.pkl')
df = df[df['mode'].str.startswith('scale_')]
alpha_df: pd.DataFrame = df[(df['alpha'] == 0.0) & (df['speed'] == speed)]
print(alpha_df)

plt.figure(figsize=plots.A5_L * [1 / 4, 1 / 3])
sns.lineplot(data=alpha_df, x=alpha_df['k'] + 1, y='nb_reward_perc', marker='o', markeredgecolor='none', alpha=0.7)
plt.tight_layout()
plt.xlabel('Mode * 0.0')
plt.ylabel('Uncued reward %')
ticks = [1, 256, 512]
plt.xticks(ticks, [str(t) for t in ticks])

analysis.save_figure(figure_path / 'rank_subtraction.pdf')
plt.show()

# %%
# Rank subtraction
df: pd.DataFrame = torch.load(original_run.model_path / 'derivatives/rank/df.pkl')
df = df[df['mode'].str.startswith('scale_') & (df['k'] < 8) & (df['speed'] == speed)]  # TODO rnd slow

plt.figure(figsize=plots.A5_L * [1 / 4, 1 / 3])
sns.lineplot(data=df, x=df['k'] + 1, y='nb_reward_perc', marker='o', markeredgecolor='none', alpha=0.7, hue='alpha')
plt.xlabel('Mode * scaling factor')
plt.ylabel('Uncued reward %')

# Update legend labels to show 2 decimal places
handles = plt.gca().get_legend().legendHandles
labels = [f'{float(t.get_label()):.2f}' for t in handles]
plt.legend(
    handles,
    labels,
    frameon=False,
    bbox_to_anchor=(0.99, 0.85),
    loc='upper left',
    fontsize=7,
    handlelength=1.0,
    title='Scaling\nfactor',
    title_fontsize=7,
)

plt.xticks([4, 8])

plots.tight_layout(right=1.06)

analysis.save_figure(figure_path / 'rank_scaling.pdf')


plt.show()


# %%
ks: List[int] = [0, 1, 2, 3]
plt.figure(figsize=plots.A5_L * [1 / 4, 1 / 3])
gs = plt.GridSpec(nrows=4, ncols=1)

alpha: float = 0.75
for i, (_, row) in enumerate(df[df['k'].isin(ks) & (df['alpha'] == alpha) & (df['speed'] == speed)].iterrows()):
    plots.asp(gs[i])
    k = row['k']
    run: Run = torch.load(EXPERIMENTS / Path(row['record_path']))['run']
    max_stops: int = analysis.get_max_stops_for_dist([run], trial_types=('nb',))
    analysis.plot_stop_distributions(run, trial_types=['nb'], divide_y_by=max_stops, below_stops=False, plot_base=True)

    plots.strip_plot_axes()
    plt.ylim(0, 1)

    plt.gca().annotate(f'{k + 1}', xy=(1.09, 0.5), xycoords='axes fraction', va='center', ha='center', rotation=0)

    if i == 3:
        plt.xticks([0, 100, 200])
        plt.xlabel('Location')

plt.gcf().text(s=f'Mode * {alpha:.2f}', x=0.91, y=0.6, va='center', ha='left', rotation=-90)

plots.supylabel('Stops', x=0.09)
plots.tight_layout(right=0.97)

analysis.save_figure(figure_path / 'rank_scaling_behaviour.pdf')
plt.show()
