from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

import sf.analysis as analysis
import sf.plots as plots
import sf.subspace as subspace
from sf.analysis import get_fig_dir
from sf.notebook_setup import setup_notebook
from sf.utils import Run

setup_notebook()
analysis.make_paper_theme()

figure_path: Path = get_fig_dir('rank_main_information')

plots.set_font_size(9)

original_run = analysis.get_run(analysis.get_exemplar_model())
hrun = analysis.get_hrun(analysis.get_exemplar_model())

# %%
run: Run = original_run

mappings: Dict[str, torch.Tensor] = subspace.calculate_mapping_matrices(run)
latents: torch.Tensor = (mappings['hxs_to_latent'] @ run.states.T).T

n_modes: int = 48
latents_np: np.ndarray = latents[:, :n_modes].detach().cpu().numpy().T  # [mode, time]

n_components = latents_np.shape[0]
n_cols = 4
n_rows = (n_components + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=plots.A4, dpi=300)
axes = axes.flatten()

speeds = run.velocities_offset.detach().cpu().numpy()

for i in range(n_components):
    plt.sca(axes[i])

    # scatter coloured by speed
    sc = plt.scatter(
        run.locations.detach().cpu().numpy(), latents_np[i], s=4, c=speeds, cmap='Reds', alpha=0.7, edgecolors='none'
    )

    analysis.plot_stops_base()
    plots.despine_topright()
    analysis.strip_plot_axes()

    plt.title(f'Mode {i + 1}')
    if i % n_cols == 0:
        plt.ylabel('Activity')
    if i >= len(axes) - n_cols:
        plt.xlabel('Location')

for j in range(n_components, len(axes)):
    fig.delaxes(axes[j])

plots.tight_layout(right=0.88)

cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
cb = fig.colorbar(sc, cax=cax)
cb.set_label('Speed')

analysis.save_figure(figure_path / 'modes_across_track_speed.png', allow_non_pdf=True)
plt.show()


# %%
def make_continuous_cmap_from_palette(
    palette: Dict[float, tuple[float, float, float]], *, vmin: float, vmax: float, name: str = 'custom_vlag_continuous'
) -> matplotlib.colors.LinearSegmentedColormap:
    xs = np.array(sorted(palette.keys()), dtype=float)
    colors = np.array([palette[x] for x in xs], dtype=float)  # [n, 3]

    if xs[0] != vmin:
        xs = np.concatenate([[vmin], xs])
        colors = np.concatenate([colors[:1], colors], axis=0)
    if xs[-1] != vmax:
        xs = np.concatenate([xs, [vmax]])
        colors = np.concatenate([colors, colors[-1:]], axis=0)

    positions = (xs - vmin) / (vmax - vmin)

    return matplotlib.colors.LinearSegmentedColormap.from_list(
        name, list(zip(positions.tolist(), colors.tolist())), N=256
    )


def plot_modes_across_track_speed_row(
    run: Run,
    mode_ids: List[int],
    *,
    alpha: float = 0.7,
    before_final_speeds: List[float] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
    make_2p5_black: bool = True,
) -> None:
    mappings: Dict[str, torch.Tensor] = subspace.calculate_mapping_matrices(run)
    latents: torch.Tensor = (mappings['hxs_to_latent'] @ run.states.T).T  # [time, n_modes]

    fig = plt.figure(figsize=plots.A5_L * [1 / 2, 1.0 / 4])
    gs = plt.GridSpec(nrows=1, ncols=len(mode_ids))

    speeds = run.velocities_offset.detach().cpu().numpy()
    locations = run.locations_offset.detach().cpu().numpy()

    vmin = float(np.nanmin(speeds))
    vmax = float(np.nanmax(speeds))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)  # type: ignore

    palette = dict(zip(before_final_speeds, sns.color_palette('vlag', n_colors=len(before_final_speeds))))
    if make_2p5_black and 2.5 in palette:
        palette[2.5] = (0.0, 0.0, 0.0)

    cmap = make_continuous_cmap_from_palette(palette, vmin=vmin, vmax=vmax)

    for col_i, mode_i in enumerate(mode_ids):
        plt.sca(fig.add_subplot(gs[0, col_i]))
        analysis.plot_stops_base()

        ys = latents[:, mode_i].detach().cpu().numpy()
        plt.scatter(locations, ys, c=speeds, cmap=cmap, norm=norm, alpha=alpha, edgecolors='none')

        plt.title(f'Mode {mode_i + 1}')
        plt.xlabel('Location')
        plt.ylabel('')
        plt.yticks([])
        plt.xticks([0, 200])

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.90, 0.18, 0.02, 0.64])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('Speed')

    plots.tight_layout(right=0.90)


plot_modes_across_track_speed_row(run, mode_ids=[0, 1, 49, 99])
analysis.save_figure(figure_path / 'mode_activity.pdf')
plt.show()
