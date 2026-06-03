# %%
import itertools
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import torch
from matplotlib.ticker import ScalarFormatter

import sf.analysis as analysis
import sf.plots as plots
import sf.utils as utils
from sf.analysis import get_fig_dir
from sf.notebook_setup import MIDDLE_CONTEXT, setup_notebook
from sf.specific.scaling import RZ_MAXES, RZ_MINS
from sf.specific.systematic_perturbations import CONDITION_COLOURS, CONDITION_LABELS, load_scaling_df, load_zeroing_df
from sf.utils import Run

setup_notebook()
analysis.make_paper_theme()

figure_path: Path = get_fig_dir('single_neurons_exemplar')


run: Run = torch.load(analysis.get_exemplar_model() / 'derivatives/run.pkl')

# %%
zeroing_df: pd.DataFrame = load_zeroing_df(run.model_path)
scaling_df: pd.DataFrame = load_scaling_df(run.model_path)

# %%
CONDITION_COLOURS['Displacement'] = '#8c564b'
CONDITION_COLOURS['Location r'] = 'orange'

# %%
subset_conditions: List[str] = ['Displacement', 'Random']  # noqa


def create_plot(
    df: pd.DataFrame,
    descending_val: bool,
    reward_type: Literal['nb', 'b'],
    show_legend: bool = False,
    subset: bool = False,
):
    if subset:
        df = df[df['condition'].isin(subset_conditions)]

    plots.make_figure(w=1.45)
    plot = sns.lineplot(
        data=df.query(f'descending=={descending_val}'),
        x='n_perturbed',
        y=f'{reward_type}_reward_perc',
        hue='condition',
        palette=CONDITION_COLOURS,
        legend=show_legend,
        marker='o',
        markeredgecolor='none',
    )
    plt.ylim(-5.0, 100)
    plt.ylabel('')
    plt.xlabel('')
    if not show_legend:
        analysis.save_figure(
            figure_path
            / 'zeroing'
            / ('subset' if subset else 'all')
            / f'{reward_type}_{"desc" if descending_val else "asc"}.pdf'
        )

    return plot


for subset in [True, False]:
    for descending, reward_type in itertools.product([True, False], ['nb', 'b']):
        create_plot(zeroing_df, descending, reward_type, subset=subset)
        plt.show()

    # Create legend-only plot
    plots.make_figure(w=1.25)
    legend_plot = create_plot(zeroing_df, True, 'nb', show_legend=True, subset=subset)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    analysis.save_figure(figure_path / f'legend_subset{subset}.pdf')
    plt.show()

# %%
plots.set_font_size(9)

fig, gs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=plots.A5_L * [0.5, 1 / 3])

subset_df: pd.DataFrame = zeroing_df[zeroing_df['condition'].isin(subset_conditions)]

for i, trial_type in enumerate(['nb', 'b']):
    descending = True
    plt.sca(gs[i])

    sns.lineplot(
        subset_df[subset_df['descending'] == descending],
        x='n_perturbed',
        y=f'{trial_type}_reward_perc',
        hue='condition',
        marker='o',
        markeredgecolor='none',
        legend=i == 1,
        palette=CONDITION_COLOURS,
    )

    plt.ylim(-5.0, 100)

    plt.ylabel('% reward')
    plt.ylabel('')
    plt.xlabel('')
    plt.title({'nb': 'Uncued', 'b': 'Cued'}[trial_type])

    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(50))


fig.supylabel('% reward', y=0.55, x=0.05)

sns.move_legend(
    plt.gca(),
    'center right',
    frameon=False,
    title=None,
    ncol=1,
    # bbox_to_anchor=(0.57, -0.01),
    bbox_to_anchor=(1.02, 0.55),  # something like this if using 'cente right'
    bbox_transform=plt.gca().figure.transFigure,
    handlelength=0.5,
    handletextpad=0.25,
    columnspacing=0.7,
    fontsize=9,
)
# plt.tight_layout(rect=(0.00, 0.03, 1.00, 1.00))  # Might want to add some spacing at the bottom
plots.tight_layout(bottom=0.1, right=0.91)

mid = plots.get_midpoint_x()
fig.supxlabel('Neurons zeroed', y=0.055, x=mid, ha='center')

analysis.save_figure(figure_path / 'zeroed_subset.pdf')
plt.show()


# %%
plots.set_font_size(9)

fig, gs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=plots.A5_L * [0.5, 1 / 3])

subset_df: pd.DataFrame = zeroing_df[zeroing_df['condition'].isin(subset_conditions)]

for i, trial_type in enumerate(['nb', 'b']):
    descending = True
    plt.sca(gs[i])

    df: pd.DataFrame = subset_df[subset_df['descending'] == descending]
    lineplot = sns.lineplot(
        df,
        x='n_perturbed',
        y=f'{trial_type}_displacement_mean',
        hue='condition',
        marker='o',
        markeredgecolor='none',
        legend=i == 1,
        estimator=None,
        palette=CONDITION_COLOURS,
    )

    for condition in subset_df.condition.unique():
        d = df.sort_values('n_perturbed').query(f'condition == "{condition}"')
        plt.fill_between(
            d['n_perturbed'],
            d[f'{trial_type}_displacement_mean'] - d[f'{trial_type}_displacement_sd'],
            d[f'{trial_type}_displacement_mean'] + d[f'{trial_type}_displacement_sd'],
            color=CONDITION_COLOURS[condition],
            alpha=0.5,
        )

    analysis.plot_stops_base_vert()

    plt.ylabel('')
    plt.xlabel('')
    plt.title({'nb': 'Uncued', 'b': 'Cued'}[trial_type])

    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(50))


fig.supylabel('Stop readout peak', y=0.55, x=0.05)

sns.move_legend(
    plt.gca(),
    'center right',
    frameon=False,
    title=None,
    ncol=1,
    # bbox_to_anchor=(0.57, -0.01),
    bbox_to_anchor=(1.02, 0.55),  # something like this if using 'cente right'
    bbox_transform=plt.gca().figure.transFigure,
    handlelength=0.5,
    handletextpad=0.25,
    columnspacing=0.7,
    fontsize=9,
)
# plt.tight_layout(rect=(0.00, 0.03, 1.00, 1.00))  # Might want to add some spacing at the bottom
plots.tight_layout(bottom=0.1, right=0.91)

mid = plots.get_midpoint_x()
fig.supxlabel('Neurons zeroed', y=0.055, x=mid, ha='center')

analysis.save_figure(figure_path / 'zeroed_subset_displacement.pdf')
plt.show()


# %%
plots.set_font_size(9)

fig, gs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=plots.A5_L * [0.5, 1 / 3])

subset_df: pd.DataFrame = zeroing_df[zeroing_df['condition'].isin(subset_conditions)]

for i, trial_type in enumerate(['nb', 'b']):
    descending = True
    plt.sca(gs[i])

    df: pd.DataFrame = subset_df[subset_df['descending'] == descending]
    lineplot = sns.lineplot(
        df,
        x='n_perturbed',
        y=f'mean_first_{trial_type}_stop',
        hue='condition',
        marker='o',
        markeredgecolor='none',
        legend=i == 1,
        estimator=None,
        palette=CONDITION_COLOURS,
    )

    analysis.plot_stops_base_vert()

    plt.ylabel('')
    plt.xlabel('')
    plt.title({'nb': 'Uncued', 'b': 'Cued'}[trial_type])

    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(50))


fig.supylabel('Avg. first stop', y=0.55, x=0.05)

sns.move_legend(
    plt.gca(),
    'center right',
    frameon=False,
    title=None,
    ncol=1,
    # bbox_to_anchor=(0.57, -0.01),
    bbox_to_anchor=(1.02, 0.55),  # something like this if using 'cente right'
    bbox_transform=plt.gca().figure.transFigure,
    handlelength=0.5,
    handletextpad=0.25,
    columnspacing=0.7,
    fontsize=9,
)
# plt.tight_layout(rect=(0.00, 0.03, 1.00, 1.00))  # Might want to add some spacing at the bottom
plots.tight_layout(bottom=0.1, right=0.91)

mid = plots.get_midpoint_x()
fig.supxlabel('Neurons zeroed', y=0.055, x=mid, ha='center')

analysis.save_figure(figure_path / 'zeroed_subset_avgfirststop.pdf')
plt.show()


# %%
def plot_systematic_scaling_modern(results_df: pd.DataFrame):
    plots.set_font_size(9)
    plt.figure(figsize=plots.A5_L * [1 / 5, 1 / 3])
    ax = plt.gca()

    trial_type = 'nb'

    for condition in results_df.condition.unique():
        df_c: pd.DataFrame = results_df[results_df['condition'] == condition].sort_values('scaling_factor')
        plt.plot(
            df_c['scaling_factor'],
            df_c['nb_displacement_mean'],
            color=CONDITION_COLOURS[condition],
            alpha=0.7,
            marker='o',
            linestyle='-',
            markersize=3.5,
            zorder=2.0,
        )
        plt.fill_between(
            df_c['scaling_factor'],
            df_c[f'{trial_type}_displacement_mean'] - df_c[f'{trial_type}_displacement_sd'],
            df_c[f'{trial_type}_displacement_mean'] + df_c[f'{trial_type}_displacement_sd'],
            color=CONDITION_COLOURS[condition],
            alpha=0.5,
        )

    # TODO: refactor - use track background plotting helper?
    ax.axhspan(0, 30, alpha=0.15, color='black')
    blackbox2_end: int = 60 + 60 + 20 + 30 + 30
    ax.axhspan(60 + 60 + 20 + 30, blackbox2_end, alpha=0.15, color='black')
    ax.axhspan(RZ_MINS[MIDDLE_CONTEXT], RZ_MAXES[MIDDLE_CONTEXT], alpha=0.15, color='green')

    plt.ylim(0, 200)
    plt.yticks([0, 50, 100, 150, 200])

    ax.set_xticks([0.3, 1, 2])

    xmin, xmax = ax.get_xlim()

    x_smooth = np.linspace(xmin, xmax, 512)
    anchor = 30.0
    rz_min0 = RZ_MINS[MIDDLE_CONTEXT]
    rz_max0 = RZ_MAXES[MIDDLE_CONTEXT]

    # TODO: Might not need this
    s_safe = np.clip(x_smooth, 1e-6, None)
    y_min = anchor + (rz_min0 - anchor) / s_safe
    y_max = anchor + (rz_max0 - anchor) / s_safe

    ax.fill_between(x_smooth, y_min, y_max, color='orange', alpha=0.40, linewidth=0, zorder=1.0)

    plt.xlim(xmin, xmax)

    plt.xlabel('Scaling factor')
    plt.ylabel('Stop readout peak')
    plt.tight_layout()


plot_systematic_scaling_modern(scaling_df[scaling_df['descending'] & scaling_df['condition'].isin(subset_conditions)])
analysis.save_figure(figure_path / 'scaling_modern_displacement.pdf')
plt.show()


# %%
def make_time_to_zero_df(trial_type: Literal['b', 'nb'], zero_df: pd.DataFrame = zeroing_df) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for ind in zero_df.indices.unique():
        df: pd.DataFrame = zero_df[zero_df.indices == ind]

        n_perturbed: npt.NDArray[np.int32] = df.n_perturbed.to_numpy()
        assert utils.is_monotonic_increasing(n_perturbed), n_perturbed

        rewards: npt.NDArray[np.float32]
        if trial_type == 'b':
            rewards = df.b_reward_perc.to_numpy()
        elif trial_type == 'nb':
            rewards = df.nb_reward_perc.to_numpy()
        else:
            raise ValueError

        result: int = max(n_perturbed)
        for n, r in zip(n_perturbed, rewards):
            if (np.abs(r) < 50) or np.isnan(r):
                result = n
                print(ind, result)
                break

        records.append({'indices': ind, 'n_perturbed': result, 'condition': CONDITION_LABELS[ind[1:]]})

    return pd.DataFrame(records)


perturbations_to_keep: List[str] = [
    'location_correlation',
    'speed_correlation',
    'pca0',
    'pca1',
    'pca2',
    'random',
    'displacement',
]


def plot_time_to_zero():
    plots.set_font_size(9)

    fig = plt.figure(figsize=plots.A5_L * [2 / 5, 2 / 5])
    gs = plt.GridSpec(2, 2)

    shared_ax: Optional[plt.Axes] = None
    for column_i, trial_type in enumerate(['nb', 'b']):
        for row_i, (direction_readable, direction) in enumerate(zip(['pos', 'neg'], ['+', '-'])):
            df: pd.DataFrame = make_time_to_zero_df(
                trial_type
            )  # redundant to do this again but it is deterministic.. TODO: refactor
            df = df[df['indices'].str.startswith(direction)]
            df['indices'] = pd.Categorical(
                df['indices'], categories=[f'{direction}{k}' for k in perturbations_to_keep], ordered=True
            )

            plt.sca(fig.add_subplot(gs[row_i, column_i], sharey=shared_ax))
            shared_ax = plt.gca()
            sns.barplot(
                df,
                x='condition',
                y='n_perturbed',
                color='black',
                order=['Displacement', 'Location r', 'Speed r', 'PC1', 'PC2', 'PC3', 'Random'],
            )  # TODO: assert these are in the df
            plt.gca().set_xticklabels(
                ['Displ.', 'Location r', 'Speed r', 'PC1', 'PC2', 'PC3', 'Random'], rotation=45, ha='right'
            )
            plt.xticks(rotation=45, ha='right')

            if column_i != 0:
                plt.gca().yaxis.set_visible(False)
                plt.gca().annotate(
                    f'{direction}', xy=(1.05, 0.5), xycoords='axes fraction', va='center', ha='left', rotation=0
                )

            if row_i != (gs.nrows - 1):
                plt.gca().xaxis.set_visible(False)
                plt.title({'nb': 'Uncued', 'b': 'Cued'}[trial_type])

            plt.ylabel('')
            plt.xlabel('')
            plt.yticks([0, 50, 100], [0, 50, '≥100'])

    fig.supylabel('Num. zeroed for\n50% reward', y=plots.get_midpoint_y() * 1.2, ha='center', x=0.05)
    fig.supxlabel('Selection metric', x=plots.get_midpoint_x() * 1.05, y=0.05)

    plt.tight_layout(pad=0.7, w_pad=-0.3)


plot_time_to_zero()
analysis.save_figure(figure_path / 'time_to_zero.pdf')
plt.show()
