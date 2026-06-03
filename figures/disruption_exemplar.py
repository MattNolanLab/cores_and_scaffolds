# %%
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import pearsonr, rankdata

import sf.attribution.disruption as disruption
from sf import analysis, plots, specific
from sf.analysis import get_fig_dir
from sf.specific.disruption_perturbations import get_argsorted_metrics
from sf.utils import Run

analysis.make_paper_theme()
plots.set_font_size(11)

figure_path: Path = get_fig_dir('disruption_exemplar')

# %%
original_run: Run = analysis.get_hrun(analysis.get_exemplar_model())
correlations: torch.Tensor = torch.tensor(specific.systematic_perturbations.a_location_correlations(original_run))
disruptions: disruption.DisruptionResults = disruption.calculate_disruption_scores(
    original_run, preprocessing='softmax'
)
pc2_loadings: np.ndarray = analysis.fit_pca(original_run).components_[1]
argsorted_metrics: Dict[str, torch.Tensor] = get_argsorted_metrics(original_run)

# %%
plots.set_font_size(9)
exemplar_i: int = 1

fig = plt.figure(figsize=plots.A5_L * [2 / 4, 1 / 4])
gs = plt.GridSpec(1, 2)

plt.sca(fig.add_subplot(gs[0, 0]))
disruption.plot_disruption_result(disruptions, exemplar_i, False)

# Put 'Go' legend on top
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(
    handles[::-1],
    labels[::-1],
    frameon=False,
    loc='upper left',
    bbox_to_anchor=(-0.00, 0.95),
    ncol=1,
    fontsize=8,
    handlelength=0.5,
    handletextpad=0.25,
    columnspacing=0.2,
)

plt.sca(fig.add_subplot(gs[0, 1]))
disruption.plot_disruption_result(disruptions, exemplar_i, True)

plt.tight_layout(w_pad=0.1)

analysis.save_figure(figure_path / 'example_neuron_joint.pdf')
plt.show()

# %%
sns.jointplot(
    y=disruptions.displacements, x=pc2_loadings, marginal_kws=dict(bins=20), height=50 / 10 / 2.25, alpha=0.25
)
plt.xlabel('PC2 loading')
plt.ylabel('Displacement')
plt.tight_layout()
analysis.save_figure(figure_path / 'pc_vs_displacement.pdf')
plt.show()


# %%
def plot_hist(x: np.ndarray) -> None:
    plt.hist(x, color='#444444')
    plt.axvline(np.percentile(x, 10), linewidth=1)
    plt.axvline(np.percentile(x, 90), linewidth=1)


plots.set_font_size(9)

plt.figure(figsize=np.array([4.0, 3.6]) / 2.25)
plot_hist(disruptions.delta_heights.numpy())
plt.ylabel('Count')
plt.xlabel('$\Delta$ peak height')
plt.tight_layout()
analysis.save_figure(figure_path / 'distribution_peakheight.pdf')
plt.show()


# %%
plt.figure(figsize=np.array([4.0, 3.6]) / 2.25)
plot_hist(disruptions.displacements.numpy() / 2)
plt.ylabel('Count')
plt.xlabel('Displacement %')
plt.tight_layout()
analysis.save_figure(figure_path / 'distribution_displacement.pdf')
plt.show()


# %%
plt.figure(figsize=np.array([4.0, 3.6]) / 2.25)
plot_hist(correlations)
plt.ylabel('Count')
plt.xlabel('Location r')
plt.tight_layout()
analysis.save_figure(figure_path / 'distribution_location.pdf')
plt.show()

# %%
plt.figure(figsize=np.array([4.0, 3.6]) / 2.25)
plot_hist(pc2_loadings)
plt.ylabel('Count')
plt.xlabel('PC2 loading')
plt.tight_layout()
analysis.save_figure(figure_path / 'distribution_pc2.pdf')
plt.show()


# %%
plots.set_font_size(9)
plt.figure(figsize=np.array([4.0, 3.6]) / 2.25)
plt.scatter(disruptions.delta_heights.numpy(), disruptions.displacements.numpy() / 2, alpha=0.7)
plt.xlabel('$\Delta$ peak height')
plt.ylabel('Displacement %')
plt.tight_layout()
analysis.save_figure(figure_path / 'peak_vs_displacement_scatter.pdf')
plt.show()


# %%
plots.set_font_size(9)

plt.figure(figsize=np.array([4.0, 3.6]) / 2.25)
plt.scatter(correlations, disruptions.displacements.numpy() / 2, alpha=0.7)
plt.xlabel('Location r')
plt.ylabel('Displacement %')
plt.tight_layout()
analysis.save_figure(figure_path / 'displ_vs_location_scatter.pdf')
plt.show()

# %%
plt.figure(figsize=np.array([4.0, 3.6]) / 2.25)
plt.scatter(pc2_loadings, disruptions.displacements.numpy() / 2, alpha=0.7)
plt.xlabel('PC2 loading')
plt.ylabel('Displacement %')
plt.tight_layout()
analysis.save_figure(figure_path / 'pc2_vs_displ_scatter.pdf')
plt.show()


# %%
plots.set_font_size(9)


def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis='x', labelbottom=False, bottom=False)
    ax_histy.tick_params(axis='y', labelleft=False, left=False)

    # the scatter plot:
    ax.scatter(x, y, alpha=0.3)

    # now determine nice limits by hand:
    ax_histx.hist(x, color='#444444')

    ax_histy.hist(y, orientation='horizontal', color='#444444')


def marginal_hist_orig(x: np.ndarray, y: np.ndarray):
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html
    fig = plt.figure(layout='constrained', figsize=[8 / 2.25] * 2)
    ax = fig.add_subplot()
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    sns.despine(ax=ax_histx)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    sns.despine(ax=ax_histy)

    scatter_hist(x, y, ax, ax_histx, ax_histy)


marginal_hist_orig(correlations.numpy(), disruptions.displacements.numpy() / 2)
plt.xlabel('Location r')
plt.ylabel('Displacement %')
plt.tight_layout()
analysis.save_figure(figure_path / 'marginal.pdf')
plt.show()

# %%
marginal_hist_orig(pc2_loadings, disruptions.displacements.numpy() / 2)
plt.xlabel('PC2 loading')
plt.ylabel('Displacement %')
plt.tight_layout()
analysis.save_figure(figure_path / 'marginal_pc2.pdf')
plt.show()


# %%
plots.set_font_size(9)
plt.figure(figsize=np.array([4.0 * 1.0, 3.6]) / 2.25)
sns.kdeplot(x=rankdata(correlations.numpy()), y=rankdata(disruptions.displacements.numpy()), linewidths=0.8)
plt.tight_layout()
plt.xticks([1, 256, 512])
plt.yticks([1, 256, 512])
plt.xlabel('Location r rank')
plt.ylabel('Displacement rank')
plt.tight_layout()
analysis.save_figure(figure_path / 'copula_location_vs_displ.pdf')
plt.show()


plt.figure(figsize=np.array([4.0, 3.6]) / 2.25)
sns.kdeplot(x=rankdata(pc2_loadings), y=rankdata(disruptions.displacements.numpy()), linewidths=0.8)
plt.tight_layout()
plt.xticks([1, 256, 512])
plt.yticks([1, 256, 512])
plt.xlabel('PC2 rank')
plt.ylabel('Displacement rank')
plt.tight_layout()
analysis.save_figure(figure_path / 'copula_pc2_vs_displ.pdf')
plt.show()


# %%
plt.figure(figsize=np.array([4.0 * 1.20, 3.6]) / 2.25)
n = 40
top40_idx = torch.argsort(correlations, descending=True)[:n]

corr_rank_full = torch.empty_like(correlations, dtype=torch.long)
corr_rank_full[top40_idx] = torch.arange(1, n + 1)

disp_rank_full = torch.empty_like(disruptions.displacements, dtype=torch.long)
disp_rank_full[torch.argsort(disruptions.displacements, descending=True)] = torch.arange(
    1, len(disruptions.displacements) + 1
)

correlation_ranks = corr_rank_full[top40_idx]
displacement_ranks = disp_rank_full[top40_idx]

alpha = 0.7
plt.scatter([0] * n, correlation_ranks, color='black', alpha=alpha)
plt.scatter([1] * n, displacement_ranks, color='black', alpha=alpha)
plt.yticks([1, 50, 100], [1, 50, 100])

for a, b in zip(correlation_ranks, displacement_ranks):
    plt.plot([0, 1], [a, b], alpha=0.4, color='black')

plt.gca().invert_yaxis()
plt.xlim(-0.4, 1.4)
plt.xticks([0, 1], ['Location r', 'Displacement'])
plt.ylabel('Rank')
plt.title('Top 40 ranked by Location r')
plt.tight_layout()
analysis.save_figure(figure_path / 'rank.pdf')
plt.show()


# %%
# TODO: refactor this into plotting function above?

pca_loadings: torch.tensor = torch.tensor(np.copy(pc2_loadings))

plt.figure(figsize=np.array([4.0 * 1.20, 3.6]) / 2.25)
n = 40
top40_idx = torch.argsort(pca_loadings, descending=True)[:n]

corr_rank_full = torch.empty_like(pca_loadings, dtype=torch.long)
corr_rank_full[top40_idx] = torch.arange(1, n + 1)

disp_rank_full = torch.empty_like(disruptions.displacements, dtype=torch.long)
disp_rank_full[torch.argsort(disruptions.displacements, descending=True)] = torch.arange(
    1, len(disruptions.displacements) + 1
)

correlation_ranks = corr_rank_full[top40_idx]
displacement_ranks = disp_rank_full[top40_idx]

alpha = 0.7
plt.scatter([0] * n, correlation_ranks, color='black', alpha=alpha)
plt.scatter([1] * n, displacement_ranks, color='black', alpha=alpha)
plt.yticks([1, 50, 100], [1, 50, 100])

for a, b in zip(correlation_ranks, displacement_ranks):
    plt.plot([0, 1], [a, b], alpha=0.4, color='black')

plt.gca().invert_yaxis()
plt.xlim(-0.4, 1.4)
plt.xticks([0, 1], ['PC2 loading', 'Displacement'])
plt.ylabel('Rank')
plt.title('Top 40 ranked by PC2 loading')
plt.tight_layout()
analysis.save_figure(figure_path / 'rank_pc2.pdf')
plt.show()


# %%
# Supplemental figure panels
plots.set_font_size(11)
neurons: List[int] = np.unique(
    [
        torch.argmax(disruptions.displacements),
        torch.argmin(disruptions.displacements),
        torch.argmax(disruptions.scores),
        torch.argmin(disruptions.scores),
        torch.argmax(disruptions.heights),
        torch.argmin(disruptions.heights),
        10,
        20,
        30,
        *torch.argsort(disruptions.displacements, descending=True)[0:4].tolist(),
    ]
)

fig = plt.figure(figsize=plots.A4)
gs = plt.GridSpec(nrows=len(neurons), ncols=3)

logits_ax: Optional[plt.Axes] = None
for row_i, neuron_i in enumerate(neurons):
    plt.sca(fig.add_subplot(gs[row_i, 0]))
    analysis.plot_neurons_combined(original_run, [neuron_i], alpha=1)
    plt.ylabel('Activity')
    if row_i != (gs.nrows - 1):
        plt.gca().xaxis.set_visible(False)

    plt.sca(fig.add_subplot(gs[row_i, 1], sharey=logits_ax))
    logits_ax = plt.gca()
    disruption.plot_disruption_result(disruptions, neuron_i, False)
    if row_i != (gs.nrows - 1):
        plt.gca().xaxis.set_visible(False)
    plt.title(
        f'Neuron {neuron_i + 1}\ndispl. = {disruptions.displacements[neuron_i] / 2:.1f}%, $\Delta$peak = {disruptions.delta_heights[neuron_i]:.1f}'
    )

    if row_i == 0:
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(
            handles[::-1],
            labels[::-1],
            frameon=False,
            loc='upper left',
            bbox_to_anchor=(0.0000, 0.85),
            ncol=1,
            fontsize=8,
            handlelength=0.5,
            handletextpad=0.25,
            columnspacing=0.2,
        )

    plt.sca(fig.add_subplot(gs[row_i, 2]))
    disruption.plot_disruption_result(disruptions, neuron_i, True)

    if row_i != (gs.nrows - 1):
        plt.gca().xaxis.set_visible(False)


fig.align_ylabels()
plt.tight_layout()
analysis.save_figure(figure_path / 'neuron_examples_plot.png')
plt.show()


# %%
print('Correlation vs displacement pearson r = ', round(pearsonr(correlations, disruptions.displacements).statistic, 2))
print('PC2 vs displacement pearson r = ', round(pearsonr(pca_loadings, disruptions.displacements).statistic, 2))

pos_mask = correlations > 0
print(
    'Pos correlation only. Correlation vs displacement pearson r = ',
    round(pearsonr(correlations[pos_mask], disruptions.displacements[pos_mask]).statistic, 2),
)

pos_mask = pca_loadings > 0
print(
    'Pos pca loadings only. PC2 vs displacement pearson r = ',
    round(pearsonr(pca_loadings[pos_mask], disruptions.displacements[pos_mask]).statistic, 2),
)

print(len(set(argsorted_metrics['displacement'][0:8].tolist()) - set(argsorted_metrics['pc2'][0:8].tolist())))
# %%
