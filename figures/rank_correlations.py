# %%
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import pearsonr

import sf.analysis as analysis
import sf.attribution.disruption as disruption
import sf.plots as plots
from sf import EXPERIMENTS
from sf.analysis import get_fig_dir
from sf.notebook_setup import setup_notebook
from sf.specific.systematic_perturbations import a_location_correlations
from sf.utils import Run

setup_notebook()
analysis.make_paper_theme()

figure_path: Path = get_fig_dir('rank_correlations')

original_run: Run = analysis.get_hrun(analysis.get_exemplar_model())

df: pd.DataFrame = torch.load(EXPERIMENTS / original_run.model_path / 'derivatives/rank/df.pkl')

plots.set_font_size(11)
location_correlations: np.ndarray = a_location_correlations(original_run)
pc2_loadings: np.ndarray = analysis.fit_pca(original_run).components_[1]
disruptions: disruption.DisruptionResults = disruption.calculate_disruption_scores(
    original_run, preprocessing='softmax'
)

u, s, v = torch.linalg.svd(original_run.rnn.weight_hh_l0.detach(), full_matrices=True)


# %%
def plot_metric(metric, metric_name):
    n_modes: int = 10
    plt.figure(figsize=plots.A4 * [1 / 2, 1])
    gs = plt.GridSpec(nrows=n_modes, ncols=2)

    for mode_i in range(n_modes):
        for vec_i, (axis, loadings) in enumerate(zip(['u', 'k'], [u.T, v])):
            plots.asp(gs[mode_i, vec_i])
            sns.scatterplot(x=metric, y=loadings[mode_i])

            if mode_i != (n_modes - 1):
                plt.xlabel(None)
                plt.xticks([])
            else:
                plt.xlabel(metric_name)
            plt.ylabel(f'${axis}_{mode_i + 1}$ coefficient')

            r = pearsonr(metric, loadings[mode_i]).statistic
            plt.title(f'r = {r:.2f}')

        plt.gca().annotate(
            f'Mode {mode_i + 1}', xy=(1.05, 0.5), xycoords='axes fraction', va='center', ha='left', rotation=-90
        )

    plots.tight_layout()
    plt.gcf().align_ylabels()


plot_metric(disruptions.displacements.numpy(), 'Displacement')
analysis.save_figure(figure_path / 'mode_vs_displacement_plot.png', dpi=300)
plt.show()

plot_metric(location_correlations, 'Location correlation')
analysis.save_figure(figure_path / 'mode_vs_locr_plot.png', dpi=300)
plt.show()

# %%
plot_metric(pc2_loadings, 'PC2 loading')
analysis.save_figure(figure_path / 'mode_vs_pc2_plot.png', dpi=300)
plt.show()


# %%
def compute_metrics() -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    n_modes: int = 512
    for metric_name, metric in (
        ('Displacement', disruptions.displacements.numpy()),
        ('Location correlation', location_correlations),
        ('PC2 loading', pc2_loadings),
    ):
        for mode_i in range(n_modes):
            for vec_i, (axis, loadings) in enumerate(zip(['u', 'k'], [u.T, v])):
                r = pearsonr(metric, loadings[mode_i]).statistic
                records.append(
                    {
                        'r': r,
                        'r_abs': np.abs(r),  # TODO: could do this at the end
                        'mode': mode_i + 1,
                        'metric_name': metric_name,
                        'axis': axis,
                    }
                )
    return pd.DataFrame(records)


df: pd.DataFrame = compute_metrics()


# %%
def plot_summary():
    plt.figure(figsize=plots.A5_L * [1, 2 / 3])
    gs = plt.GridSpec(ncols=len(df.metric_name.unique()), nrows=2)

    for metric_i, metric_name in enumerate(df.metric_name.unique()):
        plots.asp(gs[0, metric_i])
        plt.title(f'{metric_name}')
        sns.lineplot(
            df[df.metric_name == metric_name].query('axis == "u"'),
            x='mode',
            y='r_abs',
            marker='o',
            markeredgecolor=None,
        )
        plt.ylabel('r: u vs metric')
        plt.xlabel('')
        plt.xticks([])
        plt.ylim(-0.1, 0.9)

        plots.asp(gs[1, metric_i])
        sns.lineplot(
            df[df.metric_name == metric_name].query('axis == "k"'),
            x='mode',
            y='r_abs',
            marker='o',
            markeredgecolor=None,
        )
        plt.ylabel('r: k vs metric')
        plt.xlabel('Mode')
        plt.xticks([1, 256, 512])
        plt.ylim(-0.1, 0.9)

    plt.tight_layout()


plot_summary()
analysis.save_figure(figure_path / 'correlation_summary_plot.png', dpi=300)
plt.show()
