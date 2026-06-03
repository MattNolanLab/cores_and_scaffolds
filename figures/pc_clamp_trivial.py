# %%
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA

import sf.analysis as analysis
import sf.perturbations as perturbations
import sf.plots as plots
import sf.specific.systematic_perturbations as systematic_perturbations
from sf.analysis import get_fig_dir
from sf.utils import Run, get_top_indices, run_model

analysis.make_paper_theme()

figure_path: Path = get_fig_dir('pc_clamp_trivial')

model_path = analysis.get_exemplar_model()
derivatives_dir: Path = model_path / 'derivatives'

run: Run = torch.load(derivatives_dir / 'run.pkl')
by_pcs_kept: Dict[int, Run] = torch.load(derivatives_dir / 'pc_inclusions.pkl')
by_pcs_clamped: Dict[int, Run] = torch.load(derivatives_dir / 'pc_clamps.pkl')


# %%
exemplar_pc: int = 5  # PC 6
location_correlations: torch.Tensor = torch.tensor(systematic_perturbations.a_location_correlations(run))
indices: torch.Tensor = get_top_indices(location_correlations, n=20, positive=False)

with perturbations.TransientWeightModificationRNN(run.rnn) as (ih, hh):
    hh *= 1.1
    perturbed_run: Run = run_model(run.model_path, core_override=run.rnn)[0]


pca: PCA = analysis.fit_pca(run)

# %%

plots.set_font_size(11)
fig = plt.figure(figsize=plots.A5_L / [1, 2.5])
alpha = 0.3
# First pair
plt.subplot(1, 4, 1)
analysis.plot_pca_mpl_2d(run, pca=pca, color='black', alpha=alpha)
analysis.plot_pca_mpl_2d(by_pcs_clamped[exemplar_pc], pca=pca, color='purple', alpha=alpha)
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.subplot(1, 4, 2)
analysis.plot_neurons_combined(run, indices)
analysis.plot_neurons_combined(by_pcs_clamped[exemplar_pc], indices, neuron_colour='purple')
plt.xticks([])
plt.ylabel('Avg. activity')

# Add label a
fig.text(0.001, 0.92, 'a', fontsize=16)

# Second pair
plt.subplot(1, 4, 3)
analysis.plot_pca_mpl_2d(run, pca=pca, color='black', label='Control', alpha=alpha)
analysis.plot_pca_mpl_2d(perturbed_run, pca=pca, color='purple', label='Perturbed', alpha=alpha)
plt.xlabel('PC1')
plt.ylabel('\n\nPC2')  # Hack to increase padding

plt.subplot(1, 4, 4)
analysis.plot_neurons_combined(run, indices)
analysis.plot_neurons_combined(perturbed_run, indices, neuron_colour='purple')
plt.xticks([])
plt.ylabel('Avg. activity')

# Add label b
fig.text(0.50, 0.92, 'b', fontsize=16)

fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.02), frameon=False, ncol=2)
plt.tight_layout(rect=[0, 0.08, 1, 0.85])

fig.text(0.25, 0.85, 'Actual perturbation: PC6 = 0', ha='center', va='bottom', fontsize=11, transform=fig.transFigure)
fig.text(
    0.77,
    0.85,
    'Trivial perturbation: hh weights × 110 %',
    ha='center',
    va='bottom',
    fontsize=11,
    transform=fig.transFigure,
)

analysis.save_figure(figure_path / 'example_plot.png')
plt.show()
