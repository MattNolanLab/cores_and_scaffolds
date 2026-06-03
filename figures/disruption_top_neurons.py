# %%
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import torch

import sf.analysis as analysis
import sf.attribution.disruption as disruption
import sf.plots as plots
import sf.specific.disruption_perturbations
from sf.analysis import get_fig_dir
from sf.utils import Run

analysis.make_paper_theme()
plots.set_font_size(9)

figure_path: Path = get_fig_dir('disruption_top_neurons')  # Legacy reasons: we split out these plots from that dir

# %%
original_run: Run = analysis.get_hrun(analysis.get_exemplar_model())

# %%
disruptions: disruption.DisruptionResults = disruption.calculate_disruption_scores(
    original_run, preprocessing='softmax'
)

# %%
argsorted_metrics: Dict[str, torch.Tensor] = sf.specific.disruption_perturbations.get_argsorted_metrics(original_run)

# %%
plt.figure(figsize=plots.A5_L * [1 / 2, 1 / 3])

n: int = 16
plt.subplot(1, 2, 1)
plt.ylabel('Avg activity')
plt.title(f'Top {n} displacement')
analysis.plot_neurons_combined(original_run, argsorted_metrics['displacement'][0:n])

plt.subplot(1, 2, 2)
plt.title(f'Bottom {n} displacement')
analysis.plot_neurons_combined(original_run, argsorted_metrics['displacement'][512 - n :])

plots.tight_layout()

analysis.save_figure(figure_path / 'top_disruption_plot.png')
plt.show()
