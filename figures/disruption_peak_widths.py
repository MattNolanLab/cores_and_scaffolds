# %%
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.signal import peak_widths

import sf.attribution.disruption as disruption
from sf import analysis, plots, specific
from sf.analysis import get_fig_dir
from sf.attribution import Action
from sf.attribution.disruption import DisruptionResults, find_highest_peak
from sf.utils import Run

analysis.make_paper_theme()
plots.set_font_size(9)

figure_path: Path = get_fig_dir('disruption_exemplar_peak_widths')

# %%
original_run: Run = analysis.get_hrun(analysis.get_exemplar_model())
correlations: torch.Tensor = torch.tensor(specific.systematic_perturbations.a_location_correlations(original_run))
disruptions: disruption.DisruptionResults = disruption.calculate_disruption_scores(
    original_run, preprocessing='softmax'
)


# %%
def plot(results: DisruptionResults) -> None:
    def t(x):
        return nn.functional.softmax(x, dim=-1)

    analysis.plot_stops_base()
    xs: List[float] = results.locations.tolist()
    for i in range(512):
        readout: torch.Tensor = results.readouts[i]
        v: torch.Tensor = t(readout)[:, Action.STOP]
        plt.plot(xs, v, color='black', alpha=0.2)
    plt.xlim(0, 200)


plot(disruptions)
plt.show()


# %% Peak height analysis
def peak_width(curve: torch.Tensor, xs: torch.Tensor, rel_height: float = 0.5) -> float:
    y = curve.detach().cpu().numpy()
    # x = xs.detach().cpu().numpy()
    p = find_highest_peak(y)[0]
    return peak_widths(y, [p], rel_height=rel_height)[0][0]


def calculate_widths(results: DisruptionResults) -> torch.Tensor:
    def t(x):
        return nn.functional.softmax(x, dim=-1)

    assert len(results.locations) == 170  # Only supports speed = 1 because of the mirroring trick

    widths: List[float] = []
    for i in range(512):
        readout: torch.Tensor = results.readouts[i]
        v: torch.Tensor = t(readout)[:, Action.STOP]
        widths.append(peak_width(torch.concat([v, torch.flip(v, [0])]), xs=torch.arange(300)))
        if any(v[160:]) > 0.01:  # Deal with peaks that haven't fallen by position 170 + the mirroring
            widths[-1] /= 2

    return torch.tensor(widths)


widths: torch.Tensor = calculate_widths(disruptions)


# %%
def plot(results: DisruptionResults) -> None:
    def t(x):
        return nn.functional.softmax(x, dim=-1)

    neurons = torch.argsort(widths, descending=True)[0:15]

    analysis.plot_stops_base()
    xs: List[float] = results.locations.tolist()
    for i in neurons:
        readout: torch.Tensor = results.readouts[i]
        v: torch.Tensor = t(readout)[:, Action.STOP]
        plt.plot(xs, v, color='black', alpha=0.2)
    plt.xlim(0, 200)


plot(disruptions)
plt.show()

# %%
plt.scatter(disruptions.displacements / 2, widths, alpha=0.2)
plt.show()

# %%
legend_fontsize: int = 16

fig = plt.figure(figsize=plots.A5_L * [1, 1 / 3])
gs = plt.GridSpec(ncols=4, nrows=1)

plt.sca(fig.add_subplot(gs[0]))
plot(disruptions)
plt.title('Top 15 widest')
plt.ylabel('Stop prob.')
fig.text(0.02, 0.91, 'a', fontsize=legend_fontsize)

plt.sca(fig.add_subplot(gs[1]))
plt.hist(widths)
plt.xlabel('Peak FWHM')
plt.ylabel('Count')
fig.text(0.26, 0.91, 'b', fontsize=legend_fontsize)

plt.sca(fig.add_subplot(gs[2]))
plt.scatter(disruptions.displacements / 2, widths, alpha=0.3)
plt.xlabel('Displacement %')
plt.ylabel('Peak FWHM')
fig.text(0.51, 0.91, 'c', fontsize=legend_fontsize)

plt.sca(fig.add_subplot(gs[3]))
plt.scatter(disruptions.delta_heights / 2, widths, alpha=0.3)
plt.xlabel('$\Delta$ peak height')
plt.ylabel('Peak FWHM')
fig.text(0.75, 0.91, 'd', fontsize=legend_fontsize)

plots.tight_layout(top=0.97, left=0.021)
analysis.save_figure(figure_path / 'peak_widths_plot.png')
plt.show()
