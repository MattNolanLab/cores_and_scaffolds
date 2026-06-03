# %%
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm.autonotebook import tqdm

import sf.analysis as analysis
import sf.plots as plots
from sf.analysis import get_fig_dir
from sf.attribution import disruption
from sf.plots import y_displacement
from sf.utils import Run

analysis.make_paper_theme()

figure_path: Path = get_fig_dir('pc_clamp_metrics')

model_path = analysis.get_exemplar_model()
derivatives_dir: Path = model_path / 'derivatives'

original_run: Run = torch.load(derivatives_dir / 'run.pkl')
by_pcs_kept: Dict[int, Run] = torch.load(derivatives_dir / 'pc_inclusions.pkl')
by_pcs_clamped: Dict[int, Run] = torch.load(derivatives_dir / 'pc_clamps.pkl')

plots.set_font_size(9)


# %%
# For each PC, compute the average first stop and average displacement
control_location = np.mean(disruption.calculate_run_displacements(original_run, 'nb').peak_locations)
print(control_location)


def make_displ_df(data: Dict[int, Run]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for trial_type in ['nb', 'b']:
        for pc, run in data.items():
            dis_results: disruption.RunDisplacementResults = disruption.calculate_run_displacements(run, trial_type)
            for i, dis in enumerate(dis_results.peak_locations):
                records.append({'pc': pc, 'trial': i, 'trial_type': trial_type, 'displacement': dis})

    return pd.DataFrame(records)


kept_displ_df: pd.DataFrame = make_displ_df(by_pcs_kept)

plt.figure(figsize=plots.A5_L * [1 / 4, 1 / 3])
analysis.plot_stops_base_vert()
for t, colour in (('b', 'blue'), ('nb', 'black')):
    sns.lineplot(
        kept_displ_df[kept_displ_df.trial_type == t],
        x='pc',
        y='displacement',
        errorbar='sd',
        marker='o',
        markeredgecolor='none',
        color=colour,
    )

y_displacement(control_location)
plt.xlabel('PCs kept')
plt.ylabel('Displacement %')
plt.xticks([4, 8, 12, 16])

plt.tight_layout()
analysis.save_figure(figure_path / 'pcs_kept_displ.pdf')

plt.show()


# %%
clamped_displ_df: pd.DataFrame = make_displ_df(by_pcs_clamped)
assert clamped_displ_df['pc'].min() == 0
clamped_displ_df['pc'] = clamped_displ_df['pc'] + 1  # Workaround for 0 indexing


plt.figure(figsize=plots.A5_L * [1 / 4, 1 / 3])
analysis.plot_stops_base_vert()
for t, colour in (('b', 'blue'), ('nb', 'black')):
    sns.lineplot(
        clamped_displ_df[clamped_displ_df.trial_type == t],
        x='pc',
        y='displacement',
        errorbar='sd',
        marker='o',
        markeredgecolor='none',
        color=colour,
    )

y_displacement(control_location)
plt.xlabel('PC clamped')
plt.ylabel('Displacement %')
plt.xticks([4, 8, 12])

plt.tight_layout()
analysis.save_figure(figure_path / 'pcs_clamped_displ.pdf')

plt.show()


# %%
def make_kept_stop_df(data: Dict[int, Run]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for pc, run in data.items():
        b_stops, nb_stops, _ = run.get_first_stop_locations()
        print(pc, len(b_stops), len(nb_stops))
        for trial_type, locations in [['nb', nb_stops], ['b', b_stops]]:
            if len(locations) > 3:
                for i, stop_loc in enumerate(locations):
                    records.append({'pc': pc, 'trial_type': trial_type, 'trial': i, 'stop': stop_loc})

    return pd.DataFrame(records)


kept_stop_df: pd.DataFrame = make_kept_stop_df(by_pcs_kept)

plt.figure(figsize=plots.A5_L * [1 / 4, 1 / 3])
analysis.plot_stops_base_vert()
for t, colour in (('b', 'blue'), ('nb', 'black')):
    sns.lineplot(
        kept_stop_df[kept_stop_df.trial_type == t],
        x='pc',
        y='stop',
        errorbar='sd',
        marker='o',
        markeredgecolor='none',
        color=colour,
    )

plt.xlabel('PCs kept')
plt.ylabel('Avg. first stop')
plt.xticks([4, 8, 12, 16])

plt.tight_layout()
analysis.save_figure(figure_path / 'pcs_kept_avgsfirsttop.pdf')

plt.show()

# %%
clamped_stop_df: pd.DataFrame = make_kept_stop_df(by_pcs_clamped)
clamped_stop_df['pc'] = clamped_stop_df['pc'] + 1  # Workaround for 0 indexing

plt.figure(figsize=plots.A5_L * [1 / 4, 1 / 3])
analysis.plot_stops_base_vert()
for t, colour in (('b', 'blue'), ('nb', 'black')):
    sns.lineplot(
        clamped_stop_df[clamped_stop_df.trial_type == t],
        x='pc',
        y='stop',
        errorbar='sd',
        marker='o',
        markeredgecolor='none',
        color=colour,
    )

plt.xlabel('PC clamped')
plt.ylabel('Avg. first stop')
plt.xticks([4, 8, 12])
plt.xlim(0, 16)

plt.tight_layout()
analysis.save_figure(figure_path / 'pcs_clamped_avgsfirsttop.pdf')

plt.show()

# %%

runs_across_track_clamp: Dict[int, Run] = torch.load(derivatives_dir / 'clamp_across_track.pkl')


# %%
def make_df_across():
    records: List[Dict[str, Any]] = []
    for trial_type in ['b', 'nb']:
        for location, run in tqdm(runs_across_track_clamp.items()):
            peak_locations = disruption.calculate_run_displacements(run, trial_type).peak_locations
            print(len(peak_locations))
            for displacement in peak_locations:
                records.append({'location': location, 'trial_type': trial_type, 'displacement': displacement})
    return pd.DataFrame(records)


df = make_df_across()


# %%
plt.figure(figsize=plots.A5_L * [1 / 4, 1 / 3])
analysis.plot_stops_base_vert()
analysis.plot_stops_base()
for t, colour in (('b', 'blue'), ('nb', 'black')):
    sns.lineplot(
        df[df.trial_type == t],
        x='location',
        y='displacement',
        errorbar='sd',
        marker='o',
        markeredgecolor='none',
        color=colour,
    )

    # sns.scatterplot(df[df.trial_type == t],
    #              x='location', y='displacement',
    #              marker='o',  color=colour, edgecolor='none', alpha=0.2)
    #

plt.xlabel('Clamp location')
plt.ylabel('Displacement %')
y_displacement(control_location)
plots.tight_layout()

analysis.save_figure(figure_path / 'systematic_clamp_readout.pdf')

plt.show()


# %%
def make_df_across():
    records: List[Dict[str, Any]] = []
    for trial_type in ['b', 'nb']:
        for location, run in tqdm(runs_across_track_clamp.items()):
            b_stops, nb_stops, _ = run.get_first_stop_locations()
            print(location, len(b_stops), len(nb_stops))
            for trial_type, locations in [['nb', nb_stops], ['b', b_stops]]:
                if len(locations) > 3:
                    for i, stop_loc in enumerate(locations):
                        records.append(
                            {'location': location, 'trial_type': trial_type, 'stop_loc': stop_loc, 'trial': i}
                        )
    return pd.DataFrame(records)


df = make_df_across()

# %%
plt.figure(figsize=plots.A5_L * [1 / 4, 1 / 3])
analysis.plot_stops_base_vert()
analysis.plot_stops_base()
for t, colour in (('b', 'blue'), ('nb', 'black')):
    sns.lineplot(
        df[df.trial_type == t],
        x='location',
        y='stop_loc',
        errorbar='sd',
        marker='o',
        markeredgecolor='none',
        color=colour,
    )

    # sns.scatterplot(df[df.trial_type == t],
    #              x='location', y='displacement',
    #              marker='o',  color=colour, edgecolor='none', alpha=0.2)
    #

plt.xlabel('Clamp location')
plt.ylabel('Avg. first stop')
plots.tight_layout()

analysis.save_figure(figure_path / 'systematic_clamp_avgfirststop.pdf')

plt.show()
