from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

import sf.analysis as analysis
import sf.plots as plots
import sf.style as style
from sf import EXP, EXPERIMENTS
from sf.analysis import get_fig_dir
from sf.utils import Run


analysis.make_paper_theme()

figure_path: Path = get_fig_dir('rank_lso_additional_metrics')


plots.set_font_size(9)

original_run = analysis.get_run(analysis.get_exemplar_model())
model_path: Path = EXP / original_run.model_path


# %%
def load_df():
    results: pd.DataFrame = torch.load(model_path / 'derivatives/lsornn/recurrent_upto.pkl')

    results['source_modes_zeroed'] = 512 - results['source_i']

    def pepper_avg_first_stop(row: pd.Series) -> dict[str, float]:
        run: Optional[Run] = torch.load(EXPERIMENTS / Path(row['record_path']))['run']

        values: dict[str, float] = {'cued_first_stop': np.nan, 'displacement_avg': np.nan, 'displacement_sd': np.nan}

        if run is not None:
            locations = run.get_uncued_first_stop_locations()
            if len(locations) > 5:
                values['uncued_first_stop'] = float(np.mean(locations))
            # Implicit else here... relies on opaque fallback behaviour

            displacements: np.ndarray = run.get_displacements('nb').peak_locations
            values['displacement_avg'] = float(np.mean(displacements))
            values['displacement_sd'] = float(np.std(displacements))
        return values

    return results.join(results.apply(pepper_avg_first_stop, axis=1, result_type='expand'))


df = load_df()


# %%
palette: dict[str, str] = {
    'Normal': style.SPEEDS_TO_COLOUR['standard'],
    'Rnd Slow': style.SPEEDS_TO_COLOUR['slow'],
    'Rnd Fast': style.SPEEDS_TO_COLOUR['fast'],
}

line_order = ['Rnd Slow', 'Normal', 'Rnd Fast']


def plot(df=df):
    plt.figure(figsize=plots.A5_L * [1, 1 / 2])
    gs = plots.GridSpec(ncols=len(df.latent_i.unique()), nrows=2)

    for col_i, (latent_i, ddf) in enumerate(df.groupby('latent_i')):
        plots.asp(gs[0, col_i])

        plt.title(f'Mode {latent_i + 1}')

        sns.lineplot(
            ddf,
            x='source_modes_zeroed',
            y='uncued_first_stop',
            hue='speed',
            marker='o',
            legend=False,
            palette=palette,
            hue_order=line_order,
            markeredgecolor='none',
        )
        plots.plot_track_vertical_base()
        plt.ylim(0, 200)
        plt.yticks([0, 200])
        plt.xlabel('')
        plt.xlim(-5, 515)
        plt.xticks()
        if col_i > 0:
            plots.hide_y()
        else:
            plt.ylabel('Avg. first uncued stop')

        plots.asp(gs[1, col_i])
        sns.lineplot(
            ddf,
            x='source_modes_zeroed',
            y='displacement_avg',
            hue='speed',
            marker='o',
            legend=False,
            palette=palette,
            hue_order=line_order,
            markeredgecolor='none',
        )
        plt.ylim(0, 200)
        plt.yticks([0, 200])
        plots.plot_track_vertical_base()
        plt.xticks([])
        plt.xlim(-5, 515)
        plt.xlabel('Source modes zeroed')

        for speed, speed_df in ddf.groupby('speed'):
            speed_df = speed_df.sort_values('source_modes_zeroed')

            xs = speed_df['source_modes_zeroed'].to_numpy(dtype=float)
            ys = speed_df['displacement_avg'].to_numpy(dtype=float)
            yerr = speed_df['displacement_sd'].to_numpy(dtype=float)

            plt.fill_between(xs, ys - yerr, ys + yerr, color=palette[speed], alpha=0.2, linewidth=0, zorder=0)

        if col_i > 0:
            plots.hide_y()
        else:
            plt.ylabel('Stop readout peak')

        plt.xticks([1, 256, 512])

    plots.tight_layout()


plot()
analysis.save_figure(figure_path / 'lso_expanded_plot.png')
plt.show()
