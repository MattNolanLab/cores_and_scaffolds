# %%
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

import sf
import sf.analysis as analysis
import sf.cebra_analysis as cebra_analysis
import sf.plots as plots
import sf.style as style
from sf.analysis import get_fig_dir
from sf.utils import Run



figure_path: Path = get_fig_dir('pc_multiple_speeds')

# %%

original_run: Run = analysis.get_run(analysis.get_exemplar_model())
model_path: Path = sf.EXP / original_run.model_path

pc_inclusions_slow = torch.load(model_path / 'derivatives' / 'pc_inclusionsRnd Slow.pkl')
pc_clamps_slow = torch.load(model_path / 'derivatives' / 'pc_clampsRnd Slow.pkl')
pc_inclusions_fast = torch.load(model_path / 'derivatives' / 'pc_inclusionsRnd Fast.pkl')
pc_clamps_fast = torch.load(model_path / 'derivatives' / 'pc_clampsRnd Fast.pkl')
pc_inclusions = torch.load(model_path / 'derivatives' / 'pc_inclusions.pkl')
pc_clamps = torch.load(model_path / 'derivatives' / 'pc_clamps.pkl')

# %%
reducer = analysis.fit_pca(original_run, n_components=15)
decoder = cebra_analysis.fit_location_decoder(original_run, cebra_model=reducer)


def compute_r2(test_run: Run) -> float:
    embedding = torch.tensor(reducer.transform(test_run.states))
    return cebra_analysis.calculate_location_r2(test_run, embedding, decoder)


def make_df():
    records: List[Dict[str, Any]] = []
    for speed, data in (('slow', pc_inclusions_slow), ('fast', pc_inclusions_fast), ('standard', pc_inclusions)):
        for pc_i, run in data.items():
            for reward_type, reward_val in (('nb', run.nonbeaconed_reward_avg), ('b', run.beaconed_reward_avg)):
                dis = run.get_displacements(reward_type)
                print(dis.peak_locations)
                records.append(
                    {
                        'speed': speed,
                        'pc_i': pc_i,
                        'reward_type': reward_type,
                        'reward': reward_val * 2,
                        'displacement_shift': np.mean(dis.peak_locations),
                        'displacement_sd': np.std(dis.peak_locations),
                        'uncued_first_stop': np.mean(run.get_uncued_first_stop_locations()),
                        'r2': compute_r2(run),
                    }
                )
                print(len(records))
            analysis.plot_stops(run)
            plt.title(f'{pc_i} - {speed}')
            plt.show()

    return pd.DataFrame(records)


df_kept = make_df().query('reward_type == "nb"')


# %%
def make_df():
    records: List[Dict[str, Any]] = []
    for speed, data in (('slow', pc_clamps_slow), ('fast', pc_clamps_fast), ('standard', pc_clamps)):
        for pc_i, run in data.items():
            for reward_type, reward_val in (('nb', run.nonbeaconed_reward_avg), ('b', run.beaconed_reward_avg)):
                dis = run.get_displacements(reward_type)
                print(dis.peak_locations)
                records.append(
                    {
                        'speed': speed,
                        'pc_i': pc_i,
                        'reward_type': reward_type,
                        'reward': reward_val * 2,
                        'displacement_shift': np.mean(dis.peak_locations),
                        'displacement_sd': np.std(dis.peak_locations),
                        'uncued_first_stop': np.mean(run.get_uncued_first_stop_locations()),
                        'r2': compute_r2(run),
                    }
                )
                print(len(records))
            analysis.plot_stops(run)
            plt.title(f'{pc_i} - {speed}')
            plt.show()

    return pd.DataFrame(records)


df_clamped = make_df().query('reward_type == "nb"')


# %%
def plot():
    # TODO refactor: lots of duplication here
    palette: dict[str, str] = style.SPEEDS_TO_COLOUR

    plt.figure(figsize=plots.A5_L * [1, 1])
    gs = plots.GridSpec(ncols=4, nrows=2)

    plots.asp(gs[0, 0])
    sns.lineplot(
        df_kept, x='pc_i', y='reward', hue='speed', palette=palette, legend=False, marker='o', markeredgecolor='none'
    )
    plt.ylabel('% uncued reward')
    plt.xlabel('PCs kept')
    plt.ylim(-5, 105)
    plt.xticks([10, 20, 30])

    plots.asp(gs[0, 1])
    sns.lineplot(
        df_kept, x='pc_i', y='r2', hue='speed', palette=palette, legend=False, marker='o', markeredgecolor='none'
    )
    plt.ylabel('Location R$^2$')
    plt.xlabel('PCs kept')
    plt.xticks([10, 20, 30])

    plots.asp(gs[0, 2])
    sns.lineplot(
        df_kept,
        x='pc_i',
        y='displacement_shift',
        hue='speed',
        palette=palette,
        legend=False,
        marker='o',
        markeredgecolor='none',
    )

    for speed, df_speed in df_kept.groupby('speed'):
        df_speed = df_speed.sort_values('pc_i')

        xs = df_speed['pc_i'].to_numpy(dtype=float)
        ys = df_speed['displacement_shift'].to_numpy(dtype=float)
        yerr = df_speed['displacement_sd'].to_numpy(dtype=float)

        plt.fill_between(xs, ys - yerr, ys + yerr, color=palette[speed], alpha=0.2, linewidth=0)

    plt.ylabel('Uncued stop readout peak')
    analysis.plot_stops_base_vert()
    plt.xticks([10, 20, 30])
    plt.xlabel('PCs kept')

    plots.asp(gs[0, 3])
    sns.lineplot(
        df_kept,
        x='pc_i',
        y='uncued_first_stop',
        hue='speed',
        palette=palette,
        legend=False,
        marker='o',
        markeredgecolor='none',
    )
    plt.ylabel('Avg. first uncued stop')
    analysis.plot_stops_base_vert()
    plt.xticks([10, 20, 30])
    plt.xlabel('PCs kept')

    plots.asp(gs[1, 0])
    sns.lineplot(
        df_clamped, x='pc_i', y='reward', hue='speed', palette=palette, legend=False, marker='o', markeredgecolor='none'
    )
    plt.ylabel('% uncued reward')
    plt.xlabel('PC clamped')
    plt.ylim(-5, 105)
    plt.xticks([9, 19, 29], [10, 20, 30])

    plots.asp(gs[1, 1])
    sns.lineplot(
        df_clamped, x='pc_i', y='r2', hue='speed', palette=palette, legend=False, marker='o', markeredgecolor='none'
    )
    plt.ylabel('Location R$^2$')
    plt.xlabel('PC clamped')
    plt.xticks([9, 19, 29], [10, 20, 30])
    # plt.ylim(-5, 105)

    plots.asp(gs[1, 2])
    sns.lineplot(
        df_clamped,
        x='pc_i',
        y='displacement_shift',
        hue='speed',
        palette=palette,
        legend=False,
        marker='o',
        markeredgecolor='none',
    )
    for speed, df_speed in df_clamped.groupby('speed'):
        df_speed = df_speed.sort_values('pc_i')

        xs = df_speed['pc_i'].to_numpy(dtype=float)
        ys = df_speed['displacement_shift'].to_numpy(dtype=float)
        yerr = df_speed['displacement_sd'].to_numpy(dtype=float)

        plt.fill_between(xs, ys - yerr, ys + yerr, color=palette[speed], alpha=0.2, linewidth=0)
    analysis.plot_stops_base_vert()
    plt.ylabel('Uncued stop readout peak')
    plt.xticks([9, 19, 29], [10, 20, 30])
    plt.xlabel('PC clamped')

    plots.asp(gs[1, 3])
    sns.lineplot(
        df_clamped,
        x='pc_i',
        y='uncued_first_stop',
        hue='speed',
        palette=palette,
        legend=False,
        marker='o',
        markeredgecolor='none',
    )
    plt.ylabel('Avg. first uncued stop')
    analysis.plot_stops_base_vert()
    plt.xticks([9, 19, 29], [10, 20, 30])
    plt.xlabel('PC clamped')

    plt.tight_layout(h_pad=4.5)
    analysis.save_figure(figure_path / 'pc_clamp_other_speeds_plot.pdf')
    plt.show()


plot()
