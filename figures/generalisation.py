# %%
from pathlib import Path
from typing import Dict, Optional

import gym
import matplotlib.pyplot as plt
import seaborn as sns

from sf import analysis, plots, style
from sf.analysis import get_fig_dir
from sf.utils import Run

analysis.make_paper_theme()

model_path: Path = analysis.get_exemplar_model()

figure_path: Path = get_fig_dir('generalisation')


# %%
def by_speed(speed: str):
    return f'LinearNavigationContextual12Static3Speed{speed}-v2'


# TODO factor out non-standard speeds into run_model function
runs_by_condition: Dict[str, Run] = {
    'Standard': analysis.run_model(model_path)[0],
    'Constant 1.5': analysis.run_model(model_path, env_name=by_speed('constant_1p5'))[0],
    'Constant 4.0': analysis.run_model(model_path, env_name=by_speed('constant_4p0'))[0],
    'Standard * 0.5': analysis.run_model(model_path, env_name=by_speed('slow'))[0],
    'Standard * 2.0': analysis.run_model(model_path, env_name=by_speed('fast'))[0],
    'Standard or 0 (50/50)': analysis.run_model(model_path, env_name=by_speed('halfzero'))[0],
}
# %%
plots.set_font_size(9)

fig = plt.figure(figsize=plots.A5_L * [1, 2 / 3])
gs = plt.GridSpec(nrows=4, ncols=len(runs_by_condition), height_ratios=[1, 0.7, 1, 1 / 3])
gs.update(hspace=0.15)

hist_ax: Optional[plt.Axes] = None
for i, (condition, run) in enumerate(runs_by_condition.items()):
    plt.sca(fig.add_subplot(gs[0, i], sharex=hist_ax))

    title = []

    plt.title(
        '{head}\n{condition}'.format(
            condition=condition,
            head={'Standard': 'Train distribution', 'Standard * 0.5': 'Test distributions'}.get(condition, ''),
        )
    )

    sns.histplot(x=run.velocities, stat='probability', element='step', bins=30)
    if i != 0:
        plt.yticks([])
    plt.ylim(0, 1)
    plt.ylabel('Probability' if i == 0 else '')
    plt.xlabel('Speed')
    if i == gs.ncols - 1:
        plt.gca().annotate(
            'Speed inputs', xy=(1.05, 0.5), xycoords='axes fraction', va='center', ha='left', rotation=-90
        )

    hist_ax = plt.gca()

    plt.sca(fig.add_subplot(gs[2, i]))
    plt.title(f'Reward: {run.reward_avg:.1f}%')
    analysis.plot_stops(run)
    plt.ylabel('Trial' if i == 0 else '')
    if i == gs.ncols - 1:
        plt.gca().annotate('Behaviour', xy=(1.05, 0.35), xycoords='axes fraction', va='center', ha='left', rotation=-90)

    plt.sca(fig.add_subplot(gs[3, i]))
    analysis.plot_stop_distributions(run, below_stops=False)
    analysis.strip_plot_axes()
    plt.ylabel('Stops\n(normalised)' if i == 0 else '')
    plt.xlabel('Location')

plt.tight_layout()
analysis.save_figure(figure_path / 'generalisation_plot.png')
plt.show()


# %%
plots.set_font_size(9)

fig = plt.figure(figsize=plots.A5_L * [1, 1 / 3])
gs = plt.GridSpec(nrows=1, ncols=len(runs_by_condition))

hist_ax: Optional[plt.Axes] = None
for i, (condition, run) in enumerate(runs_by_condition.items()):
    plt.sca(fig.add_subplot(gs[0, i], sharex=hist_ax))

    title = []

    plt.title(
        '{head}\n{condition}'.format(
            condition=condition,
            head={'Standard': 'Train distribution', 'Standard * 0.5': 'Test distributions'}.get(condition, ''),
        )
    )

    env = gym.make(run.env_name)
    sns.histplot(x=[env.get_random_velocity() for _ in range(50000)], stat='probability', element='step', bins=30)
    plt.ylabel('Probability' if i == 0 else '')
    plt.xlabel('Speed')

    hist_ax = plt.gca()

plt.tight_layout()
analysis.save_figure(figure_path / 'generalisation_distributions_plot.png')
plt.show()


# %%
plots.set_font_size(9)

fig = plt.figure(figsize=plots.A5_L * [1 / 3, 1 / 3])

speed_to_colours: Dict[str, str] = {
    'Standard': 'black',
    'Standard * 2.0': style.SPEEDS_TO_COLOUR['fast'],
    'Standard * 0.5': style.SPEEDS_TO_COLOUR['slow'],
}


for i, condition in enumerate(['Standard', 'Standard * 2.0', 'Standard * 0.5']):
    run = runs_by_condition[condition]

    env = gym.make(run.env_name)
    sns.histplot(
        x=[env.get_random_velocity() for _ in range(50000)],
        stat='probability',
        element='step',
        bins=30,
        color=speed_to_colours[condition],
        alpha=0.4,
    )
    plt.ylabel('Probability')
    plt.xlabel('Speed')

    hist_ax = plt.gca()

plt.yticks([0.0, 0.08], [0.0, 0.08])
plt.xticks([0, 10], [0, 10])
plt.tight_layout()
plots.despine_topright()
analysis.save_figure(figure_path / 'distributions_overlaid.pdf')
plt.show()


# %%
def plot():
    plt.figure(figsize=plots.A5_L * [1 / 4, 1 / 4])
    gs = plots.GridSpec(nrows=2, ncols=1)

    for i, (label, s, env) in enumerate([['* 2.0', 2.0, 'Standard * 2.0'], ['* 0.5', 0.5, 'Standard * 0.5']]):
        plots.asp(gs[i])
        analysis.plot_stop_distributions(runs_by_condition[env], below_stops=False, lw=2)
        plt.yticks([])

        plt.gca().axvspan(
            30 + (analysis.DEFAULT_RZ_MIN - 30) / s, 30 + (analysis.DEFAULT_RZ_MAX - 30) / s, alpha=0.3, color='red'
        )

        plt.gca().annotate(label, xy=(1.05, 0.5), xycoords='axes fraction', va='center', ha='left', rotation=0)

    plots.supylabel('Stops', x=0.05)

    plt.tight_layout()
    analysis.save_figure(figure_path / 'generalisation_behaviour.pdf')

    plt.show()


plot()
