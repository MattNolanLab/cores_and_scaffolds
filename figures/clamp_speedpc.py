# |include: false
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from sf import analysis, plots
from sf.analysis import get_fig_dir
from sf.specific import clamp
from sf.utils import Run

run = analysis.get_run(analysis.get_exemplar_model())

pca: PCA = analysis.fit_pca(run, n_components=15)

fig_path: Path = get_fig_dir('clamp_speedpc')


# %%
def by_speed(speed: str):
    return f'LinearNavigationContextual12Static3Speed{speed}-v2'


envs_by_speed: Dict[str, str] = {
    'Standard': 'LinearNavigationContextual12Static3-v2',
    'Constant 1.5': by_speed('constant_1p5'),
    'Constant 4.0': by_speed('constant_4p0'),
    'Standard * 0.5': by_speed('slow'),
    'Standard * 2.0': by_speed('fast'),
}

control_results: Dict[str, Run] = {}
for env_readable, env in envs_by_speed.items():
    control_results[env_readable] = analysis.run_model(run.model_path, env_name=env)[0]

perturbation_results: Dict[str, Run] = {}
for env_readable, env in tqdm(envs_by_speed.items()):
    clamped_rnn = clamp.ClampedRNN(pc_i=2, pc_value=0.0, pca=pca, core=run.rnn)

    perturbation_results[env_readable] = analysis.run_model(run.model_path, env_name=env, core_override=clamped_rnn)[0]

# %%
fig = plt.figure(figsize=plots.A4)

gs = plt.GridSpec(len(envs_by_speed), 2)

plots.set_font_size(12)

for row_i, env_readable in enumerate(envs_by_speed.keys()):
    control: Run = control_results[env_readable]
    perturbation: Run = perturbation_results[env_readable]

    plt.sca(fig.add_subplot(gs[row_i, 0]))
    analysis.plot_stops(control)
    plt.ylabel('Trial')

    title = 'Control\n' if row_i == 0 else ''
    title += f'Reward: Uncued: {control.nonbeaconed_reward_avg * 2:.1f}% - Cued: {control.beaconed_reward_avg * 2:.1f}%'
    plt.title(title)

    plt.sca(fig.add_subplot(gs[row_i, 1]))
    analysis.plot_stops(perturbation)

    title = 'PC3 clamped to 0\n' if row_i == 0 else ''
    title += f'Reward: Uncued: {perturbation.nonbeaconed_reward_avg * 2:.1f}% - Cued: {perturbation.beaconed_reward_avg * 2:.1f}%'
    plt.title(title)

    plt.gca().annotate(env_readable, xy=(1.05, 0.5), xycoords='axes fraction', va='center', ha='left', rotation=-90)

    if row_i == len(envs_by_speed.keys()) // 2:
        plt.gca().annotate(
            'Speed distribution',
            xy=(1.15, 0.5),
            xycoords='axes fraction',
            va='center',
            ha='left',
            rotation=-90,
            fontsize=12,
        )

fig.supxlabel('Location', x=0.48)

plt.tight_layout()
analysis.save_figure(fig_path / 'plot.png', allow_non_pdf=True)
plt.show()
