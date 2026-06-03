from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

import sf
import sf.analysis as analysis
import sf.fixedpoints as fixedpoints
import sf.plots as plots
from sf.analysis import get_fig_dir
from sf.utils import Run

analysis.make_paper_theme()

models: List[Path] = analysis.get_valid_models()


figure_path: Path = get_fig_dir('fixedpoints_neverstop')

# %%
model_path: Path = sf.EXP / 'w103_neverstop__seed__0'
run: Run = analysis.run_model(model_path)[0]

conditions: list[str] = ['middle_0v', 'middle_3v']
fps_by_condition: dict[str, list[fixedpoints.FP]] = fixedpoints.find_fps_at_conditions(
    run, conditions=conditions, base_context=analysis.MIDDLE_CONTEXT
)

qtol: float = np.percentile(fixedpoints.calculate_trial_speeds(run), 5) / 100
pca: PCA = analysis.fit_pca(run, n_components=2)

plt.figure(figsize=plots.A5_L * [2 / 3, 1 / 3])
grid = plots.GridSpec(nrows=1, ncols=2)

for panel_i, condition in enumerate(conditions):
    plt.sca(plt.gcf().add_subplot(grid[0, panel_i]))

    stable_fps: list[fixedpoints.FP] = fixedpoints.filter_stable(fps_by_condition[condition])
    stable_fps = fixedpoints.filter_fps(stable_fps, qtol=qtol)

    analysis.plot_pca_mpl_2d(run, pca=pca, alpha=0.25)
    analysis.plot_fps_2d(run, stable_fps)

    plt.title(f'{condition} (n={len(stable_fps)})')
    plots.despine_topright()

    if panel_i > 0:
        plt.ylabel('')
        plt.yticks([])

plots.tight_layout()
plt.show()


# %%
def build_inputs_pause_at_location(
    run: Run,
    travel_speed: float = 3.0,
    target_location: float = 60.0,
    n_zero_steps: int = 5,
    context: int = analysis.MIDDLE_CONTEXT,
) -> tuple[torch.Tensor, np.ndarray]:
    inputs_raw: list[torch.Tensor] = []
    locations: list[float] = []

    current_location: float = 0.0

    while current_location < target_location:
        step_speed: float = min(travel_speed, target_location - current_location)
        in_blackbox: bool = current_location < 30.0

        inputs_raw.append(
            analysis.prepare_input_for_obs(
                run, velocity=step_speed, indicator=False, blackbox=in_blackbox, beacon=False, context=context
            )
        )

        current_location += step_speed
        locations.append(current_location)

    for _ in range(n_zero_steps):
        inputs_raw.append(
            analysis.prepare_input_for_obs(
                run, velocity=0.0, indicator=False, blackbox=False, beacon=False, context=context
            )
        )
        locations.append(current_location)

    inputs: torch.Tensor = torch.concat(inputs_raw, dim=0)  # [time, input_dim]
    return inputs, np.array(locations, dtype=float)


n_zero_steps: int = 20
inputs, trajectory_locations = build_inputs_pause_at_location(
    run, travel_speed=3.0, target_location=60.0, n_zero_steps=n_zero_steps
)

trajectory_states: torch.Tensor = analysis.run_over_inputs(run, inputs)  # [time, hidden]

stable_fps_0v: list[fixedpoints.FP] = fixedpoints.filter_stable(fps_by_condition['middle_0v'])
stable_fps_0v = fixedpoints.filter_fps(stable_fps_0v, qtol=qtol)

pause_start_i: int = len(trajectory_locations) - n_zero_steps
moving_xy: np.ndarray = pca.transform(trajectory_states[:pause_start_i])  # [time_move, 2]
pause_xy: np.ndarray = pca.transform(trajectory_states[pause_start_i:])  # [time_pause, 2]

plt.figure(figsize=plots.A5_L * [1 / 2, 1 / 2])

analysis.plot_pca_mpl_2d(run, pca=pca, alpha=0.15)
analysis.plot_fps_2d(run, stable_fps_0v)

analysis.plot_trajectory_2d(run, trajectory_states=trajectory_states, pca=pca, color='#A50000', lw=2.5, s=0)

plt.scatter(moving_xy[:, 0], moving_xy[:, 1], color='#A50000', edgecolors='#A50000', s=30, zorder=15)

plt.scatter(pause_xy[:, 0], pause_xy[:, 1], color='orange', edgecolors='black', s=45, zorder=20)

speed3_handle = plt.scatter([], [], s=30, color='#A50000', edgecolors='#A50000', label='Speed: 3')
speed0_handle = plt.scatter([], [], s=45, color='orange', edgecolors='black', label='Speed: 0')
fp_handle = plt.scatter([], [], s=300, marker='*', edgecolors='black', c='blue', label='Stable fixed point')

plt.legend(handles=[speed3_handle, speed0_handle, fp_handle], frameon=False, loc='best')

plt.title(f'To ~60, then 0 speed for {n_zero_steps} timesteps')
plots.despine_topright()
plots.tight_layout()
plt.show()


# %%
plots.set_font_size(9)

n_zero_steps: int = 20
inputs, trajectory_locations = build_inputs_pause_at_location(
    run, travel_speed=3.0, target_location=60.0, n_zero_steps=n_zero_steps
)

trajectory_states: torch.Tensor = analysis.run_over_inputs(run, inputs)  # [time, hidden]

stable_fps_0v: list[fixedpoints.FP] = fixedpoints.filter_stable(fps_by_condition['middle_0v'])
stable_fps_0v = fixedpoints.filter_fps(stable_fps_0v, qtol=qtol)

pause_start_i: int = len(trajectory_locations) - n_zero_steps
moving_xy: np.ndarray = pca.transform(trajectory_states[:pause_start_i])  # [time_move, 2]
pause_xy: np.ndarray = pca.transform(trajectory_states[pause_start_i:])  # [time_pause, 2]

plt.figure(figsize=plots.A5_L * [1 / 2, 1 / 3])

analysis.plot_pca_mpl_2d(run, pca=pca, alpha=0.15)
analysis.plot_fps_2d(run, stable_fps_0v)

analysis.plot_trajectory_2d(run, trajectory_states=trajectory_states, pca=pca, color='#A50000', lw=2.0, s=0)

plt.scatter(moving_xy[:, 0], moving_xy[:, 1], color='#A50000', edgecolors='#A50000', s=22, zorder=15)

plt.scatter(pause_xy[:, 0], pause_xy[:, 1], color='orange', edgecolors='black', s=28, zorder=20)

speed3_handle = plt.scatter([], [], s=22, color='#A50000', edgecolors='#A50000', label='Speed: 3')
speed0_handle = plt.scatter([], [], s=28, color='orange', edgecolors='black', label='Speed: 0')
fp_handle = plt.scatter([], [], s=220, marker='*', edgecolors='black', c='blue', label='Stable fixed point')

plt.legend(
    handles=[speed3_handle, speed0_handle, fp_handle],
    frameon=False,
    loc='center left',
    bbox_to_anchor=(1.05, 0.5),
    fontsize=9,
)

plt.xlabel('PC1')
plt.xticks([])
plt.ylabel('PC2')
plt.yticks([])
plots.despine_topright()
plots.tight_layout(right=0.93)
plt.show()
