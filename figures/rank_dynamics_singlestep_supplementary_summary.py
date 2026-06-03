from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm.autonotebook import tqdm

import sf.analysis as analysis
import sf.plots as plots
from sf.analysis import get_fig_dir
from sf.notebook_setup import setup_notebook
from sf.specific.singlestep import DrivenTrajectory, simulate_driven_trajectory_simple
from sf.subspace import calculate_mapping_matrices
from sf.utils import Run

setup_notebook()
analysis.make_paper_theme()

figure_path: Path = get_fig_dir('rank_dynamics_supplemental')

plots.set_font_size(9)

original_run = analysis.run_model(analysis.get_exemplar_model())[0]
# %%
before_final_speeds: List[float] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
palette = dict(zip(before_final_speeds, sns.color_palette('vlag', n_colors=len(before_final_speeds))))
palette[2.5] = (0, 0, 0)


# %%
def make_simulations(
    run: Run,
    base_v: float = 2.5,
    ranks: Tuple[int, ...] = (512,),
    outbound_locations: Tuple[int, ...] = np.arange(35 - 30, 170 - 30, 15),
) -> pd.DataFrame:
    # assert base v not in final speeds twice
    assert before_final_speeds[4] == base_v
    assert len(before_final_speeds) == 9

    mappings: Dict[str, torch.Tensor] = calculate_mapping_matrices(run)

    records: List[pd.DataFrame] = []
    experiment_id: int = 0
    for outbound_distance in tqdm(outbound_locations):
        for before_final_v in tqdm(before_final_speeds):
            for rank in ranks:
                trajectory: DrivenTrajectory = simulate_driven_trajectory_simple(
                    original_run,
                    base_v=base_v,
                    outbound_distance=outbound_distance,
                    before_final_v=before_final_v,
                    rank=rank,
                )
                latents: torch.Tensor = mappings['hxs_to_latent'] @ trajectory.states.T
                record: pd.DataFrame = pd.DataFrame(
                    [
                        {
                            'before_final_v': before_final_v,
                            'outbound_distance': outbound_distance,
                            'rank': rank,
                            'base_v': base_v,
                            'experiment_id': f'{experiment_id}_{before_final_v}_{base_v}_{outbound_distance}_{rank}',
                            # Arrays
                            't': trajectory.ts,
                            'location': trajectory.locations,
                            'latent_0': latents[0].numpy(),
                            'latent_1': latents[1].numpy(),
                        }
                    ]
                )

                record = record.explode(['t', 'location', 'latent_0', 'latent_1'])

                t_final: int = record.t.max()
                record = record.query(f't >= {t_final - 2}')  # Just get t-1 -> t transition
                records.append(record)

    return pd.concat(records, ignore_index=True)


simulation_df = make_simulations(original_run, 2.5, [2, 512])


# %%
def plot_transitions_rank(df: pd.DataFrame, mode_i: int):
    df = df.query('rank in [2, 512]')
    print(df)

    plt.figure(figsize=plots.A4 * [1 / 2, 1])
    gs = plots.GridSpec(nrows=len(df.outbound_distance.unique()), ncols=len(df['rank'].unique()))

    row_axes: dict[int, plt.Axes | None] = {}
    for row_i, (_, ddf) in enumerate(df.sort_values('outbound_distance').groupby('outbound_distance')):
        for col_i, (_, dddf) in enumerate(ddf.sort_values('rank').groupby('rank')):
            plots.asp(gs[row_i, col_i], sharey=row_axes.get(row_i))
            row_axes[row_i] = plt.gca()
            t_final: int = dddf.t.max()
            dddf_final = dddf.query(f't >= {t_final - 1}')

            assert all([len(dddf[f].unique()) == 1 for f in ['outbound_distance', 'rank', 'base_v']])

            sns.lineplot(
                dddf,
                x='location',
                y=f'latent_{mode_i}',
                hue='before_final_v',
                lw=3,
                legend=False,
                marker='o',
                markersize=4.5,
                markeredgecolor='none',
                alpha=0.2,
                palette=palette,
            )

            sns.lineplot(
                dddf_final,
                x='location',
                y=f'latent_{mode_i}',
                hue='before_final_v',
                lw=3,
                legend=False,
                marker='o',
                markersize=4.5,
                markeredgecolor='none',
                alpha=1.0,
                palette=palette,
            )

            if not gs.is_last_row(row_i):
                plt.xlabel('')
            if gs.is_first_row(row_i):
                plt.title(f'Rank {dddf["rank"].max()}')

            plt.ylabel('')
            if gs.is_last_row(row_i):
                plt.xlabel('Location')
            if gs.is_last_col(col_i):
                plt.gca().tick_params(axis='y', which='both', left=False, labelleft=False)

            #     plt.gca().annotate(f'Rank {dddf["rank"].max()}', xy=(1.05, 0.5), xycoords='axes fraction', va='center', ha='left', rotation=-90)

    plt.suptitle(f'Mode {mode_i + 1}', y=0.99, x=0.54)
    plt.tight_layout()
    plots.supylabel('Activity along mode u-vector', x=0.0, y=0.5)


for i, rank in enumerate(range(2)):
    plot_transitions_rank(simulation_df, rank)
    analysis.save_figure(figure_path / f'summary_transitions_mode{i}.pdf')
    plt.show()
