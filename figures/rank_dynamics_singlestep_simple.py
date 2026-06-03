from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.linear_model import LinearRegression
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

figure_path: Path = get_fig_dir('rank_dynamics')

plots.set_font_size(9)

original_run = analysis.run_model(analysis.get_exemplar_model())[0]
# %%
before_final_speeds: list[float] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
palette = dict(zip(before_final_speeds, sns.color_palette('vlag', n_colors=len(before_final_speeds))))
palette[2.5] = (0, 0, 0)

cmap = mpl.colors.LinearSegmentedColormap.from_list('before_final_v', [palette[v] for v in before_final_speeds], N=256)
norm = mpl.colors.Normalize(vmin=min(before_final_speeds), vmax=max(before_final_speeds))
mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array([])

plt.figure(figsize=plots.A5_L * [1 / 8, 1 / 5])
plt.axis('off')
cbar = plt.colorbar(mappable, orientation='vertical', fraction=0.95, pad=0.02, aspect=8)
cbar.set_ticks([0.5, 2.5, 4.5])
cbar.set_ticklabels([str(v) for v in [0.5, 2.5, 4.5]])

plt.tight_layout()
analysis.save_figure(figure_path / 'before_final_v_colorbar_vertical.pdf')
plt.show()


# %%
def make_simulations(
    run: Run,
    base_v: float = 2.5,
    ranks: Tuple[int, ...] = (512,),
    outbound_locations: Tuple[int, ...] = (50 - 30, 70 - 30),
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


simulation_df = make_simulations(original_run, 2.5, (512,))

# %%
roman_labels: list[str] = ['i', 'ii']


def plot_track():
    plt.figure(figsize=plots.A5_L * [1.10 / 5, 1 / 4])
    analysis.plot_stops_base()
    for i, start in enumerate(sorted(simulation_df['outbound_distance'].unique() + 30)):
        label: str = roman_labels[i]
        plt.text(
            start + 0.5,
            1.02,
            f'{label}',
            transform=plt.gca().get_xaxis_transform(),
            ha='center',
            va='bottom',
            fontsize=9,
        )
        plt.axvline(start, color='black', alpha=0.95)
    plt.yticks([])
    plt.ylabel('')
    plt.xticks([0, 200], [0, 200])
    plots.tight_layout()


plot_track()
analysis.save_figure(figure_path / 'locations.pdf')
plt.show()

# %%
base_v: float = 2.5


def extract_control_transition(df: pd.DataFrame, loc_a, control_v: float) -> pd.DataFrame:
    result = df.query(f'(location == {loc_a}) or (location == {loc_a + control_v}')
    print(result)
    return result


def predict_latent_at_location(df, latent_i, location):
    X = df[['location']].to_numpy()
    y = df[f'latent_{latent_i}'].to_numpy()
    model = LinearRegression().fit(X, y)
    return float(model.predict([[location]])[0])


def plot_control_line(
    df,
    outbound_distance: float,
    pred_location: float,
    base_v: float = base_v,
    before_final_v: float = base_v,
    getter=slice(None),  # TODO: refactor!
    color='black',
    rank: int = 512,
) -> None:

    control_transition: pd.DataFrame = (
        df.query(f'outbound_distance == {outbound_distance}')
        .query(f'base_v == {base_v}')
        .query(f'before_final_v == {before_final_v}')
        .query(f'rank == {rank}')
    )
    control_transition = control_transition.sort_values('location').iloc[0:2]
    print(control_transition)

    X = control_transition['location'].to_numpy()
    y = control_transition['latent_1'].to_numpy()

    sns.lineplot(
        x=[X[0], X[1], pred_location][getter],
        y=[y[0], y[1], predict_latent_at_location(control_transition, 1, pred_location)][getter],
        color=color,
        linestyle='--',
    )


def plot_transitions_pre(df: pd.DataFrame = simulation_df):
    plt.figure(figsize=plots.A5_L * [3 / 6, 1 / 3])
    gs = plots.GridSpec(ncols=len(df['outbound_distance'].unique()), nrows=1)

    for col_i, (_, ddf) in enumerate(df.sort_values('outbound_distance').groupby('outbound_distance')):
        t_final: int = ddf.t.max()
        ddf = ddf.query(f't < {t_final}')
        for row_i, latent_i in enumerate([1]):
            plots.asp(gs[row_i, col_i])

            plot_control_line(ddf, outbound_distance=ddf.outbound_distance.max(), pred_location=ddf.location.max())

            sns.lineplot(
                ddf,
                x='location',
                y=f'latent_{latent_i}',
                hue='before_final_v',
                hue_order=sorted(ddf['before_final_v'].unique(), reverse=True),
                lw=3,
                legend=False,
                marker='o',
                markersize=4.5,
                markeredgecolor='none',
                palette=palette,
            )

            if not gs.is_last_row(row_i):
                plt.xlabel('')
                plt.xticks([])
            if gs.is_first_row(row_i):
                plt.title(f'({roman_labels[col_i]})')
            plt.ylabel('')
            if gs.is_last_row(row_i):
                plt.xlabel('Location')
            plots.set_xaxis_n_ticks(2)

    plots.supylabel('Mode 2 projection', ha='center', x=0.05)
    plt.tight_layout()


plot_transitions_pre()
analysis.save_figure(figure_path / 'onestep_pres.pdf')
plt.show()

# %%


def plot_transitions(df: pd.DataFrame = simulation_df):
    plt.figure(figsize=plots.A5_L * [3 / 6, 1 / 3])
    gs = plots.GridSpec(ncols=len(df['outbound_distance'].unique()), nrows=1)

    for col_i, (_, ddf) in enumerate(df.sort_values('outbound_distance').groupby('outbound_distance')):
        for row_i, latent_i in enumerate([1]):
            plots.asp(gs[row_i, col_i])

            # sns.lineplot(ddf,
            #              x='location', y=f'latent_{latent_i}', hue='before_final_v',
            #              legend=False, marker='o', markersize=2.5, markeredgecolor='none')

            sns.lineplot(
                ddf,
                x='location',
                y=f'latent_{latent_i}',
                hue='before_final_v',
                legend=False,
                marker='o',
                markersize=2.5,
                markeredgecolor='none',
                alpha=0.2,
                palette=palette,
            )

            t_final: int = ddf.t.max()
            ddf_final = ddf.query(f't >= {t_final - 1}')
            print(ddf_final)
            sns.lineplot(
                ddf_final,
                x='location',
                y=f'latent_{latent_i}',
                hue='before_final_v',
                hue_order=sorted(ddf['before_final_v'].unique(), reverse=True),
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
                plt.xticks([])
            if gs.is_first_row(row_i):
                plt.title(f'({col_i + 1})')

            plt.ylabel('')
            if gs.is_last_row(row_i):
                plt.xlabel('Location')

            plots.set_xaxis_n_ticks(2)

    plt.tight_layout()
    plots.supylabel('Activity along mode 2 u-vector', x=0.0, y=0.45)


plot_transitions()
analysis.save_figure(figure_path / 'onestep_fullrank_compensation.pdf')
plt.show()

# %%
simulation_df_rank = make_simulations(original_run, base_v, tuple([2, 512]))


# %%
def plot_transitions_rank(df: pd.DataFrame, outbound_distance: int):
    df = df.query(f'outbound_distance == {outbound_distance}')
    df = df.query('rank in [2, 512]')
    print(df)

    plt.figure(figsize=plots.A5_L * [3 / 6, 1 / 3])
    gs = plots.GridSpec(ncols=2, nrows=1)

    row_axes = [None, None]
    for col_i, (rank, ddf) in enumerate(df.sort_values('rank', ascending=False).groupby('rank', sort=False)):
        t_final: int = ddf.t.max()
        ddf_final = ddf.query(f't >= {t_final - 1}')

        for row_i, latent_i in enumerate([1]):
            plots.asp(gs[row_i, col_i], sharey=row_axes[row_i])
            row_axes[row_i] = plt.gca()

            assert all([len(ddf[f].unique()) == 1 for f in ['outbound_distance', 'rank', 'base_v']])

            plot_control_line(
                df,
                outbound_distance=ddf.outbound_distance.max(),
                pred_location=ddf.location.max(),
                getter=slice(1, None),
            )

            sns.lineplot(
                ddf,
                x='location',
                y=f'latent_{latent_i}',
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
                ddf_final,
                x='location',
                y=f'latent_{latent_i}',
                hue='before_final_v',
                lw=3 if rank == 512 else 1.5,
                legend=False,
                marker='o',
                markersize=4.5,
                markeredgecolor='none',
                alpha=1.0,
                palette=palette,
            )

            min_v: float = ddf.location.min()
            plt.xticks([min_v, min_v + base_v, min_v + base_v + base_v])

            if not gs.is_last_row(row_i):
                plt.xlabel('')
                plt.xticks([])

            rank_titles: dict[int, str] = {512: 'Full rank', 2: 'Rank 2'}
            if gs.is_first_row(row_i):
                plt.title(f'{rank_titles[ddf["rank"].unique()[0]]}')
            if gs.is_last_col(col_i):
                plt.gca().tick_params(axis='y', which='both', labelleft=False)

            plt.ylabel('')
            if gs.is_last_row(row_i):
                plt.xlabel('Location')

    plots.supylabel('Mode 2 projection', ha='center', x=0.05)
    plt.tight_layout()


for i, od in enumerate(simulation_df_rank['outbound_distance'].unique()):
    plot_transitions_rank(simulation_df_rank, od)
    #    plt.suptitle(f'{roman_labels[i]}')
    analysis.save_figure(figure_path / f'plot_transitions_rank_od_index{i}.pdf')
    plt.show()


# %%
simulation_df_rank_systematic = make_simulations(
    original_run, base_v, tuple([2, 16, 64, 128, 224, 256, 512]), outbound_locations=tuple(np.arange(40, 160) - 30)
)


# %%
def predict_controL_activity(
    df,
    outbound_distance: float,
    pred_location: float,
    latent_i: int,
    base_v: float = base_v,
    before_final_v: float = base_v,
    rank: int = 512,
) -> float:

    control_transition: pd.DataFrame = (
        df.query(f'outbound_distance == {outbound_distance}')
        .query(f'base_v == {base_v}')
        .query(f'before_final_v == {before_final_v}')
        .query(f'rank == {rank}')
    )
    control_transition = control_transition.sort_values('location').iloc[0:2]
    print(control_transition)

    return predict_latent_at_location(control_transition, latent_i, pred_location)


def compute_systematic(data_df: pd.DataFrame) -> pd.DataFrame:
    results: list[dict[str, float]] = []

    for outbound_distance in data_df.outbound_distance.unique():
        df = data_df.query(f'outbound_distance == {outbound_distance}')
        # df = df.query('rank in [2, 512]')
        print(df)

        for col_i, (rank, ddf) in enumerate(df.sort_values('rank', ascending=False).groupby('rank', sort=False)):
            t_final: int = ddf.t.max()
            ddf_final = ddf.query(f't >= {t_final - 1}')

            for row_i, latent_i in enumerate([0, 1]):  # Dummy for now - just interested in latent 1
                assert all([len(ddf[f].unique()) == 1 for f in ['outbound_distance', 'rank', 'base_v']])

                for before_final_v, dddf in ddf_final.groupby('before_final_v'):
                    assert (
                        ddf.outbound_distance.max() == dddf.outbound_distance.max()
                    )  # No idea how this could be violated

                    control_act = predict_controL_activity(
                        df,
                        outbound_distance=ddf.outbound_distance.max(),
                        pred_location=dddf.location.max(),
                        latent_i=latent_i,
                    )
                    # # Debugging/verification
                    # plt.figure()
                    # plot_control_line(df, outbound_distance=ddf.outbound_distance.max(), pred_location=dddf.location.max())
                    # sns.lineplot(dddf, x='location', y='latent_1')
                    #
                    # print(control_act, dddf, ddf.outbound_distance.min())
                    # plt.plot([dddf.location.max(), dddf.location.max()],
                    #          [
                    #              dddf.sort_values('location', ascending=True).iloc[1].latent_1,
                    #              control_act])
                    # plt.show()

                    results.append(
                        {
                            'latent_i': latent_i,
                            'rank': rank,
                            'control_act': control_act,
                            'actual_act': dddf.sort_values('location', ascending=True).iloc[1][f'latent_{latent_i}'],
                            'tsub2location': ddf.outbound_distance.min() + 30,  # Location of t-2.
                            'before_final_v': before_final_v,
                        }
                    )
    return pd.DataFrame(results)


systematic_df = compute_systematic(simulation_df_rank_systematic)
systematic_df['raw_error'] = systematic_df['actual_act'] - systematic_df['control_act']
systematic_df['raw_error_perc'] = (systematic_df['actual_act'] - systematic_df['control_act']) / systematic_df[
    'control_act'
]

print(systematic_df)


# %%
def plot():
    plt.figure(figsize=plots.A5_L * [2 / 3, 2 / 3])
    gs = plots.GridSpec(ncols=2, nrows=2)

    for row_i, latent_i in enumerate([0, 1]):
        ax: plt.Axes | None = None
        for col_i, rank in enumerate([512, 2]):
            plots.asp(gs[row_i, col_i], sharey=ax)
            ax = plt.gca()
            sns.lineplot(
                systematic_df.query(f'rank == {rank}').query('tsub2location < 90').query(f'latent_i == {latent_i}'),
                x='tsub2location',
                y='raw_error',
                hue='before_final_v',
                palette=palette,
                legend=False,
            )
            analysis.plot_stops_base()
            plt.xticks([0, 200])
            plt.ylabel('')

            if row_i == 0:
                plt.xlabel('')
                plt.xticks([])
                plt.title({2: 'Rank 2', 512: 'Full rank'}[rank])

            if col_i == 1:
                plt.gca().annotate(
                    f'Mode {latent_i + 1}',
                    xy=(1.05, 0.5),
                    xycoords='axes fraction',
                    va='center',
                    ha='left',
                    rotation=-90,
                )

    plots.supylabel('Single step error')

    plt.tight_layout()
    plt.show()


plot()


# %%
def plot():
    plt.figure(figsize=plots.A5_L * [2 / 3, 1 / 3])
    gs = plots.GridSpec(ncols=2, nrows=1)

    for row_i, latent_i in enumerate([1]):
        ax: plt.Axes | None = None
        for col_i, rank in enumerate([512, 2]):
            plots.asp(gs[row_i, col_i], sharey=ax)
            ax = plt.gca()
            sns.lineplot(
                systematic_df.query(f'rank == {rank}').query('tsub2location < 90').query(f'latent_i == {latent_i}'),
                x='tsub2location',
                y='raw_error',
                hue='before_final_v',
                palette=palette,
                legend=False,
            )
            analysis.plot_stops_base()
            plt.xticks([20, 105], [30, 105])
            plt.xlim(20, 105)
            plt.ylabel('')

            if col_i == 1:
                plots.hide_y()

            if row_i == 0:
                plt.title({2: 'Rank 2', 512: 'Full rank'}[rank])

    plots.supylabel('Error after correction')

    plt.tight_layout()
    analysis.save_figure(figure_path / 'error_after_correction.pdf')
    plt.show()


plot()


# %%
def plot():
    plt.figure(figsize=plots.A4)

    ranks: list[int] = systematic_df['rank'].unique()
    gs = plots.GridSpec(ncols=2, nrows=len(ranks))
    print(systematic_df['rank'].unique())

    for col_i, latent_i in enumerate([0, 1]):
        ax: plt.Axes | None = None
        for row_i, rank in enumerate(ranks):
            plots.asp(gs[row_i, col_i], sharey=ax)
            ax = plt.gca()
            sns.lineplot(
                systematic_df.query(f'rank == {rank}').query('tsub2location < 90').query(f'latent_i == {latent_i}'),
                x='tsub2location',
                y='raw_error',
                hue='before_final_v',
                palette=palette,
                legend=False,
            )
            analysis.plot_stops_base()
            plt.axhline(y=0, alpha=0.3)
            plt.xticks([20, 105], [30, 105])
            plt.xlim(20, 105)
            plt.ylabel('')

            #            if col_i == 1:
            #                plots.hide_y()
            plt.xlabel('')
            if row_i == 0:
                plt.title(f'Mode {latent_i + 1}')
            elif row_i == (len(ranks) - 1):
                plt.xlabel('Location')

            if col_i == 1:
                plt.gca().annotate(
                    f'Rank {rank}', xy=(1.05, 0.5), xycoords='axes fraction', va='center', ha='left', rotation=-90
                )

    plots.supylabel('Error after correction')

    analysis.save_figure(figure_path / 'error_after_correction_multiple.pdf')
    plt.tight_layout()

    plt.show()


plot()


# %%


def compute_systematic_precorrection(data_df: pd.DataFrame) -> pd.DataFrame:
    results: list[dict[str, float]] = []

    for outbound_distance in data_df.outbound_distance.unique():
        df = data_df.query(f'outbound_distance == {outbound_distance}')
        df = df.query('rank in [512]')
        print(df)

        for col_i, (rank, ddf) in enumerate(df.sort_values('rank', ascending=False).groupby('rank', sort=False)):
            t_final: int = ddf.t.max()
            ddf_final = ddf.query(f't >= {t_final - 2}').query(f't < {t_final}')

            for row_i, latent_i in enumerate([0, 1]):  # Dummy for now - just interested in latent 1
                assert all([len(ddf[f].unique()) == 1 for f in ['outbound_distance', 'rank', 'base_v']])

                for before_final_v, dddf in ddf_final.groupby('before_final_v'):
                    assert (
                        ddf.outbound_distance.max() == dddf.outbound_distance.max()
                    )  # No idea how this could be violated

                    control_act = predict_controL_activity(
                        df,
                        outbound_distance=ddf.outbound_distance.max(),
                        pred_location=dddf.location.max(),
                        latent_i=latent_i,
                    )

                    results.append(
                        {
                            'latent_i': latent_i,
                            'rank': rank,
                            'control_act': control_act,
                            'actual_act': dddf.sort_values('location', ascending=True).iloc[1][f'latent_{latent_i}'],
                            'tsub2location': ddf.outbound_distance.min() + 30,  # Location of t-2.
                            'before_final_v': before_final_v,
                        }
                    )
    df = pd.DataFrame(results)
    df['raw_error'] = df['actual_act'] - df['control_act']
    df['raw_error_perc'] = (df['actual_act'] - df['control_act']) / df['control_act']

    return df


systematic_df_precorrection = compute_systematic_precorrection(simulation_df_rank_systematic)


# %%
def plot():
    plt.figure(figsize=plots.A5_L * [1 / 3, 1 / 3])

    latent_i = 1
    rank = 512
    sns.lineplot(
        systematic_df_precorrection.query(f'rank == {rank}')
        .query('tsub2location < 90')
        .query(f'latent_i == {latent_i}')
        .query('before_final_v != 2.5'),
        x='tsub2location',
        y='raw_error',
        hue='before_final_v',
        palette=palette,
        legend=False,
    )
    analysis.plot_stops_base()
    plt.xticks([20, 105], [30, 105])
    plt.xlim(20, 105)
    plt.ylabel('')

    plt.ylabel('Single-step error')

    plt.title(' ')  # Dummy title for spacing
    plt.tight_layout()
    analysis.save_figure(figure_path / 'error_before_correction.pdf')
    plt.show()


plot()
