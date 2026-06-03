# %%
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch

from sf import analysis, cebra_analysis, model_analysis, plots
from sf.analysis import get_fig_dir
from sf.cebra_analysis import fit_location_decoder
from sf.utils import Run, run_model

figure_path: Path = get_fig_dir('cebra_example')

analysis.make_paper_theme()


run: Run = torch.load(analysis.get_exemplar_model() / 'derivatives/run.pkl')

milestones: List[int] = model_analysis.training_curve_with_episodes(
    run, analysis.get_curriculum_exemplar_params()['episodes']
)

runs_by_stage: Dict[str, Run] = {
    k: run_model(run.model_path, milestone_index_override=m)[0]
    for k, m in zip(['Untrained', 'Partially trained', 'Fully trained'], milestones, strict=True)
}

runs_by_stage_test: Dict[str, Run] = {
    k: run_model(run.model_path, milestone_index_override=m)[0]
    for k, m in zip(['Untrained', 'Partially trained', 'Fully trained'], milestones, strict=True)
}

cebra_models: Dict[str, Any] = {
    k: analysis.fit_cebra(r, max_iterations=5_000, output_dimensions=2) for k, r in runs_by_stage.items()
}

r2_scores: Dict[str, float] = {}
for stage, train_run in runs_by_stage.items():
    decoder = fit_location_decoder(train_run, cebra_model=cebra_models[stage])

    test_run: Run = runs_by_stage_test[stage]
    embedding = cebra_models[stage].transform(test_run.states)

    r2 = cebra_analysis.calculate_location_r2(test_run, torch.tensor(embedding), decoder)
    r2_scores[stage] = r2


# %%
plots.set_font_size(9)

fig = plt.figure(figsize=plots.A5_L / [1, 1])
gs = gridspec.GridSpec(4, 4, figure=fig, height_ratios=[1, 1, 0.01, 1], width_ratios=[0.1, 1, 1, 1])

plt.figtext(0.05, 0.98, 'a', fontsize=16)
plt.figtext(0.05, 0.3, 'b', fontsize=16)

for idx, (stage, run) in enumerate(runs_by_stage.items(), start=1):
    ax = fig.add_subplot(gs[0, idx])
    plt.sca(ax)

    analysis.plot_pca_mpl_2d(run, pca=cebra_models[stage], alpha=0.5)
    plots.despine_topright()
    if idx == 1:
        plt.ylabel('CEBRA dim 2')
    # plt.xlabel('CEBRA dim 1')
    plt.title(f'{stage}\nLocation R²: {r2_scores[stage]:.2f}')

    ax = fig.add_subplot(gs[1, idx])
    plt.sca(ax)

    analysis.plot_pca_mpl_2d(
        run,
        pca=cebra_models[stage],
        color=['blue' if t == 'b' else 'black' for t in run.trial_types],
        n=5005,
        alpha=0.5,
    )
    plots.despine_topright()
    if idx == 1:
        plt.ylabel('CEBRA dim 2')
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='blue', label='Cued', linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='black', label='Uncued', linestyle='None'),
        ]
        plt.legend(
            handles=legend_elements,
            loc='lower left',
            frameon=False,
            handlelength=1,
            ncols=2,
            handletextpad=0.25,
            bbox_to_anchor=(0.0, 0.9),
        )
        # sns.move_legend(plt.gca(), ncol=1, bbox_to_anchor=(0.5, 1.0), frameon=False, handles=legend_elements, loc='lower left')

    plt.xlabel('CEBRA dim 1')

# TODO hardcoded to assume CEBRA model is 2d
embedding: torch.Tensor = cebra_models['Fully trained'].transform(runs_by_stage['Fully trained'].states)
for dim, s in enumerate(embedding.T, start=1):
    ax = fig.add_subplot(gs[3, dim])
    plt.scatter(runs_by_stage['Fully trained'].locations, s, alpha=0.5)
    plt.ylabel(f'CEBRA dim {dim}')
    analysis.plot_stops_base()

plt.figtext(0.35, 0.3, 'Fully trained', fontsize=9)

plt.tight_layout()
analysis.save_figure(figure_path / 'cebra_training_stages_plot.png')
plt.show()
