from typing import Dict, List

import torch
from tqdm.auto import tqdm

from sf.analysis import fit_pca
from sf.attribution import disruption
from sf.perturbations import RecordingScaleOutputWrapper
from sf.specific.systematic_perturbations import a_location_correlations
from sf.utils import Run, run_model


def secondary_argsort(a: torch.Tensor, b: torch.Tensor, n: int) -> torch.Tensor:
    topk_indices: torch.Tensor = torch.topk(a, n).indices
    order: torch.Tensor = torch.argsort(b[topk_indices], descending=True)
    final: torch.Tensor = topk_indices[order]
    return final


def get_argsorted_metrics(run: Run) -> Dict[str, torch.Tensor]:
    """Gets metrics that we use for disruption perturbations"""

    correlations: torch.Tensor = torch.tensor(a_location_correlations(run))
    disruption_results: disruption.DisruptionResults = disruption.calculate_disruption_scores(
        run, preprocessing='softmax'
    )
    displacements: torch.Tensor = disruption_results.displacements

    correlations_excluding_lazy: torch.Tensor = correlations.clone()
    correlations_excluding_lazy[displacements < 5] = -1
    correlations_only_lazy: torch.Tensor = correlations.clone()
    correlations_only_lazy[displacements >= 5] = -1

    pc2: torch.Tensor = torch.tensor(fit_pca(run, n_components=3).components_[1])

    argsorted_metrics: Dict[str, torch.Tensor] = {
        'correlation': torch.argsort(correlations, descending=True),
        'displacement': torch.argsort(displacements, descending=True),
        'pc2': torch.argsort(pc2, descending=True),
        'correlation_displacement': secondary_argsort(correlations, displacements, 128),
        'pc2_displacement': secondary_argsort(pc2, displacements, 128),
    }

    return argsorted_metrics


def analyse_metrics(
    run: Run,
    scaling_factor: float,
    argsorted_metrics: Dict[str, torch.Tensor],
    n_chunks: int,
    n_in_chunk: int,
    n_frames: int = 5_000,
) -> Dict[str, List[Run]]:
    """Chunkwise perturbation by following the metrics"""
    results: Dict[str, List[Run]] = {}
    for metric_name, metric_indices in argsorted_metrics.items():
        results[metric_name] = []
        for chunk in tqdm(torch.split(metric_indices, n_in_chunk)[0:n_chunks]):
            core = RecordingScaleOutputWrapper(run.model.core, indices=chunk, scaling_factor=scaling_factor)
            perturbed_run: Run = run_model(run.model_path, core_override=core, n_frames=n_frames)[0]
            results[metric_name].append(core.run_with_pre_states(perturbed_run))
    return results
