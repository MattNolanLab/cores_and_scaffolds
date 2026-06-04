from sf import EXPERIMENTS
workdir: str(EXPERIMENTS)

import torch
import numpy as np

from sf.specific.systematic_perturbations import get_indices, run_perturbations, run_perturbations_scaling
from sf.attribution.disruption import DisruptionResults, calculate_disruption_scores
from sf.analysis import Run, get_exemplar_model, get_valid_models, get_other_valid_models
from sf.utils import disable_jit
from typing import Dict, Tuple, List

MODEL_IDS: List[str] = [get_exemplar_model().name]

OTHER_MODEL_IDS: List[str] = [m.name for m in get_valid_models()]


rule all:
    input:
        expand('exp_001/models/{model_id}/derivatives/single/zeroing_{score}.pkl',
             model_id=MODEL_IDS,
             score=['location_correlation', 'speed_correlation', 'pca0', 'pca1', 'pca2', 'speed_attribution', 'displacement'] + [f'random_{s}' for s in range(10)]
         ),
        expand('exp_001/models/{model_id}/derivatives/single/zeroing_{score}.pkl',
            model_id=OTHER_MODEL_IDS,
            score=['location_correlation', 'disruption', 'displacement'] + [f'random_{s}' for s in range(3)]
        ),
        expand('exp_001/models/{model_id}/derivatives/single/scaling_{score}.pkl',
                model_id=MODEL_IDS,
                score=['location_correlation', 'speed_correlation', 'pca0', 'pca1', 'pca2', 'speed_attribution', 'displacement'] +[f'random_{s}' for s in range(5)]
         )

rule zero_neurons:
    input:
        run_path = 'exp_001/models/{model_id}/derivatives/run.pkl'
    output:
        path = protected('exp_001/models/{model_id}/derivatives/single/zeroing_{score}.pkl')
    params:
        minimal = lambda w: w.model_id not in MODEL_IDS
    run:
        run: Run = torch.load(input.run_path)
        disable_jit()

        score_name: str = wildcards.score
        if score_name == 'disruption':
            indices: torch.tensor = calculate_disruption_scores(run).scores
        elif score_name == 'displacement':
            indices = calculate_disruption_scores(run).displacements
        else:
            indices: torch.tensor = get_indices(run)[score_name]

        ns: Tuple[int, ...]
        if params.minimal:
            ns = tuple({
                64: range(0,8,1),
                256: range(0,11,2),
                512: range(0,21,2),
            }[run.rnn.hidden_size])
        else:
            ns = tuple({
                64: range(0,14,1),
                256: range(0,51,2),
                512: range(0,102,2),
            }[run.rnn.hidden_size])

        data: Dict[str, Dict[int, Run]] = {}
        for sign, descending in [('+', True), ('-', False)]:
            data[sign + score_name] = run_perturbations(run,
                0.0,
                indices=indices,
                descending=descending,
                ns=ns
            )

        torch.save(data, output.path)


rule scale_neurons:
    input:
        run_path = 'exp_001/models/{model_id}/derivatives/run.pkl'
    output:
        path = protected('exp_001/models/{model_id}/derivatives/single/scaling_{score}.pkl')
    run:
        run: Run = torch.load(input.run_path)
        disable_jit()

        score_name: str = wildcards.score
        if score_name == 'disruption':
            indices: torch.tensor = calculate_disruption_scores(run).scores
        elif score_name == 'displacement':
            indices = calculate_disruption_scores(run).displacements
        else:
            indices: torch.tensor = get_indices(run)[score_name]

        n_perturb: int = 8  # TODO hardcoded - this should depend on the model
        scaling_factors: Tuple[float, ...] = tuple(np.geomspace(0.30,2.25,12))

        data: Dict[str, Dict[float, Run]] = {}
        for sign, descending in [('+', True), ('-', False)]:
            data[sign + score_name] = run_perturbations_scaling(run,
                indices=indices,
                n_perturb=n_perturb,
                scaling_factors=scaling_factors,
                descending=descending,
            )

        torch.save(data, output.path)
