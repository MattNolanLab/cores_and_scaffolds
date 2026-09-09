from sf import EXPERIMENTS
workdir: str(EXPERIMENTS)

import torch
from typing import List, Iterable
from pathlib import Path

from sf.analysis import Run, get_exemplar_model, get_all_models
from sf.utils import run_model, disable_jit


MODEL_IDS: List[str] = [m.name for m in get_all_models()]
EXEMPLAR_MODELS: List[str] = [get_exemplar_model().name]

rule all:
    input:
        expand('exp_001/models/{model_id}/derivatives/run.pkl',
            model_id=MODEL_IDS + EXEMPLAR_MODELS) +
        expand('exp_001/models/{model_id}/derivatives/run_high.pkl',
            model_id=EXEMPLAR_MODELS)

rule run_models:
    input:
        'exp_001/models/{model_id}/config.json'  # TODO: bit of a hack
    output:
        protected('exp_001/models/{model_id}/derivatives/{run_type}.pkl')
    params:
        n_frames=lambda wildcards: {'run_high': 20_000, 'run': 5_000}[wildcards.run_type]
    run:
        disable_jit()
        run: Run = run_model(Path(input[0]).parent, n_frames=params.n_frames, verbose=False)[0]
        torch.save(run, output[0])
