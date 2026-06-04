import pandas as pd

from sf import EXPERIMENTS
workdir: str(EXPERIMENTS)

import torch
from typing import Dict, List, Optional, Any

import sf.perturbations as perturbations
from sf.analysis import Run, get_exemplar_model, run_model
from sf.utils import disable_jit, rank_truncate
from tqdm.autonotebook import tqdm

EXEMPLAR_MODELS: List[str] = [get_exemplar_model().name]

envs_by_speed: Dict[str, str] = {
    'Normal': 'LinearNavigationContextual12Static3-v2',
    'Rnd Slow': 'LinearNavigationContextual12Static3Speedslow-v2',
    'Rnd Fast': 'LinearNavigationContextual12Static3Speedfast-v2',
    'Half zero': 'LinearNavigationContextual12Static3Speedhalfzero-v2',
    **{name: f'LinearNavigationContextual12Static3Speed{speed_str}-v2' for speed_str, name in [
        (f'constant_{str(s).replace(".", "p")}', f'Constant {s}') for s in [1.0, 1.5, 2.5, 3.5, 4.5]
    ]}}


rule all:
    input:
        expand('exp_001/models/{model_id}/derivatives/rank/df.pkl', model_id=EXEMPLAR_MODELS)

rule aggregate:
    input:
        expand('exp_001/models/{model_id}/derivatives/rank/raw/{mode}__{k}__{speed}.record',
            model_id=EXEMPLAR_MODELS, mode='normal', k=[512, 256, 224, 192, 128, 64, 32, 16, 8, 7, 6, 5, 4, 3, 2, 1], speed=envs_by_speed.keys()),
        expand('exp_001/models/{model_id}/derivatives/rank/raw/{mode}__{k}__{speed}.record',
            model_id=EXEMPLAR_MODELS,
            mode=[f'scale_{a:.2f}' for a in [0.00]],
            k=list(range(512)),
            speed=['Rnd Slow']),
        expand('exp_001/models/{model_id}/derivatives/rank/raw/{mode}__{k}__{speed}.record',
            model_id=EXEMPLAR_MODELS,
            mode=[f'scale_{a:.2f}' for a in [0.00, 0.25, 0.50, 0.75]],
            k=list(range(8)),
            speed=['Rnd Slow']),
    output:
        path = 'exp_001/models/{model_id}/derivatives/rank/df.pkl'
    run:
        rows: List[pd.DataFrame] = []
        for f in tqdm(input):
            row: pd.DataFrame = torch.load(f)

            del row['run']
            row['record_path'] = f

            rows.append(row)

        torch.save(pd.DataFrame(rows), output.path)


def get_svs(m: torch.Tensor) -> List[torch.Tensor]:
    m = m.detach()
    U, S, Vh = torch.linalg.svd(m, full_matrices=False)

    return [torch.outer(U[:, i], Vh[i, :]) * S[i] for i in range(m.shape[0])]


rule run:
    input:
        run_path = 'exp_001/models/{model_id}/derivatives/run.pkl'
    output:
        path = 'exp_001/models/{model_id}/derivatives/rank/raw/{mode}__{k}__{speed}.record',
    run:
        mode: str = wildcards.mode
        k: int = int(wildcards.k)
        speed: str = wildcards.speed
        alpha: Optional[float] = None  # only defined for scale_ perturbations

        original_run: Run = torch.load(input.run_path)
        original_run_initial = run_model(original_run.model_path, milestone_index_override=0, n_frames=5)[0]
        disable_jit()

        if mode == 'normal':
            target_m: torch.Tensor = rank_truncate(original_run.rnn.weight_hh_l0.detach(), k)
            W_hh: torch.Tensor = target_m
        elif mode == 'delta':
            W_0: torch.Tensor = original_run_initial.rnn.weight_hh_l0.detach()
            W_d: torch.Tensor = original_run.rnn.weight_hh_l0.detach() - W_0

            target_m: torch.Tensor = W_d
            W_hh = W_0 + rank_truncate(W_d, k)
        elif mode.startswith('scale_'):
            alpha = float(mode.split('_')[1])  # retained fraction
            W = original_run.rnn.weight_hh_l0.detach()
            sv_k = get_svs(W)[k]
            W_hh = (W - sv_k) + (alpha * sv_k)
            target_m = W_hh
        else:
            raise NotImplementedError

        run: Optional[Run]
        with perturbations.TransientWeightModificationRNN(original_run.rnn) as (_, w_hh):
            w_hh[:] = W_hh
            try:
                run: Run = run_model(original_run.model_path,core_override=original_run.rnn,env_name=envs_by_speed[speed])[0]
            except RuntimeError as e:
                if 'probability tensor contains either `inf`' in str(e):
                    run = None
                else:
                    raise e

        record: Dict[str, Any] = {
            'mode': mode, 'k': k, 'speed': speed, 'target': target_m, 'w_hh': W_hh, 'run': run,
            'alpha': alpha,
            'target_rank': torch.linalg.matrix_rank(target_m),
            'failure': run is None,
            'nb_reward_perc': 0.0 if run is None else run.nonbeaconed_reward_avg * 2,
            'b_reward_perc': 0.0 if run is None else run.beaconed_reward_avg * 2,
        }
        torch.save(record, output.path)

