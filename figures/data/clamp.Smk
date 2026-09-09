from sf import EXPERIMENTS
workdir: str(EXPERIMENTS)

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List

import sf.specific.clamp as clamp
from sf.analysis import Run, get_exemplar_model, get_valid_models, fit_pca, bin_single_neuron_activity_by_location
from tqdm.autonotebook import tqdm
from sf.utils import disable_jit, envs_by_speed

EXEMPLAR_MODEL = get_exemplar_model().name
MODEL_IDS = [EXEMPLAR_MODEL] + [m.name for m in get_valid_models()]


def parse_speed(run: Run, wildcard: str) -> str:
    if wildcard == '':
        return run.env_name
    else:
        return envs_by_speed[wildcard]


rule all:
    input:
        expand('exp_001/models/{model_id}/derivatives/{name}.pkl',
            model_id=MODEL_IDS,
            name=['pc_inclusions', 'pc_clamps']),
        expand('exp_001/models/{model_id}/derivatives/clamp_across_track.pkl',
            model_id=[EXEMPLAR_MODEL]),
        expand('exp_001/models/{model_id}/derivatives/{name}{speed}.pkl',
            model_id=[EXEMPLAR_MODEL],
            speed=['Rnd Slow', 'Rnd Fast'],
            name=['pc_inclusions', 'pc_clamps'])


rule clamp_pcs:
    wildcard_constraints:
        speed = ".*"  # To handle legacy pc_clamps.pkl name
    input:
        run_path = 'exp_001/models/{model_id}/derivatives/run.pkl'
    output:
        path = protected('exp_001/models/{model_id}/derivatives/pc_clamps{speed}.pkl'),
    run:
        run: Run = torch.load(input.run_path)
        disable_jit()

        by_pcs_clamped: Dict[int, Run] = {}
        for pc_i in tqdm(range(0, 15)):
            by_pcs_clamped[pc_i] = clamp.analyse_pc_clamp(run,pc_i, n_pcs=15, plot=False, env_name=parse_speed(run, wildcards.speed))

        torch.save(by_pcs_clamped, output.path)

rule include_pcs:
    wildcard_constraints:
        speed = ".*"  # To handle legacy pc_clamps.pkl name
    input:
        run_path = 'exp_001/models/{model_id}/derivatives/run.pkl'
    output:
        path = protected('exp_001/models/{model_id}/derivatives/pc_inclusions{speed}.pkl'),
    run:
        run: Run = torch.load(input.run_path)
        disable_jit()

        by_pcs_kept: Dict[int, Run] = {}
        for n_pcs in tqdm(range(1, 16)):
            by_pcs_kept[n_pcs] = clamp.analyse_pc_inclusions(run, n_pcs, plot=False, env_name=parse_speed(run, wildcards.speed))

        torch.save(by_pcs_kept, output.path)


rule clamp_across_track:
    input:
        run_path = 'exp_001/models/{model_id}/derivatives/run.pkl'
    output:
        path = protected('exp_001/models/{model_id}/derivatives/clamp_across_track.pkl')
    run:
        original_run: Run = torch.load(input.run_path)
        disable_jit()

        pca = fit_pca(original_run,n_components=15)
        pcs: np.ndarray = pca.transform(original_run.states.numpy()).T[0:2]
        location_binned_pcs: List[np.ndarray] = [bin_single_neuron_activity_by_location(torch.tensor(pc), original_run.locations_offset)[0] for pc in pcs]
        def pc_values_at(location: int) -> Tuple[float, float]:
            return float(location_binned_pcs[0][location]), float(location_binned_pcs[1][location])


        def simulate_clamp_at_location(location: int,
                                       n_frames: int = 15_000,
                                       ) -> Run:
            pc_1, pc_2 = pc_values_at(location)
            print(f'Simulating at {location}: {pc_1=:.2f}, {pc_2=:.2f}')
            return clamp.analyse_pc_clamp_multi(original_run, {0: pc_1, 1: pc_2}, pca=pca, plot=False, n_frames=n_frames)

        runs_across_track_clamp: Dict[int, Run] = {l: simulate_clamp_at_location(l) for l in tqdm(range(0,200,5))}

        torch.save(runs_across_track_clamp, output.path)

