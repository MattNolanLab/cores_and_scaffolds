from typing import Dict

import torch

from sf.utils import Run


def calculate_mapping_matrices(run: Run) -> Dict[str, torch.Tensor]:
    W_hh: torch.Tensor = torch.clone(run.rnn.weight_hh_l0.detach())
    W_ih: torch.Tensor = torch.clone(run.rnn.weight_ih_l0.detach())
    B: torch.Tensor = torch.clone(run.rnn.bias_hh_l0.detach() + run.rnn.bias_ih_l0.detach())

    left_singular_vecs, singular_values, right_singular_vecs = torch.linalg.svd(W_hh, full_matrices=False)

    hxs_to_latent: torch.Tensor = left_singular_vecs[:, :].T  # [n_rank, 512]
    input_to_latent: torch.Tensor = hxs_to_latent @ W_ih  # [n_rank, 16]
    latent_to_latent: torch.Tensor = hxs_to_latent @ (W_hh @ hxs_to_latent.T)  # [n_rank, n_rank]

    # Bias
    latent_bias: torch.Tensor = hxs_to_latent @ B

    return {
        'hxs_to_latent': hxs_to_latent,
        'input_to_latent': input_to_latent,
        'latent_to_latent': latent_to_latent,
        'latent_bias': latent_bias,
    }
