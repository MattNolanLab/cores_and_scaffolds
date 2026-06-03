from typing import List, Optional

import captum.attr as attr
import torch
from einops import rearrange
from tqdm.autonotebook import tqdm

from sf.utils import Run


class CaptumRNNWrapper(torch.nn.Module):
    """Indended for Neuron Conductance wrapping"""

    def __init__(self, run: Run) -> None:
        super().__init__()

        self.rnn = run.rnn
        self.encoder = run.model.encoder.encoders.obs

    def forward(self, x):
        x = self.encoder(x)
        return self.rnn(x)[0]


def encoding_attributions(
    inputs: torch.Tensor, run: Run, n_steps: Optional[int] = 50, baselines: Optional[torch.Tensor] = None
) -> torch.Tensor:
    assert inputs.ndim == 2
    assert inputs.shape[0] == 1
    assert n_steps >= 2

    model: torch.nn.Module = CaptumRNNWrapper(run)

    nc = attr.IntegratedGradients(model)

    attributions: List[torch.Tensor] = []
    for neuron_i in tqdm(range(model.rnn.hidden_size), desc='Running attributions'):
        attribution = nc.attribute(inputs, target=neuron_i, n_steps=n_steps, baselines=baselines)

        attributions.append(rearrange(attribution, '1 n -> n'))

    return torch.Tensor(rearrange(attributions, 'n i -> n i'))
