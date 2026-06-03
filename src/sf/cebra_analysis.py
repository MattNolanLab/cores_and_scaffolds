"""Utilities for the analysis of CEBRA data"""

from typing import Optional, Union

import cebra
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
from einops import rearrange
from sklearn.decomposition import PCA

from sf.plots import plot_track_vertical_base, set_yticks_track
from sf.utils import Run


def transform_linear_to_circular(positions: np.ndarray, min_location=0, max_location=200) -> np.ndarray:
    circumference = max_location - min_location

    normalized_positions = (positions - min_location) / circumference

    x_positions = np.cos(2 * np.pi * normalized_positions)
    y_positions = np.sin(2 * np.pi * normalized_positions)

    encoded = np.array([x_positions, y_positions]).T
    return encoded


def transform_circular_to_linear(coordinates, min_location=0, max_location=200) -> np.ndarray:
    circumference = max_location - min_location
    angles = np.arctan2(coordinates[:, 1], coordinates[:, 0])
    normalized_angles = (angles % (2 * np.pi)) / (2 * np.pi)
    positions = normalized_angles * circumference + min_location
    return positions


def fit_location_decoder(
    run: Run,
    cebra_model: cebra.CEBRA,
    labels: Optional[torch.Tensor] = None,
    decode_states: bool = False,
    mask: Optional[torch.Tensor] = None,
) -> cebra.KNNDecoder:
    states: torch.Tensor = run.states
    if decode_states:
        states = run.model.decoder(states).detach()

    if labels is None:
        labels = run.locations_offset

    # mask: shape (t,), boolean
    if mask is not None:
        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool
        states = states[mask]  # (t_masked, state_dim)
        labels = labels[mask]  # (t_masked,)

    embedding: np.ndarray = cebra_model.transform(states.detach().cpu().numpy())

    labels_np: np.ndarray = rearrange(labels.detach().cpu().numpy(), 't -> t 1')

    decoder = cebra.KNNDecoder(n_neighbors=36, metric='euclidean')
    decoder.fit(embedding, transform_linear_to_circular(labels_np[:, 0]))
    return decoder


def plot_decoding(
    run: Run,
    cebra_model: Union[cebra.CEBRA, PCA],
    decoder: cebra.KNNDecoder,
    rz_min=60 + 30,
    rz_max=80 + 30,
    embedding: Optional[torch.Tensor] = None,
) -> float:
    if embedding is not None:
        print(
            'User provided an embedding. Using that instead of run.states. User is probably performing perturbed-core experiments and wants to check what the decoding looks like'
        )
    else:
        embedding: np.ndarray = cebra_model.transform(run.states)

    labels: np.ndarray = rearrange(run.locations, 't -> t 1').numpy()

    predictions = decoder.predict(embedding)

    plt.plot(labels[:, 0], color='black', lw=5)
    plt.plot(transform_circular_to_linear(predictions), color='red', alpha=1.0)
    plt.xlim(0, 5000)
    set_yticks_track()

    plot_track_vertical_base(rz_min=rz_min, rz_max=rz_max)

    r2: float = sklearn.metrics.r2_score(transform_linear_to_circular(labels[:, 0]), predictions)
    plt.title(f'Decoding r2: {r2}')
    return r2


def plot_decoding_scatter(
    run: Run,
    cebra_model: Union[cebra.CEBRA, PCA],
    decoder: cebra.KNNDecoder,
    rz_min=60 + 30,
    rz_max=80 + 30,
    embedding: Optional[torch.Tensor] = None,
    alpha=0.2,
):
    if embedding is not None:
        print(
            'User provided an embedding. Using that instead of run.states. User is probably performing perturbed-core experiments and wants to check what the decoding looks like'
        )
    else:
        embedding: np.ndarray = cebra_model.transform(run.states)

    labels: np.ndarray = rearrange(run.locations, 't -> t 1').numpy()

    predictions = decoder.predict(embedding)

    plt.scatter(labels[:, 0], transform_circular_to_linear(predictions), color='black', alpha=alpha)
    plt.gca().plot([0, 1], [0, 1], transform=plt.gca().transAxes, color='black')
    plt.ylim(0, 200)
    plt.xlim(0, 200)

    plt.yticks([0, 100, 200])
    plt.xticks([0, 100, 200])


def calculate_location_r2(
    run: Run, embedding: torch.Tensor, decoder: cebra.KNNDecoder, mask: Optional[torch.Tensor] = None
) -> float:
    """Usually used when analysing decodability of perturbed states with a decoder trained on an unperturbed model"""
    labels: torch.Tensor = (
        run.locations_offset
    )

    if mask is not None:
        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool
        assert mask.ndim == 1
        assert mask.shape[0] == labels.shape[0]

        labels = labels[mask]  # (t_masked,)
        embedding = embedding[mask]  # (t_masked, embed_dim)

    labels_np: np.ndarray = rearrange(labels.detach().cpu().numpy(), 't -> t 1')
    embedding_np: np.ndarray = embedding.detach().cpu().numpy()

    predictions = decoder.predict(embedding_np)

    r2: float = sklearn.metrics.r2_score(transform_linear_to_circular(labels_np[:, 0]), predictions)
    return r2
