# %%
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch

import sf.analysis as analysis
import sf.model_analysis as model_analysis
import sf.plots as plots
from sf.analysis import get_fig_dir
from sf.notebook_setup import setup_notebook

setup_notebook()
analysis.make_paper_theme()

figure_path: Path = get_fig_dir('training_success')

# %%
rewards_by_model: Dict[str, float] = {
    m.name: torch.load(m / 'derivatives/run.pkl').reward_avg for m in analysis.get_all_models()
}

# %%
model_analysis.extract_params(list(rewards_by_model.keys())[0])

mappings: Dict[str, Callable[[str], str]] = {
    'rnn_type': lambda f: {'rnn_relu_bias': 'ReLU', 'rnn_tanh_bias': 'Tanh'}[f],
    'encoder_mlp_layers': lambda f: {'': 0, '512': 1, '512_512_512_512': 4}[f],
    'decoder_mlp_layers': lambda f: {'': 0, '512': 1, '512_512_512_512': 4}[f],
    'rnn_size': lambda f: int(f),
    'seed': lambda f: int(f),
}


def make_df():
    data = []
    for model_name, reward in rewards_by_model.items():
        params: Dict[str, str] = model_analysis.extract_params(model_name)
        row = {param: mappings[param](value) for param, value in params.items() if param in mappings}
        row['reward'] = reward
        row['model_name'] = model_name
        data.append(row)

    return pd.DataFrame(data).set_index('model_name', verify_integrity=True)


df: pd.DataFrame = make_df()

# %%


def calculate_success_rate(data, param):
    success = data[data['reward'] > analysis.REWARD_THRESHOLD].groupby(param).size()
    total = data.groupby(param).size()
    return (success / total * 100).fillna(0)


readable_param_mappings: Dict[str, str] = {
    'rnn_type': 'Nonlinearity',
    'encoder_mlp_layers': 'Encoder\nlayers',
    'decoder_mlp_layers': 'Decoder\nlayers',
    'rnn_size': 'RNN size',
    'seed': 'Seed',
}


def plot_all_params(data: pd.DataFrame, params: List[str]):
    n_plots = len(params)
    plots.set_font_size(12)
    fig, axes = plt.subplots(1, n_plots, figsize=plots.A5.T / [1, 3], sharey=True, dpi=300)

    for idx, (ax, param) in enumerate(zip(axes, params)):
        plt.sca(ax)

        success_rates = calculate_success_rate(data, param)
        success_rates.plot(kind='bar', ax=ax)

        plt.title(f'{readable_param_mappings[param]}')
        plt.xlabel('')
        if idx == 0:
            plt.ylabel('Num models with\nreward > 95%')

    # TODO this is a bit complicated because we put an 'A' outside the plot
    plt.tight_layout(rect=[0.02, 0.0, 1.0, 0.93])
    # fig.text(0.01, 0.99, 'a', ha='left', va='top', fontsize=20)
    analysis.save_figure(figure_path / 'training_plot.png')
    plt.show()


plot_all_params(df, list(mappings.keys()))
