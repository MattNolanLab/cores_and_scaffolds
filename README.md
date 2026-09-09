# Cores and scaffolds

## Usage

### Environment

To reproduce the exact environment used for the paper, build the image with Podman (or Docker):

```sh
podman build -t cores-and-scaffolds .
```

Serve JupyterLab from the repository root:

```sh
podman run --rm -it \
    -p 127.0.0.1:8888:8888 \
    -v "$PWD:/code:Z" \
    -w /code \
    cores-and-scaffolds \
    jupyter lab \
      --ip=0.0.0.0 \
      --port=8888 \
      --no-browser \
      --ServerApp.allow_root=True \
      --ServerApp.root_dir=/code
```

Open the printed URL then navigate and run the demo notebook at `notebooks/001_demo.ipynb` which shows how to run and analyse a model.

### Figures and data

To reproduce figures, navigate to and run the appropriate script in in figures/.

All model checkpoints used in the paper are provided in this repo, but new ones can be trained using `libraries/sample-factory/sample_factory/train.py`.

The workflows in `figures/data/` generate the data used by the figure scripts:

- `data.Smk`: baseline simulations
- `clamp.Smk`: PC clamping and inclusions
- `single_neurons.Smk`: single neuron manipulations
- `rank.Smk`: rank truncation and scaling

Run `data.Smk` first. Once it finishes, run whichever other workflows you need, in any order. From a terminal inside the container:

```sh
cd /code
snakemake -s figures/data/data.Smk -c1 --rerun-incomplete

# Then run the workflows needed for your figures:
snakemake -s figures/data/clamp.Smk -c1 --rerun-incomplete
snakemake -s figures/data/single_neurons.Smk -c1 --rerun-incomplete
snakemake -s figures/data/rank.Smk -c1 --rerun-incomplete
```
