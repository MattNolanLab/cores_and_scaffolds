# Cores and scaffolds

## Usage

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

To reproduce figures, navigate to and run the appropriate script in in figures/.
