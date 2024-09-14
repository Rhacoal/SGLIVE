# Segmentation-guided Layer-wise Vectorization with Gradient Fills

[![arXiv](https://img.shields.io/badge/arXiv-2408.15741-b31b1b.svg)](https://arxiv.org/abs/2408.15741)

This repo requires PyTorch and torchvision to work.
Please refer to [Start Locally | PyTorch](https://pytorch.org/get-started/locally/) on how to install PyTorch.

We have tested on PyTorch 2.1.1 with Python 3.10.13. The environment is saved to `env.yml`.

This work is largely inspired by [LIVE](https://github.com/Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization).

### Install DiffVG

The directory `DiffVG` contains a forked version of the original [DiffVG](https://github.com/BachiLi/diffvg).
We edited `DiffVG/pydiffvg/save_svg.py` to save radial gradient parameters.

With the current working directory changed to `DiffVG`:

```bash
conda install -y numpy scikit-image cmake
conda install -y -c conda-forge ffmpeg
pip install svgwrite
pip install svgpathtools
pip install cssutils
pip install numba
pip install torch-tools
pip install visdom
python setup.py install
```

### Install SGLIVE Dependencies

```bash
pip install opencv-python==4.5.4.60 
pip install torchmetrics
pip install easydict
```

## Run

With the current working directory changed to `SGLIVE`:

```bash
python main.py --config config/sglive.yaml --experiment experiment_16x1 --signature noto_u1f61a --target data/noto_u1f61a.png
```
