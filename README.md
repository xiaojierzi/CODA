# CODA
## Integrative cross-sample alignment and spatially differential gene analysis for spatial transcriptomics
Our preprint **"Integrative cross-sample alignment and spatially differential gene analysis for spatial transcriptomics"** is now available on **bioRxiv**. 
Read the full preprint here: [https://www.biorxiv.org/content/10.1101/2025.06.05.653933](https://www.biorxiv.org/content/10.1101/2025.06.05.653933)

CODA is a computational framework designed for nonlinear alignment and spatial analysis across multiple spatial transcriptomics (ST) datasets. CODA simultaneously addresses the challenges of spatial misalignment and spatial gene variation by introducing:

- Global rigid and local nonlinear alignment in the embedding space
- Common domain identification through transformer-based keypoint matching
- A spatial cross-correlation metric to detect spatially consistent genes (SCGs) and spatially differential genes (SDGs)

CODA supports cross-platform datasets (e.g., 10X Visium, MERFISH) and enables efficient and scalable alignment and analysis across biological replicates, technologies, and conditions.

### Overview of CODA
![avatar](Pipeline/pipeline.png)

## Requirements and Installation
[![python >=3.9](https://img.shields.io/badge/python-%3E%3D3.9-brightgreen)](https://www.python.org/)
[![numpy 1.26.3](https://img.shields.io/badge/numpy-1.26.3-green)](https://pypi.org/project/numpy/)
[![pandas 2.2.2](https://img.shields.io/badge/pandas-2.2.2-yellowgreen)](https://pypi.org/project/pandas/)
[![scipy 1.13.1](https://img.shields.io/badge/scipy-1.13.1-blue)](https://pypi.org/project/scipy/)
[![matplotlib 3.9.4](https://img.shields.io/badge/matplotlib-3.9.4-yellow)](https://pypi.org/project/matplotlib/)
[![Pillow 10.2.0](https://img.shields.io/badge/Pillow-10.2.0-orange)](https://pypi.org/project/Pillow/)
[![scikit-image 0.24.0](https://img.shields.io/badge/scikit--image-0.24.0-red)](https://pypi.org/project/scikit-image/)
[![opencv-python 4.10.0.84](https://img.shields.io/badge/opencv--python-4.10.0.84-lightgrey)](https://pypi.org/project/opencv-python/)
[![anndata 0.10.8](https://img.shields.io/badge/anndata-0.10.8-blue)](https://pypi.org/project/anndata/)
[![scanpy 1.10.2](https://img.shields.io/badge/scanpy-1.10.2-red)](https://pypi.org/project/scanpy/)
[![bbknn 1.6.0](https://img.shields.io/badge/bbknn-1.6.0-lightblue)](https://pypi.org/project/bbknn/)
[![shapely 2.0.6](https://img.shields.io/badge/shapely-2.0.6-lightgrey)](https://pypi.org/project/shapely/)
[![kornia 0.7.3](https://img.shields.io/badge/kornia-0.7.3-green)](https://pypi.org/project/kornia/)
[![torch 2.4.0](https://img.shields.io/badge/torch-2.4.0-brightgreen)](https://pytorch.org/)
[![torchvision 0.19.0](https://img.shields.io/badge/torchvision-0.19.0-brightgreen)](https://pytorch.org/)


### Create and activate Python environment
It is recommended to create a virtual environment for using CODA to avoid any conflicts with existing Python installations. You can create a virtual environment using Anaconda:
```bash
conda create -n CODA-env python=3.9
conda activate CODA-env
```

### Install PyTorch with CUDA 11.8 support

CODA was developed and tested with **PyTorch 2.4.0 + CUDA 11.8**.
To ensure compatibility and performance, please follow the steps below to install PyTorch:

```bash
# Step 1: Install CUDA 11.8 runtime (if not already installed)
conda install cudatoolkit=11.8 -c pytorch

# Step 2: Install PyTorch with CUDA 11.8 support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> ✅ Note: This ensures you install the GPU version of PyTorch for CUDA 11.8.
>
> ❗️If you are using a CPU-only machine, replace the second line with:
>
> ```bash
> pip install torch torchvision torchaudio
> ```

Note: This is just an example. Please refer to the PyTorch website for the command that matches your system's specific requirements.

### Install CODA

To install CODA, simply use pip with the following command:

```bash
pip install icoda
```

This command will download and install the CODA package from PyPI, making it ready to use in your Python environment.

To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Development Status and Tutorials
At present, we have released Tutorial 1, which demonstrates the alignment and spatial analysis functionalities of CODA. 
Tutorial 2, focusing on common domain identification, is also available.
For the tutorial, you can download the necessary data of tutorial 2 from [this link](https://drive.google.com/drive/folders/1mxpS6pA3uwfZzkg50FlfIEcMMfSsZC9Y?usp=sharing). 

Additional tutorials and expanded documentation are under active development. Future updates will enhance compatibility with other spatial transcriptomics toolkits and data formats.

Stay tuned for upcoming releases and improvements.

