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

### Create and activate Python environment
It is recommended to create a virtual environment for using CODA to avoid any conflicts with existing Python installations. You can create a virtual environment using Anaconda:
```bash
conda create -n CODA-env python=3.9
conda activate CODA-env
```

### Install PyTorch

PyTorch installation depends on your system's CUDA version to enable GPU support. You should visit the [PyTorch official website](https://pytorch.org/get-started/locally/) to find the appropriate installation command based on your system configuration (operating system, CUDA version, etc.).

For example, the installation command for a system with CUDA 11.8 might look like this (you should use the command specific to your system setup):

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
Note: This is just an example. Please refer to the PyTorch website for the command that matches your system's specific requirements.

### Install CODA

To install CODA, simply use pip with the following command:

```bash
pip install icoda
```

This command will download and install the CODA package from PyPI, making it ready to use in your Python environment.

## Development Status and Tutorials
At present, we have released Tutorial 1, which demonstrates the alignment and spatial analysis functionalities of CODA. 
Tutorial 2, focusing on common domain identification, is also available.
For the tutorial, you can download the necessary data of tutorial 2 from [this link](https://drive.google.com/drive/folders/1mxpS6pA3uwfZzkg50FlfIEcMMfSsZC9Y?usp=sharing). 

Additional tutorials and expanded documentation are under active development. Future updates will enhance compatibility with other spatial transcriptomics toolkits and data formats.

Stay tuned for upcoming releases and improvements.

