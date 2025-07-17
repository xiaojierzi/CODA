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

## Development Status and Tutorials
At present, we have released Tutorial 1, which demonstrates the alignment and spatial analysis functionalities of CODA. 
Tutorial 2, focusing on common domain identification, is also available.
For the tutorial, you can download the necessary data of tutorial 2 from [this link](https://drive.google.com/drive/folders/1mxpS6pA3uwfZzkg50FlfIEcMMfSsZC9Y?usp=sharing). 

Additional tutorials and expanded documentation are under active development. Future updates will enhance compatibility with other spatial transcriptomics toolkits and data formats.

Stay tuned for upcoming releases and improvements.

