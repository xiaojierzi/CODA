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

## Development Status and Tutorials
We are currently finalizing the packaging and testing of CODA to support streamlined installation and broader compatibility. A PyPI release is planned in the near future.

At present, we have released Tutorial 1, which demonstrates the alignment and spatial analysis functionalities of CODA. 
Tutorial 2, focusing on common domain identification, is also available.

Additional tutorials and expanded documentation are under active development. Future updates will enhance compatibility with other spatial transcriptomics toolkits and data formats.

Stay tuned for upcoming releases and improvements.
