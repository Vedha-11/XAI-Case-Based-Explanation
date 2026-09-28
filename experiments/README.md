# Experiments Directory

This directory documents the experimental workflows and replication scripts for Phase 1.

## Replication Notebook Location

To preserve exact relative path resolution (e.g. data loaders, module imports, relative paths to pre-trained weights in `original_repo/Expt 1/CUB/weights/`), the primary replication notebook is preserved in its original working environment:

- **Replication Notebook**: [`original_repo/Phase1_Replication.ipynb`](../original_repo/Phase1_Replication.ipynb)

## Original Experiment Modules

The underlying experiment code from the official paper is located under `original_repo/`:
- **Experiment 1 (CUB-200 / ImageNet Occlusion & Inclusion)**: [`original_repo/Expt 1/`](../original_repo/Expt%201/)
  - `CUB/`: CUB-200-2011 dataset code, model training, evaluation scripts (`testing.py`, `functions.py`).
  - `ImageNet/`: ImageNet evaluation scripts.
- **Experiment 2 (Agnostic / Feature Highlighting)**: [`original_repo/Expt 2/`](../original_repo/Expt%202/)
- **Experiment 3 (Human User Study Materials)**: [`original_repo/Expt 3/`](../original_repo/Expt%203/)

## Execution Note

Running `Phase1_Replication.ipynb` inside `original_repo/` ensures all relative paths for dataset indexing, model weights (`cnnNORMAL_best.pth`, `cnnNORMAL_latest.pth`), and utility functions load seamlessly.
