# XAI Case-Based Explanation Replication

Replication and organization of post-hoc case-based explanation research with feature highlighting for computer vision models.

## Selected Paper

- **Title**: *"Advancing Post-Hoc Case-Based Explanation with Feature Highlighting"*
- **Authors**: Eoin M. Kenny, Eoin Delaney, Mark T. Keane
- **Conference**: IJCAI 2023 Main Track
- **Paper Link**: [IJCAI 2023 Proceedings](https://www.ijcai.org/proceedings/2023/0048)
- **Official Repository**: [EoinKenny/IJCAI-2023](https://github.com/EoinKenny/IJCAI-2023)

---

## Project Objective

The goal of this project is to replicate and analyze post-hoc case-based explanations augmented with feature highlighting. Case-based reasoning (CBR) explanation methods select nearest-neighbor reference images from training data to explain model predictions; feature highlighting further identifies which specific image segments drive these decisions.

---

## Phase 1 Replication & Reproduced Results

Phase 1 focuses on reproducing the CUB-200-2011 quantitative evaluation using 500 evaluation images across superpixel segmentation granularities (10, 20, 30, 40, 50 segments).

### Reproduced Figures & Artifacts

1. **Figure 2(A): CUB-200 Occlusion Evaluation**
   - **CSV Replication Data**: [`results/Figure2A_CUB_Occlusion_Replication_500Images.csv`](results/Figure2A_CUB_Occlusion_Replication_500Images.csv)
   - **CSV Summary Data**: [`results/Figure2A_CUB_Occlusion_Summary_500Images.csv`](results/Figure2A_CUB_Occlusion_Summary_500Images.csv)
   - **Generated Plot**: [`figures/Figure2A_CUB_Occlusion_Replication_500Images.png`](figures/Figure2A_CUB_Occlusion_Replication_500Images.png)

2. **Figure 2(C): CUB-200 Inclusion Evaluation**
   - **CSV Replication Data**: [`results/Figure2C_CUB_Inclusion_Replication_500Images.csv`](results/Figure2C_CUB_Inclusion_Replication_500Images.csv)
   - **CSV Summary Data**: [`results/Figure2C_CUB_Inclusion_Summary_500Images.csv`](results/Figure2C_CUB_Inclusion_Summary_500Images.csv)
   - **Generated Plot**: [`figures/Figure2C_CUB_Inclusion_Replication_500Images.png`](figures/Figure2C_CUB_Inclusion_Replication_500Images.png)

---

## Experimental Setup

- **Dataset**: CUB-200-2011 (Caltech-UCSD Birds-200-2011)
- **Classifier Model**: ResNet34
- **Evaluated XAI Methods**:
  - **Superpixel**: SLIC-based superpixel segmentation & insertion/deletion.
  - **CAM**: Class Activation Mapping.
  - **FAM**: Feature Activation Mapping.
  - **Random**: Random superpixel selection baseline.
- **Evaluation Subset**: 500 test images.
- **Segmentation Settings**: 10, 20, 30, 40, and 50 segments.

---

## Project Structure

```text
XAI-Case-Based-Explanation/
│
├── original_repo/           # Reference implementation & replication scripts
│   ├── Expt 1/              # Experiment 1 code (CUB-200 & ImageNet evaluation)
│   ├── Expt 2/              # Experiment 2 code (Agnostic / Feature Highlighting)
│   ├── Expt 3/              # Experiment 3 code (User Study materials)
│   └── Phase1_Replication.ipynb  # Main replication notebook
│
├── experiments/             # Documentation of experimental execution & scripts
│   └── README.md
│
├── results/                 # Verified Phase 1 quantitative results (CSVs)
│   ├── Figure2A_CUB_Occlusion_Replication_500Images.csv
│   ├── Figure2A_CUB_Occlusion_Summary_500Images.csv
│   ├── Figure2C_CUB_Inclusion_Replication_500Images.csv
│   └── Figure2C_CUB_Inclusion_Summary_500Images.csv
│
├── figures/                 # Verified Phase 1 generated plots & figures
│   ├── Figure2A_CUB_Occlusion_Replication_500Images.png
│   └── Figure2C_CUB_Inclusion_Replication_500Images.png
│
├── datasets/                # Dataset setup documentation
│   └── README.md
│
├── notes/                   # Replication notes & observations
│   └── README.md
│
├── phase1_report/           # Phase 1 summary report
│   └── README.md
│
├── README.md                # Top-level project README
├── requirements.txt         # Project dependencies
├── AI_USAGE.md              # Documentation of AI assistance
└── .gitignore               # Git exclude rules for data, weights, and venvs
```

---

## Environment Setup & Reproduction

### 1. Prerequisites

Python 3.10+ with PyTorch and CUDA (optional for GPU acceleration).

### 2. Environment Installation

```bash
# Clone the repository
git clone <REPOSITORY_URL>
cd XAI-Case-Based-Explanation

# Install dependencies
pip install -r requirements.txt
```

### 3. Dataset Setup

Download the CUB-200-2011 dataset from the [official Caltech website](https://www.vision.caltech.edu/datasets/CUB_200_2011/) and extract it into:
```text
original_repo/Expt 1/CUB/data/CUB_200_2011/
```
See [`datasets/README.md`](datasets/README.md) for details.

### 4. Running Experiments

Launch Jupyter Notebook inside `original_repo/` to run the replication workflow:
```bash
cd original_repo
jupyter notebook Phase1_Replication.ipynb
```
Outputs and summary statistics will update in `results/` and `figures/`.

---

## Citation

```bibtex
@inproceedings{kenny2023advancing,
  title     = {Advancing Post-Hoc Case-Based Explanation with Feature Highlighting},
  author    = {Kenny, Eoin M. and Delaney, Eoin and Keane, Mark T.},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence (IJCAI-23)},
  pages     = {48--56},
  year      = {2023},
  url       = {https://www.ijcai.org/proceedings/2023/0048}
}
```
