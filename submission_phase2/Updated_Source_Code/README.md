# Updated Source Code: Multi-Scale LayerCAM + Normalized Cosine Matching
## Course Project Phase 2 — Explainable AI (DA351)

- **Target Paper:** *Advancing Post-Hoc Case-Based Explanation with Feature Highlighting* (Kenny, Delaney & Keane, IJCAI 2023)
- **Original Paper GitHub Repository:** `https://github.com/EoinKenny/IJCAI-2023`
- **Our Project GitHub Repository:** `https://github.com/Vedha-11/XAI-Case-Based-Explanation`
- **Evaluation Dataset:** Caltech-UCSD Birds (`CUB-200-2011`, 500-Image Benchmark Subset, Seed=42)
- **Classifier Backbone:** Pretrained ResNet-34 (`SavedNetC` checkpoint, 83.60% top-1 accuracy)

---

## 1. Executive Summary of Improvements

This codebase implements two major architectural improvements to resolve foundational limitations in Kenny et al. (IJCAI 2023):
1. **Multi-Scale LayerCAM ($14\times14$ Grid Resolution):** Replaces single-layer $7\times7$ Class Activation Mapping with location-specific positive gradient weighting fusing intermediate `layer3` ($14\times14$) and terminal `layer4` ($7\times7$), quadrupling grid resolution from 49 to 196 cells (supported by Jiang et al., *IEEE TIP 2021*).
2. **Scale-Invariant Normalized Cosine Matching:** Solves Euclidean metric collapse by projecting 768-dimensional hypercolumns (`layer3` [256-d] + `layer4` [512-d]) onto the unit hypersphere, recovering $+9.40$ percentage points in part correspondence (supported by Hariharan et al., *CVPR 2015*).

---

## 2. Environment Setup & Dependencies

### Prerequisites
- Python 3.9, 3.10, or 3.11
- PyTorch 2.0+ (CPU or CUDA-enabled GPU)
- Standard scientific Python packages (`numpy`, `scipy`, `pandas`, `scikit-learn`, `pillow`)

### Installation Steps
```bash
# 1. Navigate to the project directory
cd Updated_Source_Code

# 2. Create and activate a clean virtual environment
python -m venv xai_env

# On Windows:
xai_env\Scripts\activate

# On Linux / macOS:
source xai_env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 3. Step-by-Step Execution Procedure

### Step 1: Instant Verification & Metric Reproduction (< 2 Seconds)
To verify and reproduce all quantitative benchmark results, paired $t$-tests, McNemar tests, and 2×2 factorial ablation tables reported in the presentation and comparison PDF:
```bash
python reproduce_results.py
```
**Expected Output:**
- Summary Benchmark Metrics across 500 CUB-200 images.
- Paired $t$-test for Deletion AUC ($t = -8.88, p = 1.22 \times 10^{-17}$).
- Paired $t$-test for Insertion AUC ($t = +7.93, p = 1.40 \times 10^{-14}$).
- McNemar test for Cosine vs. Euclidean recovery on $14\times14$ space ($p = 2.48 \times 10^{-6}$).
- 7 spatial tolerance rules with Holm-Bonferroni adjusted $p$-values.

### Step 2: Running the Benchmark Pipeline
To execute the live feature extraction, saliency generation, and nearest-neighbor matching across test images:
```bash
python run_benchmark.py
```

---

## 4. Directory & Module Structure

```
Updated_Source_Code/
├── README.md                          # Execution procedure, dependencies & quickstart
├── MODIFIED_FILES.md                  # Comprehensive identification of modified/new code
├── requirements.txt                   # Dependency specifications
├── reproduce_results.py               # Standalone reproduction & hypothesis testing script
├── run_benchmark.py                   # Benchmark pipeline execution script
├── data/                              # Benchmark dataset records (500 test images)
│   ├── original/                      # Per-image results, summary metrics, statistical tests
│   └── processed/                     # Common-valid spatial tolerance evaluations
├── models/
│   └── resnet_savednetc.py            # ResNet-34 architecture & multi-scale PyTorch hook manager
├── methods/
│   ├── multiscale_layercam.py         # Proposed Multi-Scale LayerCAM (layer3 + layer4 fusion)
│   ├── vanilla_cam.py                 # Replicated baseline CAM (Kenny et al. 7x7 grid)
│   ├── hypercolumn.py                 # 768-d hypercolumn feature extraction
│   └── matching.py                    # Normalized Cosine Matching vs. Euclidean distance
└── evaluation/
    ├── faithfulness.py                # 22-step Gaussian blur Deletion & Insertion AUC game
    └── spatial_tolerance.py           # Landmark hit rate evaluation across factorial conditions
```

---

## 5. Identification of Modified & New Modules

Detailed documentation of code differences relative to Kenny et al.'s original repo is in [`MODIFIED_FILES.md`](MODIFIED_FILES.md):
- **`methods/multiscale_layercam.py` [NEW]:** Location-specific positive gradient weighting and multi-scale fusion ($14\times14$).
- **`methods/hypercolumn.py` [NEW]:** Extraction of 768-d concatenated intermediate and terminal features.
- **`methods/matching.py` [NEW]:** Scale-invariant unit-hypersphere Normalized Cosine Matching.
- **`evaluation/faithfulness.py` [NEW]:** Continuous Gaussian blur Deletion and Insertion perturbation games.
- **`models/resnet_savednetc.py` [MODIFIED]:** Fixed 3-tuple logit unpacking `(logits, x, C)` and added zero-leak hook management.
- **Backbone Weights (`SavedNetC`):** 100% frozen (strictly post-hoc, 83.60% top-1 accuracy).

---

## 6. Official GitHub Repository Link

- **Repository URL:** `https://github.com/Vedha-11/XAI-Case-Based-Explanation`
- **Submission Form Field:** Provide this URL and the packaged `Updated_Source_Code.zip` file.
