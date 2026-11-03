# Advancing Post-Hoc Case-Based Explanation with Multi-Scale LayerCAM and Normalized Cosine Matching

**Course:** Explainable AI (DA351) – Course Project Phase 2 Evaluation  
**Institution:** Indian Institute of Information Technology, Dharwad (IIIT Dharwad)  
**Target Research Paper:** *Advancing Post-Hoc Case-Based Explanation with Feature Highlighting* (Kenny, Delaney & Keane, IJCAI 2023, Pages 3142–3150)  
**Official Paper Repository:** [https://github.com/EoinKenny/IJCAI-2023](https://github.com/EoinKenny/IJCAI-2023)  
**Project GitHub Repository:** [https://github.com/Vedha-11/XAI-Case-Based-Explanation](https://github.com/Vedha-11/XAI-Case-Based-Explanation)  
**Primary Deliverables Folder:** [`submission_phase2/`](submission_phase2/)  
**Master Submission ZIP Archive:** [`Phase_2_Final_Submission.zip`](Phase_2_Final_Submission.zip)  

---

## 1. Executive Summary & Mandatory Submission Deliverables

This repository contains the complete implementation, forensic audit, research gap resolution, experimental benchmark, presentation materials, and reproduction code for **Phase 2 Course Project Evaluation (Total Marks: 20)**.

In strict compliance with the Phase 2 Submission Guidelines, all required deliverables are organized inside [`submission_phase2/`](submission_phase2/):

| Guideline Deliverable | Location in Repository | Description & Contents |
| :--- | :--- | :--- |
| **A. Video Presentation** | [`submission_phase2/A_Video_Presentation_Assets/`](submission_phase2/A_Video_Presentation_Assets/) | • 12-slide master presentation: [`Phase_2_Project_Presentation.pptx`](submission_phase2/A_Video_Presentation_Assets/Phase_2_Project_Presentation.pptx)<br/>• 4-page printable presentation guide: [`6_Member_Team_Presentation_Guide.pdf`](submission_phase2/A_Video_Presentation_Assets/6_Member_Team_Presentation_Guide.pdf)<br/>• Spoken team script: [`6_Member_Team_Presentation_Script.md`](submission_phase2/A_Video_Presentation_Assets/6_Member_Team_Presentation_Script.md)<br/>• Interactive web slides: [`presentation_slides.html`](submission_phase2/A_Video_Presentation_Assets/presentation_slides.html) |
| **B. Original Research Article** | [`submission_phase2/B_Original_Research_Article/`](submission_phase2/B_Original_Research_Article/) | • Full original research paper: [`Original_Research_Article_IJCAI2023_Kenny_et_al.pdf`](submission_phase2/B_Original_Research_Article/Original_Research_Article_IJCAI2023_Kenny_et_al.pdf) (Kenny et al., IJCAI 2023, pp. 3142–3150)<br/>• Complete 7-page appendix: [`Original_Research_Article_Appendix_IJCAI2023.pdf`](submission_phase2/B_Original_Research_Article/Original_Research_Article_Appendix_IJCAI2023.pdf) |
| **C. Experimental Results & Comparison** | [`submission_phase2/C_Experimental_Results_and_Comparison/`](submission_phase2/C_Experimental_Results_and_Comparison/) | • Standalone 9-page PDF: [`Experimental_Results_and_Comparison.pdf`](submission_phase2/C_Experimental_Results_and_Comparison/Experimental_Results_and_Comparison.pdf)<br/>• Paper extracts from Pages 1, 2, 5, 6 with exact page citations.<br/>• Master 5-column side-by-side comparison table.<br/>• Literature citations (Jiang et al., TIP 2021; Hariharan et al., CVPR 2015; Petsiuk et al., BMVC 2018).<br/>• Master cross-reference table indexing every metric to its exact location in the original paper PDF. |
| **D. Updated Source Code** | [`submission_phase2/Updated_Source_Code/`](submission_phase2/Updated_Source_Code/) | • Complete, runnable modular codebase (`models/`, `methods/`, `evaluation/`)<br/>• Code modifications document: [`MODIFIED_FILES.md`](submission_phase2/Updated_Source_Code/MODIFIED_FILES.md)<br/>• Standalone reproduction script: [`reproduce_results.py`](submission_phase2/Updated_Source_Code/reproduce_results.py)<br/>• Standalone downloadable code zip: `Updated_Source_Code.zip` |
| **Submission Checklist** | [`submission_phase2/SUBMISSION_CHECKLIST_AND_GOOGLE_FORM_GUIDE.md`](submission_phase2/SUBMISSION_CHECKLIST_AND_GOOGLE_FORM_GUIDE.md) | Verification checklist and field-by-field instructions for the [Google Form](https://forms.gle/nSe7zwsssY4gPC7y8). |

---

## 2. The Two Core Research Gaps & Proposed Improvements

In visual classification, deep neural networks are black boxes. Case-Based Reasoning explains a prediction by retrieving a similar past training image. Kenny et al. (IJCAI 2023) proposed a **Twin-Explanation framework**: retrieving an exemplar bird and drawing bounding boxes on both query and exemplar birds to highlight the shared anatomical feature.

However, our technical audit revealed **two foundational research gaps** in their framework:

### Research Gap 1: Coarse 7×7 Spatial Attribution (The 'Thick Highlighter' Problem)
* **The Problem:** Kenny et al. computed Class Activation Mapping (CAM) exclusively on terminal `layer4` of ResNet-34. By `layer4`, a 224×224 image is shrunk to a coarse 7×7 grid (32-pixel stride) with an effective receptive field exceeding 100 pixels. The resulting 48×48 px bounding box blurs together fine bird parts (beak tip, eye ring, crown) with surrounding torso and background water.
* **Our Solution:** We implemented **Multi-Scale LayerCAM** (Jiang et al., *IEEE Transactions on Image Processing, 2021*). Instead of averaging gradients across entire channels, LayerCAM calculates individual positive weights for every pixel coordinate:
  $$w_{k,i,j} = \max\left(0, \frac{\partial y^c}{\partial A^k_{i,j}}\right)$$
  We fuse intermediate `layer3` (14×14 grid, 256-d) with terminal `layer4` (7×7 grid, 512-d), quadrupling spatial resolution to **14×14 (196 cells, 16-pixel stride)** and shrinking the bounding box to a tight 32×32 pixels.

### Research Gap 2: Distance Metric Distortion in Multi-Layer Space (The 'Broken Ruler' Problem)
* **The Problem:** Combining `layer3` (256-d) and `layer4` (512-d) creates a 768-dimensional hypercolumn (Hariharan et al., *CVPR 2015*). Kenny et al. used unnormalized Euclidean distance ($L_2$). In 768-d space, deep `layer4` activations have much larger numerical norms than intermediate `layer3`. Euclidean distance looks only at magnitude and completely ignores sharp `layer3` spatial features, causing matching accuracy to collapse from 51.80% down to 37.20%!
* **Our Solution:** We introduced **Scale-Invariant Normalized Cosine Matching**, projecting hypercolumns onto the unit hypersphere via $L_2$ normalization:
  $$S(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
  Cosine similarity evaluates pattern direction rather than raw vector magnitude, preventing `layer4` from overpowering `layer3` and recovering **+9.40 percentage points** ($p = 2.48 \times 10^{-6}$).

---

## 3. Verified Empirical Results (500 CUB-200 Test Images)

All experiments were evaluated on the pretrained ResNet-34 classifier (`SavedNetC`, 83.60% top-1 accuracy) kept 100% frozen (strictly post-hoc) across the first 500 test images of Caltech-UCSD Birds (`CUB-200-2011`, seed=42):

```text
========================================================================================================
Condition / Method           Grid & Dim    Distance Metric     Deletion AUC (↓)   Insertion AUC (↑)
========================================================================================================
1. Baseline CAM L2 (Kenny)   7×7 (512-d)   Euclidean (L2)      0.1171 ± 0.1000    0.6817 ± 0.2369
2. 7×7 Cosine Ablation       7×7 (512-d)   Cosine Similarity   0.1171 ± 0.1000    0.6817 ± 0.2369
3. LayerCAM L2 Ablation      14×14 (768-d) Euclidean (L2)      0.0989 ± 0.0896    0.6989 ± 0.2480
4. Proposed Extension        14×14 (768-d) Cosine Similarity   0.0989 ± 0.0896    0.6989 ± 0.2480
--------------------------------------------------------------------------------------------------------
Paired Difference (Proposed vs. Baseline):                     Δ = -0.0181        Δ = +0.0171
Two-tailed Paired t-test:                                      t = -8.88          t = +7.93
Statistical Significance:                                      p = 1.22 × 10⁻¹⁷   p = 1.40 × 10⁻¹⁴
========================================================================================================
```

### Key Quantitative Findings:
1. **Faithfulness Perturbation Game (22-step Gaussian blur):** Blurring our highlighted region drops model confidence significantly faster (Deletion AUC 0.0989 vs. 0.1171, $p = 1.22 \times 10^{-17}$), and restoring it recovers confidence faster (Insertion AUC 0.6989 vs. 0.6817, $p = 1.40 \times 10^{-14}$). 79.6% of all test images showed faster confidence drops under our method.
2. **Metric Ablation in 14×14 Space:** When resolution was increased to 14×14 under Euclidean distance, Part Hit Rate crashed by **-14.60 pp** (down to 37.20%, $p = 5.8 \times 10^{-8}$). Normalized Cosine Matching produced a **+9.40 pp recovery** back up to 46.60% ($p = 2.48 \times 10^{-6}$), proving cosine normalization is mathematically mandatory.
3. **Computational Feasibility (NVIDIA RTX 3050 Laptop GPU):** Total latency is 488.8 ms vs. 482.1 ms (+1.4% overhead) with an identical flat 286.8 MB peak GPU VRAM allocation (zero memory leaks).

---

## 4. Execution Procedure, Dependencies & Reproduction Steps

### Prerequisites
- Python 3.9, 3.10, or 3.11
- PyTorch 2.0+ (CUDA GPU or CPU)
- Standard scientific Python libraries: `numpy`, `scipy`, `pandas`, `scikit-learn`, `pillow`

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/Vedha-11/XAI-Case-Based-Explanation.git
cd XAI-Case-Based-Explanation

# Create and activate a clean virtual environment
python -m venv xai_env

# On Windows:
xai_env\Scripts\activate

# On Linux / macOS:
source xai_env/bin/activate

# Install dependencies
pip install -r submission_phase2/Updated_Source_Code/requirements.txt
```

### Step 2: Instant Reproduction of All Benchmark Results (< 2 Seconds)
To verify and reproduce all quantitative benchmark results, summary tables, paired $t$-tests, and ablation statistics:
```bash
python submission_phase2/Updated_Source_Code/reproduce_results.py
```

### Step 3: Running Live Inference Pipeline
To run live multi-scale feature extraction, saliency generation, and nearest-neighbor matching:
```bash
python submission_phase2/Updated_Source_Code/run_benchmark.py
```

### Step 4: Rebuilding Deliverables
```bash
# To recompile the 9-page Deliverable C Comparison PDF:
python submission_phase2/C_Experimental_Results_and_Comparison/build_comparison_pdf.py

# To recompile the 12-slide Deliverable A PowerPoint Presentation (.pptx):
python submission_phase2/A_Video_Presentation_Assets/generate_presentation_pptx.py

# To recompile the 4-page Deliverable A Team Presentation Guide (.pdf):
python _archive_old_tasks/build_team_presentation_pdf.py
```

---

## 5. Code Structure & Identification of Modified/New Files

```
XAI-Case-Based-Explanation/
├── README.md                                     # Root project documentation & reproduction guide
├── .gitignore                                    # Git ignore rules (virtual environments, caches, archives)
├── submission_phase2/                            # MASTER CANONICAL SUBMISSION DIRECTORY
│   ├── 00_START_HERE_SUBMISSION_GUIDE.md         # Submission overview & quick-start guide
│   ├── SUBMISSION_CHECKLIST_AND_GOOGLE_FORM_GUIDE.md # Pre-submission checklist & form answers
│   ├── Updated_Source_Code.zip                   # Standalone packaged code zip for Google Form
│   ├── A_Video_Presentation_Assets/              # Video presentation deliverables
│   │   ├── Phase_2_Project_Presentation.pptx     # 12-slide master presentation (speaker notes embedded)
│   │   ├── 6_Member_Team_Presentation_Guide.pdf  # 4-page clean printable presentation guide
│   │   ├── 6_Member_Team_Presentation_Script.md  # Word-for-word spoken presentation script
│   │   ├── presentation_slides.html              # Interactive web presentation deck
│   │   └── figures/                              # Extracted paper figures & empirical plots
│   ├── B_Original_Research_Article/              # Original paper deliverables
│   │   ├── Original_Research_Article_IJCAI2023_Kenny_et_al.pdf # Kenny et al. (IJCAI 2023)
│   │   └── Original_Research_Article_Appendix_IJCAI2023.pdf    # 7-page conference appendix
│   ├── C_Experimental_Results_and_Comparison/    # Comparison report deliverables
│   │   ├── Experimental_Results_and_Comparison.pdf # 9-page comprehensive comparison report
│   │   ├── build_comparison_pdf.py               # PDF generation script
│   │   └── extracted_paper_figures/              # Extracted screenshots from Pages 1, 2, 5, 6
│   └── Updated_Source_Code/                      # Standalone source code directory
│       ├── README.md                             # Code execution procedure & reproduction steps
│       ├── MODIFIED_FILES.md                     # File-by-file identification of changes
│       ├── requirements.txt                      # Dependency specifications
│       ├── reproduce_results.py                  # Standalone verification script
│       ├── run_benchmark.py                      # Live benchmark pipeline script
│       ├── data/                                 # 500-image benchmark dataset & statistics
│       ├── models/resnet_savednetc.py            # [MODIFIED] 3-tuple logit fix; PyTorch hook manager
│       ├── methods/multiscale_layercam.py        # [NEW] Multi-Scale LayerCAM implementation
│       ├── methods/hypercolumn.py                # [NEW] 768-d hypercolumn extraction
│       ├── methods/matching.py                   # [NEW] Normalized Cosine Matching implementation
│       ├── methods/vanilla_cam.py                # [REFACTORED] Replicated baseline CAM
│       └── evaluation/                           # [NEW] Faithfulness AUC & spatial tolerance modules
```

---

## 6. Literature Citations & Supporting Articles

1. **Original Research Article (Phase 1 Baseline):**  
   Kenny, E. M., Delaney, E., & Keane, M. T. (2023). *Advancing Post-Hoc Case-Based Explanation with Feature Highlighting.* In **Proceedings of the 32nd International Joint Conference on Artificial Intelligence (IJCAI 2023)**, pages 3142–3150. DOI: [10.24963/ijcai.2023/350](https://doi.org/10.24963/ijcai.2023/350).
2. **Supporting Article for Multi-Scale Saliency Attribution (Solving Gap 1):**  
   Jiang, P. T., Zhang, C. B., Hou, Q., Cheng, M. M., & Wei, Y. (2021). *LayerCAM: Exploring Hierarchical Representations for Image Attribution.* **IEEE Transactions on Image Processing (IEEE TIP)**, Vol. 30, pages 5875–5888. DOI: [10.1109/TIP.2021.3089943](https://doi.org/10.1109/TIP.2021.3089943).
3. **Supporting Article for Hypercolumn Representation & Cosine Normalization (Solving Gap 2):**  
   Hariharan, B., Arbeláez, P., Girshick, R., & Malik, J. (2015). *Hypercolumns for Object Segmentation and Fine-Grained Localization.* In **Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2015)**, pages 447–456. DOI: [10.1109/CVPR.2015.7298642](https://doi.org/10.1109/CVPR.2015.7298642).
4. **Supporting Article for Saliency Faithfulness Evaluation (Methodology):**  
   Petsiuk, V., Das, A., & Saenko, K. (2018). *RISE: Randomized Input Sampling for Explanation of Black-box Models.* In **Proceedings of the British Machine Vision Conference (BMVC 2018)**, pages 1–12.
