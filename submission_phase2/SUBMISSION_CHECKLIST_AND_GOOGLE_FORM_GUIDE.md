# Final Submission Checklist & Google Form Guide (Phase 2 Evaluation)

**Course:** Explainable AI (DA351) – IIIT Dharwad  
**Submission Google Form Link:** [https://forms.gle/nSe7zwsssY4gPC7y8](https://forms.gle/nSe7zwsssY4gPC7y8)  
**Total Marks:** 20 Marks  

---

## 1. Submission Deliverables Checklist

Ensure all four mandatory deliverables are ready before submitting the form:

- [x] **A. Video Presentation (Max 10 Minutes):**
  - **Status:** Presentation slides (`A_Video_Presentation_Assets/presentation_slides.html`) and word-for-word script (`A_Video_Presentation_Assets/10_Minute_Presentation_Script.md`) prepared.
  - **Action Required:** Record your 10-minute presentation using Windows Game Bar (`Win + Alt + R`) or OBS, upload to Google Drive, set sharing to *"Anyone with the link can view"*, and paste the link in the form.

- [x] **B. Original Research Article (PDF):**
  - **File:** `B_Original_Research_Article/Original_Research_Article_IJCAI2023_Kenny_et_al.pdf`
  - **Paper Details:** *Advancing Post-Hoc Case-Based Explanation with Feature Highlighting* by Eoin M. Kenny, Eoin Delaney, Mark T. Keane (IJCAI 2023, pp. 3142–3150, DOI: 10.24963/ijcai.2023/350).
  - **Action Required:** Upload this PDF directly to the Google Form upload field for Deliverable B.

- [x] **C. Experimental Results and Comparison (PDF):**
  - **File:** `C_Experimental_Results_and_Comparison/Experimental_Results_and_Comparison.pdf`
  - **Contents Included:**
    1. Screenshots/extracts from the original paper (Page 1 header, Page 2 Fig 1, Page 5 Fig 2, Page 6 Fig 3) with exact page numbers clearly indicated.
    2. Published results clearly distinguished from implemented experimental results.
    3. Replicated baseline results on CUB-200-2011 (ResNet-34).
    4. Proposed improvement results (Multi-Scale LayerCAM + Normalized Cosine Matching).
    5. In-depth before-and-after comparisons with tables, curves, and qualitative visualizations.
    6. Supporting research literature (LayerCAM TIP 2021, Hypercolumns CVPR 2015, Effective Receptive Fields NeurIPS 2016).
  - **Action Required:** Upload this PDF directly to the Google Form upload field for Deliverable C.

- [x] **D. Updated Source Code:**
  - **Location:** `Updated_Source_Code/` (or the compressed repository archive).
  - **Contents Included:**
    1. `MODIFIED_FILES.md` identifying all new and modified code sections.
    2. `README.md` with complete execution and reproduction instructions.
    3. `reproduce_results.py` and `run_benchmark.py` for automated reproduction.
  - **Action Required:** Upload the source code ZIP archive or provide your GitHub repository link as required by the form.

---

## 2. Recommended Form Responses / Field Answers

When filling out the Google Form, use the following verified details:

* **Project Title:**  
  *Advancing Post-Hoc Case-Based Explanation with Multi-Scale LayerCAM and Normalized Cosine Matching*
* **Original Research Paper Title:**  
  *Advancing Post-Hoc Case-Based Explanation with Feature Highlighting*
* **Original Paper Authors & Venue:**  
  *Eoin M. Kenny, Eoin Delaney, Mark T. Keane (IJCAI 2023)*
* **Original Paper Link / DOI:**  
  *https://doi.org/10.24963/ijcai.2023/350* | *https://github.com/EoinKenny/IJCAI-2023*
* **Identified Research Gap:**  
  *1. Coarse Spatial Resolution: The original framework computes visual saliency exclusively from the terminal layer (layer4 of ResNet-34) at a coarse 7×7 resolution (32 px stride), diluting fine-grained anatomical markers like beaks and eyes.*  
  *2. Metric Distortion: Latent feature matching uses unnormalized Euclidean distance ($L_2$), which suffers from severe norm dominance when combining multi-layer features into 768-dim hypercolumns, causing matching accuracy to collapse.*
* **Proposed Improvement:**  
  *We introduce Multi-Scale LayerCAM (Jiang et al., IEEE TIP 2021) combining intermediate layer3 (14×14) and terminal layer4 (7×7) attributions with location-specific positive gradient weighting to quadruple spatial density. We extract 768-dimensional hypercolumns (Hariharan et al., CVPR 2015) and match candidate exemplars using Scale-Invariant Normalized Cosine Matching to eliminate norm dominance.*
* **Supporting Research Literature:**  
  *1. LayerCAM: Exploring Hierarchical Representations for Image Attribution (Jiang et al., IEEE TIP 2021)*  
  *2. Hypercolumns for Object Segmentation and Fine-Grained Localization (Hariharan et al., CVPR 2015)*  
  *3. Understanding the Effective Receptive Field in Deep CNNs (Luo et al., NeurIPS 2016)*
* **Core Experimental Results:**  
  *On 500 test images of CUB-200-2011 with ResNet-34 (83.60% accuracy):*  
  *1. Faithfulness: Deletion AUC significantly decreases from 0.1171 to 0.0989 ($t = -8.88, p = 1.22 \times 10^{-17}$), and Insertion AUC significantly increases from 0.6817 to 0.6989 ($t = +7.93, p = 1.40 \times 10^{-14}$).*  
  *2. Latent Metric Recovery: Normalized Cosine Matching recovers +9.40 percentage points over Euclidean distance in 14×14 hypercolumn space (46.60% vs 37.20%, $p = 2.48 \times 10^{-6}$), reaching 56.43% on valid queries.*  
  *3. Computational Efficiency: Negligible +1.4% total latency overhead (488.8 ms vs 482.1 ms) with flat 286.8 MB peak GPU VRAM on an NVIDIA RTX 3050.*

