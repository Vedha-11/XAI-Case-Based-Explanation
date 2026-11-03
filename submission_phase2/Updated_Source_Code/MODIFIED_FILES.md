# Identification of Modified and New Files (Phase 2 Submission)

This document explicitly identifies all modifications, additions, and preserved components relative to the original research repository:  
**Original Repository:** *Advancing Post-Hoc Case-Based Explanation with Feature Highlighting* (Kenny et al., IJCAI 2023)  
**URL:** `https://github.com/EoinKenny/IJCAI-2023`

---

## 1. Summary of Architectural & Methodological Changes

| Component | Original Method (Kenny et al., IJCAI 2023) | Proposed Extended Method (This Work) | Impact / Purpose |
| :--- | :--- | :--- | :--- |
| **Saliency Architecture** | Vanilla Class Activation Mapping (CAM) at terminal `layer4` | **Multi-Scale LayerCAM** combining `layer3` ($14\times14$) and `layer4` ($7\times7$) | Reduces spatial stride from 32 px to 16 px; eliminates channel-pooling dilution. |
| **Saliency Fusion** | None (single layer) | **Bilinear upsampling + Convex Fusion** ($\alpha = 0.5$) | Fuses fine edge localization with semantic category context. |
| **Latent Feature Space** | 512-dimensional vector from `layer4` | **768-dimensional Hypercolumn** (`layer3` [256-d] + `layer4` [512-d]) | Bridges localized sub-organ representation with category traits. |
| **Matching Metric** | Unnormalized Euclidean distance ($L_2$) | **Scale-Invariant Normalized Cosine Matching** | Prevents norm dominance of `layer4`; enforces directional alignment ($+9.40$ pp recovery). |
| **Faithfulness Benchmark** | Single occlusion/inclusion point | **22-step continuous Gaussian blur Deletion & Insertion Game** | Rigorous quantitative causal evaluation (Petsiuk et al., BMVC 2018). |
| **Part Correspondence Evaluation** | Single-condition heuristic check | **Factorial 2×2 Ablation Evaluation** across resolution and distance metrics | Isolates independent causal effects of spatial resolution and latent metric choice. |

---

## 2. File-by-File Breakdown

### A. New Modules Implemented (Proposed Improvement)
1. **`methods/multiscale_layercam.py` [NEW]:**
   - Implements LayerCAM formula with location-specific positive weights: $w_{k,i,j} = \max(0, \partial z_c / \partial A_{k,i,j})$.
   - Handles multi-scale extraction across intermediate (`layer3`) and terminal (`layer4`) layers.
   - Bilinearly upsamples `layer4` attributions to $14\times14$ and performs convex fusion.
2. **`methods/hypercolumn.py` [NEW]:**
   - Implements spatial coordinate mapping from $14\times14$ to $7\times7$.
   - Concatenates 256-d `layer3` and 512-d `layer4` vectors into 768-d hypercolumn descriptors.
3. **`methods/matching.py` [NEW]:**
   - Implements scale-invariant Normalized Cosine Matching on unit hypersphere.
   - Implements baseline Euclidean matching for factorial ablation comparison.
4. **`evaluation/faithfulness.py` [NEW]:**
   - Implements 22-step Gaussian blur Deletion AUC and Insertion AUC computation.
5. **`evaluation/part_hit.py` / `spatial_tolerance.py`:**
   - Implements landmark hit evaluation and metric comparison across factorial conditions.

### B. Modified Modules from Original Framework
1. **`models/resnet_savednetc.py` [MODIFIED]:**
   - Fixed forward-pass 3-tuple handling: `logits, x, C = model(imgs)`.
   - Added `MultiScaleFeatureExtractor` forward and full-backward hooks for PyTorch 2.x compatibility.
   - Added explicit graph detachment and hook cleanup to eliminate GPU memory leaks.
2. **`methods/vanilla_cam.py` [REFACTORED]:**
   - Isolated Kenny et al.'s original 7×7 CAM extraction into a clean, reusable function without side effects.

### C. Preserved Components (Strict Replication Baseline)
1. **Classifier Backbone Weights (`cnnNORMAL_best.pth`):**
   - The fine-tuned ResNet-34 checkpoint was strictly preserved and kept frozen (83.60% top-1 accuracy on CUB-200).
2. **Input Preprocessing & Image Resolution:**
   - $256\times256$ resize, $224\times224$ center-crop, standard ImageNet normalization.
3. **CUB-200-2011 Test Subset:**
   - Evaluated on the first 500 test images (indices 0 to 499) with deterministic seed ($42$).

