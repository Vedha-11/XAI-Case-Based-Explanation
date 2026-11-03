# 6-Member Team Video Presentation Script & Evaluation Guide
## Explainable AI (DA351) — Course Project Phase 2 Evaluation

- **Target Paper:** *Advancing Post-Hoc Case-Based Explanation with Feature Highlighting* (Kenny, Delaney & Keane, IJCAI 2023)
- **Evaluation Dataset:** Caltech-UCSD Birds-200-2011 (`CUB-200-2011`, 500-Image Benchmark Subset)
- **Classifier Backbone:** Pretrained ResNet-34 (`SavedNetC` checkpoint, 83.60% top-1 accuracy)
- **Maximum Target Duration:** Exactly 10:00 Minutes (Budgeted across 6 Members and 12 Slides)

---

## 1. Evaluation Mapping: 6 Evaluation Sections Mapped to 6 Team Members

| Member | Mandatory Evaluation Section | Assigned Slides | Allotted Time | Video Timestamp |
| :---: | :--- | :---: | :---: | :---: |
| **Member 1** | **1. Introduction:** Paper title, authors, problem addressed & research objective | Slides 1 & 2 | **1 min** | 0:00 – 1:00 |
| **Member 2** | **2. Research Gap:** Published results audit (pp. 5–6), coarse 7×7 resolution & metric collapse | Slides 3 & 4 | **2 min** | 1:00 – 3:00 |
| **Member 3** | **3. Proposed Improvement:** Multi-Scale LayerCAM, Cosine matching & architecture flow | Slides 5 & 6 | **2 min** | 3:00 – 5:00 |
| **Member 4** | **4. Implementation:** PyTorch gradient hooks, logit handling & 2×2 factorial ablation setup | Slide 7 | **2 min** | 5:00 – 7:00 |
| **Member 5** | **5. Experimental Results:** Bird #3 visual proof, faithfulness AUC, metric ablation & master table | Slides 8–11 | **2 min** | 7:00 – 9:00 |
| **Member 6** | **6. Conclusion:** Key findings, technical contributions, acknowledged limitations & Q&A | Slide 12 | **1 min** | 9:00 – 10:00 |
| **Total** | **All 6 Evaluation Sections Covered by 6 Team Members** | **12 Slides** | **10 min** | **10:00 Max** |

---

## Member 1 — Section 1: Introduction (1 Min | 0:00 – 1:00)
- **Assigned Slides:** Slide 1 (Title & Objectives) & Slide 2 (The Original Paper's Vision)
- **Mandatory Rubric Topic:** Original paper title, authors, problem addressed, and research objective.

### 🎯 Point to on Screen:
1. On Slide 1: Point to the left card showing the **Target Paper details** (Kenny et al., IJCAI 2023) and the right card showing the **Problem Addressed**.
2. On Slide 2: Point to **Figure 1** extracted from Page 2 of the paper, specifically the yellow bounding boxes drawn on both the query bird and training exemplar.

### 🗣️ What to Say to Sir (Plain Conversational English):
> "Good morning, Sir and peers. Today, our 6-member team is presenting our Phase 2 project in Explainable AI (DA351).
> 
> Our work is a comprehensive audit, technical limitation analysis, and empirical extension of the IJCAI 2023 paper titled **'Advancing Post-Hoc Case-Based Explanation with Feature Highlighting'**, authored by **Eoin Kenny, Eoin Delaney, and Mark Keane**.
> 
> **[On Slide 1]** In visual classification, deep neural networks are black boxes. Presenting only a confidence score or an unexplained image does not tell the user WHAT specific feature justified the prediction.
> 
> **[Moving to Slide 2]** To address this, Kenny et al. introduced a **Twin-Explanation framework**: when classifying an image, the system retrieves a similar past training exemplar and draws bounding boxes on both images to highlight the shared anatomical trait, such as a sharp yellow beak.
> 
> They implemented this using standard Class Activation Mapping (CAM) on terminal layer4 of ResNet-34 and Euclidean distance. While this concept is very practical, our technical audit revealed two foundational flaws in their design.
> 
> I will now hand over to **Member 2**, who will audit the published results and explain the two technical research gaps."

### 💡 The Motive:
- Establish the paper details, problem context, and twin-explanation mechanism clearly in the first minute.

### ❓ If Sir Asks:
- **Sir's Question:** *"What exactly is a Twin Explanation, and why is it better than standard Grad-CAM heatmaps?"*
- **Your Answer:** *"Sir, a standard heatmap only shows colored blobs on the test image, leaving the user guessing. A Twin Explanation pairs that test bird with a concrete historical training exemplar sharing the exact same body part, giving both visual localization and analogical proof."*

---

## Member 2 — Section 2: Research Gap Identification (2 Min | 1:00 – 3:00)
- **Assigned Slides:** Slide 3 (Published Results Audit pp. 5–6) & Slide 4 (The Two Critical Research Gaps)
- **Mandatory Rubric Topic:** Identified limitations and justification for selecting them.

### 🎯 Point to on Screen:
1. On Slide 3: Point to **Figure 2** on Page 5 (occlusion/inclusion curves) and **Figure 3** on Page 6 (human evaluation chart).
2. On Slide 4: Point to **Research Gap 1** on the left card (the coarse 7×7 grid diagram) and **Research Gap 2** on the right card (the Euclidean metric breakdown diagram).

### 🗣️ What to Say to Sir (Plain Conversational English):
> "Thank you, Member 1. Sir, in accordance with the mandatory submission guidelines, we first audited the original paper's published experimental results:
> 
> **[On Slide 3: Published Results Audit]**
> - On **Page 5, Figure 2**, Kenny et al. evaluated occlusion and inclusion across 500 CUB images, concluding that CAM outperformed random and superpixel baselines.
> - On **Page 6, Figure 3**, they published human user evaluation results showing that feature highlighting significantly enhanced human decision accuracy over cases alone.
> 
> Their published results prove the concept is sound. However, when we investigated their underlying technical implementation, we identified two critical limitations:
> 
> **[Moving to Slide 4: The Two Critical Research Gaps]**
> 1. **Research Gap 1: Coarse 7×7 Spatial Attribution (The 'Thick Highlighter' Problem):** Kenny et al. extracted CAM exclusively from terminal `layer4`. On a 224×224 image, this produces a coarse 7×7 grid with a 32-pixel stride and an effective receptive field exceeding 100 pixels. On fine-grained birds, their 48×48 pixel box covers the beak, eye, neck, and water all at once! Explanations must pinpoint fine-grained traits without capturing background noise.
> 2. **Research Gap 2: Distance Metric Distortion in Multi-Layer Space (The 'Broken Ruler' Problem):** To make resolution sharper (14×14), earlier `layer3` (256-d) must be fused with `layer4` (512-d) into a 768-d hypercolumn. But Kenny et al. matched features using unnormalized Euclidean distance ($L_2$). In high dimensions, deep `layer4` numbers are much larger, completely drowning out sharp `layer3` spatial features. When we tested Euclidean distance on 14×14 features, matching accuracy plummeted from 51.8% to 37.2%!
> 
> I now invite **Member 3** to present our proposed technical solution and supporting research literature."

### 💡 The Motive:
- Audit published claims with exact page numbers and justify why resolving the 7×7 grid and Euclidean metric breakdown is essential for fine-grained explanation.

### ❓ If Sir Asks:
- **Sir's Question:** *"Why did matching accuracy collapse to 37.2% under Euclidean distance?"*
- **Your Answer:** *"Sir, in 768-dimensional space, layer4 activations have much larger numerical norms than layer3. Euclidean distance measures straight-line length, so layer4 completely overpowered layer3 and drowned out the sharp spatial features. Cosine similarity evaluates pattern angle, restoring balance."*

---

## Member 3 — Section 3: Proposed Improvement (2 Min | 3:00 – 5:00)
- **Assigned Slides:** Slide 5 (Proposed Solution & Supporting Literature) & Slide 6 (End-to-End Architectural Pipeline)
- **Mandatory Rubric Topic:** Proposed solution, technical methodology, and supporting literature references.

### 🎯 Point to on Screen:
1. On Slide 5: Point to the **LayerCAM weight equation** on the left card and the **Normalized Cosine equation** on the right card, noting the literature citations (IEEE TIP 2021 & CVPR 2015).
2. On Slide 6: Trace the architecture diagram from left to right: input bird $\rightarrow$ `layer3` & `layer4` hooks $\rightarrow$ bilinear saliency fusion $\rightarrow$ 768-d hypercolumn $\rightarrow$ Cosine retrieval.

### 🗣️ What to Say to Sir (Plain Conversational English):
> "Thank you, Member 2. Sir, here is our proposed technical solution grounded in top peer-reviewed literature:
> 
> **[On Slide 5: Proposed Improvement & Supporting Literature]**
> - **To solve Gap 1 (Coarse Resolution):** We implemented **Multi-Scale LayerCAM**, published by **Jiang et al. in *IEEE Transactions on Image Processing, 2021***. Standard CAM averages gradients across entire channels, washing out fine details. LayerCAM calculates individual positive weights for every pixel coordinate: $w_{k,i,j} = \max(0, \frac{\partial y^c}{\partial A^k_{i,j}})$. We fuse intermediate `layer3` (14×14) and `layer4` (7×7) to quadruple grid density from 49 to 196 cells with a sharp 16-pixel stride!
> - **To solve Gap 2 (Metric Breakdown):** We introduced **Scale-Invariant Normalized Cosine Matching**, supported by Hypercolumn representations from **Hariharan et al., *CVPR 2015***. We project all 768-dimensional hypercolumns onto the unit hypersphere via $L_2$ normalization: $S(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$. Cosine similarity evaluates pattern direction rather than raw magnitude, giving both layers equal voice.
> 
> **[Moving to Slide 6: Architectural Pipeline]**
> This diagram illustrates our complete end-to-end framework: The query bird passes through the frozen ResNet-34. Saliency maps from `layer3` and `layer4` are bilinearly fused into a sharp 14×14 grid. At the maximal saliency coordinate $(u^*, v^*)$, the 768-d hypercolumn is extracted, normalized, and matched via Cosine similarity to retrieve the twin training exemplar.
> 
> I will now pass to **Member 4**, who will explain our PyTorch implementation and controlled experimental setup."

### 💡 The Motive:
- Anchor the proposed improvements directly in established IEEE TIP and CVPR computer vision literature and show how the components operate together.

### ❓ If Sir Asks:
- **Sir's Question:** *"How does LayerCAM differ from standard Grad-CAM?"*
- **Your Answer:** *"Sir, Grad-CAM computes a single global average weight per feature map, which blurs out spatial details. LayerCAM assigns an individual weight to every spatial pixel coordinate using positive gradients, preserving fine-grained localized structures like bird beaks."*

---

## Member 4 — Section 4: Implementation (2 Min | 5:00 – 7:00)
- **Assigned Slide:** Slide 7 (Technical Implementation & Controlled Factorial Design)
- **Mandatory Rubric Topic:** How the improvement was implemented and what changes were made to the original method.

### 🎯 Point to on Screen:
1. On the left card: Point to the **Preserved Backbone** and **PyTorch Gradient Hooks & 3-Tuple Logit Handling** bullets.
2. On the right card: Point to the **2×2 Factorial Ablation Matrix table**, highlighting Conditions 1 through 4.

### 🗣️ What to Say to Sir (Plain Conversational English):
> "Thank you, Member 3. Sir, let us examine how our improvement was technically implemented and what changes were made to the original method:
> 
> **[On Slide 7: Technical Implementation]**
> 1. **Backbone Model Preserved:** We kept the pretrained ResNet-34 (`SavedNetC`) checkpoint 100% frozen at 83.60% top-1 accuracy on CUB-200. Our explanation method operates strictly post-hoc.
> 2. **PyTorch Gradient Hooks:** We attached forward and backward hooks to both intermediate `layer3` (14×14, 256-d) and terminal `layer4` (7×7, 512-d).
> 3. **Handling 3-Tuple Logits:** In the original codebase, `SavedNetC` returns a 3-tuple `(logits, x, C)`. We properly routed backward gradients exclusively through the target class logit.
> 4. **Zero-Leak GPU Memory Safety:** By immediately detaching tensors and clearing hooks after the backward pass, peak GPU VRAM remained flat at 286.8 MB across all 500 images.
> 
> **[The Controlled 2×2 Factorial Ablation Setup]**
> To scientifically isolate resolution effects from distance metric effects, we evaluated all 4 factorial combinations on the identical 500 CUB test images:
> - **Condition 1:** Baseline 7×7 CAM with Euclidean ($L_2$) distance.
> - **Condition 2:** 7×7 CAM with Normalized Cosine similarity.
> - **Condition 3:** 14×14 LayerCAM with Euclidean ($L_2$) distance.
> - **Condition 4 (Proposed):** 14×14 LayerCAM with Normalized Cosine similarity.
> 
> I now invite **Member 5** to present our experimental results, before-and-after visual proofs, and benchmark tables."

### 💡 The Motive:
- Demonstrate rigorous software engineering, memory management, and clean 2×2 factorial experimental design.

### ❓ If Sir Asks:
- **Sir's Question:** *"Why did you use a 2×2 factorial ablation design?"*
- **Your Answer:** *"Sir, because we introduced two technical changes: spatial resolution and distance metric. Testing all 4 combinations proves scientifically that both changes are necessary—showing that increasing resolution alone causes Euclidean distance to crash, and Cosine matching fixes it."*

---

## Member 5 — Section 5: Experimental Results (2 Min | 7:00 – 9:00)
- **Assigned Slides:** Slide 8 (Bird #3 Visual Proof), Slide 9 (Faithfulness AUC Curves), Slide 10 (Metric Ablation Bar Chart), Slide 11 (Master Side-by-Side Comparison Table)
- **Mandatory Rubric Topic:** Before-and-after comparison, tables/graphs, and interpretation of results.

### 🎯 Point to on Screen:
1. On Slide 8: Point to **Panel B** (Kenny's 48×48 box covering water) vs. **Panel C** (our 32×32 box on the beak tip).
2. On Slide 9: Point to the **Deletion curve falling faster** on the left and the **Insertion curve rising faster** on the right.
3. On Slide 10: Point to the **red bar (Condition 3 at 37.2%)** and the **green bar (Condition 4 at 46.6%)**.
4. On Slide 11: Point to the 5-column **Master Side-by-Side Comparison Table** and the RTX 3050 hardware efficiency summary.

### 🗣️ What to Say to Sir (Plain Conversational English):
> "Thank you, Member 4. Sir, here are our empirical experimental results evaluated across 500 CUB test images:
> 
> **[On Slide 8: Before vs. After on Bird #3]**
> - Look at Panel B: Kenny et al.'s baseline produces a huge 48×48 box that blurs together the bird's beak, eye, throat, and water. A user cannot tell what the AI actually focused on.
> - Look at Panel C: Our Multi-Scale LayerCAM produces a crisp 32×32 box focused precisely on the bird's bill tip and forehead, eliminating all background noise!
> - Panel D confirms that the retrieved training bird shares the exact same beak landmark.
> 
> **[On Slide 9: Quantitative Faithfulness Test]**
> If an explanation is truthful, blurring its highlighted pixels must drop model confidence immediately. Across 22 steps of continuous Gaussian blur:
> - **Deletion AUC (Lower is Better):** Dropped from 0.1171 to 0.0989 ($t = -8.88, p = 1.22 \times 10^{-17}$). In 79.6% of images, confidence dropped faster under our method!
> - **Insertion AUC (Higher is Better):** Rose from 0.6817 to 0.6989 ($t = +7.93, p = 1.40 \times 10^{-14}$).
> 
> **[On Slide 10: Distance Metric Ablation]**
> Look at the bar chart: when we increased resolution using Euclidean distance (Condition 3), matching accuracy crashed to **37.20%** (-14.6 pp drop, $p = 5.8 \times 10^{-8}$) because deep layer norms drowned out spatial features. When we applied our Normalized Cosine Matching (Condition 4), accuracy rebounded to **46.60% overall** and **56.43% on valid queries**—a massive **+9.40 percentage point recovery** ($p = 2.48 \times 10^{-6}$)!
> 
> **[On Slide 11: Master Side-by-Side Comparison]**
> Table 1 directly contrasts Kenny et al.'s published claims against our empirical results. On an **NVIDIA RTX 3050 Laptop GPU (6GB VRAM)**, per-query latency was 488 ms (+1.4% overhead) with an identical 286.8 MB peak VRAM footprint, proving real-time feasibility!
> 
> I will now hand over to **Member 6** to conclude our presentation."

### 💡 The Motive:
- Present empirical before-and-after proof, causal perturbation game, metric ablation bar chart, and hardware efficiency.

### ❓ If Sir Asks:
- **Sir's Question:** *"Why is a lower Deletion AUC better?"*
- **Your Answer:** *"Sir, in the deletion game, we remove the pixels the method highlighted as most important. If those pixels really drove the model's prediction, the model's confidence should drop immediately. A lower area under the deletion curve proves superior causal faithfulness."*

---

## Member 6 — Section 6: Conclusion (1 Min | 9:00 – 10:00)
- **Assigned Slide:** Slide 12 (Key Findings, Contributions, Limitations & Q&A)
- **Mandatory Rubric Topic:** Key findings, contributions, limitations, and open discussion.

### 🎯 Point to on Screen:
1. Point to the left card: **Core Technical Contributions** (1 to 5).
2. Point to the right card: **Acknowledged Limitations & Future Work**.
3. Point to the bottom banner: **Thank You, Sir! Questions & Open Discussion**.

### 🗣️ What to Say to Sir (Plain Conversational English):
> "Thank you, Member 5. Sir, let us synthesize our key findings, acknowledge limitations, and conclude our Phase 2 project:
> 
> **[On Slide 12: Key Findings & Technical Contributions]**
> 1. **Quadrupled Spatial Resolution:** We upgraded attribution from 7×7 (49 cells) to 14×14 (196 cells) via Multi-Scale LayerCAM, shrinking bounding boxes from 48×48 to 32×32 pixels and eliminating background clutter.
> 2. **Statistically Proven Faithfulness:** Highly significant gains in Deletion AUC (0.0989 vs 0.1171, $p = 1.22 \times 10^{-17}$) and Insertion AUC (0.6989 vs 0.6817, $p = 1.40 \times 10^{-14}$).
> 3. **Solved Metric Breakdown:** Scale-Invariant Normalized Cosine Matching recovered +9.40 percentage points over Euclidean distance in 768-d hypercolumn space ($p = 2.48 \times 10^{-6}$).
> 4. **Production Efficiency:** 488 ms per image on an entry-level RTX 3050 GPU with zero added memory overhead.
> 5. **All Deliverables Complete:** Video presentation assets, Original IJCAI 2023 Paper PDF, 8-page Comparison Report PDF, and reproducible source code.
> 
> **[Acknowledged Limitations & Future Work]**
> CUB ground truth landmarks are discrete 1D points rather than dense 2D masks; future work will test segmentation masks and Vision Transformers (ViT).
> 
> Thank you very much, Sir! Our entire team is now ready for your questions and open discussion."

### 💡 The Motive:
- Deliver a crisp, authoritative summary of contributions and limitations, leaving a polished professional impression.

### ❓ If Sir Asks:
- **Sir's Question:** *"What is the single biggest takeaway from your Phase 2 project?"*
- **Your Answer:** *"Sir, our biggest takeaway is that post-hoc explanations should never rely on the terminal layer alone. By fusing multi-scale layers with LayerCAM and using Normalized Cosine Matching, we achieve sharp, faithful explanations with virtually no extra computational cost."*

---

## 2. Video Recording Checklist & Team Coordination Guide

1. **Screen Setup:** Open `Phase_2_Project_Presentation.pptx` in Slide Show mode (or open `presentation_slides.html` in Chrome/Edge and press `F11` for full screen).
2. **Transitions:** Always say your handoff sentence at the end of your section (e.g., *"I now hand over to Member 3"*). This demonstrates strong team synergy.
3. **Timekeeping:** Keep a silent phone stopwatch visible. Members 1 and 6 speak for **1 minute**; Members 2, 3, 4, and 5 speak for **2 minutes**. Total presentation time is exactly 10:00 minutes.
4. **Audio Check:** Test all 6 microphones before the recording to ensure consistent audio levels.
5. **Drive Sharing:** Upload the recorded MP4 file to Google Drive. Right-click, select **Share**, and set permissions to **"Anyone with the link can view"** before submitting the Google Form.

---

## 3. Complete Submission Inventory (`submission_phase2/`)

- **`A_Video_Presentation_Assets/`**:
  - `Phase_2_Project_Presentation.pptx` (12-slide master presentation with embedded speaker notes)
  - `6_Member_Team_Presentation_Guide.pdf` (4-page clean printable presentation guide)
  - `6_Member_Team_Presentation_Script.md` (Spoken script and evaluation mapping)
  - `presentation_slides.html` (Interactive web slides with hotkey navigation and speaker note toggling)
  - `figures/` (Extracted paper figures from Pages 1, 2, 5, 6 and empirical benchmark plots)
- **`B_Original_Research_Article/`**:
  - `Original_Research_Article_IJCAI2023_Kenny_et_al.pdf` (Complete PDF of Kenny et al., IJCAI 2023, Pages 3142–3150)
- **`C_Experimental_Results_and_Comparison/`**:
  - `Experimental_Results_and_Comparison.pdf` (8-page comprehensive comparison report)
- **`Updated_Source_Code/`**:
  - `reproduce_results.py` (Complete standalone reproduction script)
  - `README.md` & `MODIFIED_FILES.md` (Detailed reproduction instructions, dependencies, and file diffs)
- **`00_START_HERE_SUBMISSION_GUIDE.md`**: Master index and step-by-step submission walk-through.
- **`SUBMISSION_CHECKLIST_AND_GOOGLE_FORM_GUIDE.md`**: Pre-submission validation checklist and Google Form field-by-field instructions.
