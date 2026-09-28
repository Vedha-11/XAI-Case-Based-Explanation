# Phase 1 Replication Report

## Project Overview

This report summarizes the Phase 1 replication of the paper:
**"Advancing Post-Hoc Case-Based Explanation with Feature Highlighting"** (Kenny et al., IJCAI 2023).

- **Official Paper**: [IJCAI 2023 Proceedings](https://www.ijcai.org/proceedings/2023/0048)
- **Official Repository**: [EoinKenny/IJCAI-2023](https://github.com/EoinKenny/IJCAI-2023)

## Reproduced Experimental Results

Phase 1 successfully reproduced two major quantitative evaluations on the CUB-200-2011 dataset using a ResNet34 classifier across 500 evaluation images and superpixel segmentations (10, 20, 30, 40, 50 segments):

### 1. Figure 2(A): CUB-200 Occlusion Evaluation
- **Objective**: Evaluate model performance drop when explaining features are progressively occluded.
- **Methods Evaluated**: Superpixel, CAM, FAM, Random.
- **Result CSVs**:
  - `results/Figure2A_CUB_Occlusion_Replication_500Images.csv`
  - `results/Figure2A_CUB_Occlusion_Summary_500Images.csv`
- **Generated Figure**:
  - `figures/Figure2A_CUB_Occlusion_Replication_500Images.png`

### 2. Figure 2(C): CUB-200 Inclusion Evaluation
- **Objective**: Evaluate model accuracy when only explaining features are progressively included.
- **Methods Evaluated**: Superpixel, CAM, FAM, Random.
- **Result CSVs**:
  - `results/Figure2C_CUB_Inclusion_Replication_500Images.csv`
  - `results/Figure2C_CUB_Inclusion_Summary_500Images.csv`
- **Generated Figure**:
  - `figures/Figure2C_CUB_Inclusion_Replication_500Images.png`

## Execution Summary

All experiments were executed with 500 evaluation images on CUB-200-2011 using PyTorch and scikit-image superpixel segmentation. Results matched the trend and characteristics reported in the official paper.
