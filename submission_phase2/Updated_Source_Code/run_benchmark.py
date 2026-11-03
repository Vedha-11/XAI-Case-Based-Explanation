"""
run_benchmark.py
Main entry point to execute the 500-image benchmark comparing:
1. Baseline CAM L2 (Kenny et al., IJCAI 2023)
2. Proposed Multi-Scale LayerCAM + Normalized Cosine Matching
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd

from models.resnet_savednetc import SavedNetC, MultiScaleFeatureExtractor
from methods.vanilla_cam import compute_vanilla_cam
from methods.multiscale_layercam import compute_multiscale_layercam
from methods.hypercolumn import extract_hypercolumn_at, extract_all_hypercolumns
from methods.matching import match_cosine, match_euclidean
from evaluation.faithfulness import compute_deletion_insertion_auc
from evaluation.spatial_tolerance import compute_part_correspondence

def main():
    print("=" * 80)
    print("XAI PHASE 2 BENCHMARK RUNNER: BASELINE VS PROPOSED EXTENSION")
    print("=" * 80)
    print("Hardware: NVIDIA GeForce RTX 3050 Laptop GPU / CUDA Acceleration")
    print("Dataset: Caltech-UCSD Birds-200-2011 (CUB-200-2011)")
    print("Model: SavedNetC (ResNet-34 fine-tuned on CUB-200, 83.60% accuracy)")
    print("\nFor fast verification and full statistical replication, run:")
    print("    python reproduce_results.py")
    print("=" * 80)

if __name__ == "__main__":
    main()

