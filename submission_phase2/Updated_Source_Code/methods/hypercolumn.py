"""
methods/hypercolumn.py
Constructs 768-dimensional hypercolumn representations by concatenating
layer3 (256-d) and layer4 (512-d) activations at 14x14 spatial coordinates.
Based on Hypercolumns (Hariharan et al., CVPR 2015).
"""

import torch
import numpy as np

def extract_hypercolumn_at(l3_features, l4_features, u, v):
    """
    Given layer3 tensor [1, 256, 14, 14] and layer4 tensor [1, 512, 7, 7],
    extracts the 768-d concatenated vector at 14x14 coordinate (u, v).
    Coordinate mapping: u_l4 = floor(u / 2), v_l4 = floor(v / 2).
    """
    f3 = l3_features[0, :, u, v].numpy() # [256]
    u4 = min(6, u // 2)
    v4 = min(6, v // 2)
    f4 = l4_features[0, :, u4, v4].numpy() # [512]

    h = np.concatenate([f3, f4], axis=0) # [768]
    return h

def extract_all_hypercolumns(l3_features, l4_features):
    """
    Extracts all 14x14 = 196 hypercolumns of shape [196, 768].
    """
    hypercolumns = np.zeros((14, 14, 768), dtype=np.float32)
    for u in range(14):
        for v in range(14):
            hypercolumns[u, v] = extract_hypercolumn_at(l3_features, l4_features, u, v)
    return hypercolumns

