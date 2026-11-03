"""
evaluation/spatial_tolerance.py
Evaluates Part Hit Rate across:
1. Native Grid-Tied Rule (1.5x stride: 48x48 px for 7x7 vs 32x32 px for 14x14)
2. Standardized Matched Chebyshev Box (R in {16, 24, 32} px)
3. Standardized Matched Euclidean Circle (R in {16, 24, 32} px)
"""

import numpy as np

def evaluate_keypoint_hit(center_x, center_y, keypoints, radius_px, geometry='box'):
    """
    Checks if any visible ground-truth keypoint falls within the tolerance window.
    keypoints: list of (part_id, x, y, visible)
    geometry: 'box' (Chebyshev L_infty) or 'circle' (Euclidean L_2)
    """
    valid_parts = []
    for pid, kx, ky, vis in keypoints:
        if vis == 0:
            continue
        
        dx = abs(kx - center_x)
        dy = abs(ky - center_y)
        
        if geometry == 'box':
            if dx <= radius_px and dy <= radius_px:
                valid_parts.append(pid)
        elif geometry == 'circle':
            dist = np.sqrt(dx**2 + dy**2)
            if dist <= radius_px:
                valid_parts.append(pid)

    return valid_parts

def compute_part_correspondence(q_center, e_center, q_keypoints, e_keypoints, radius_px, geometry='box'):
    """
    Determines if query and exemplar highlight the same shared anatomical part:
    Returns (has_valid_query_part, is_hit, shared_parts)
    """
    q_parts = evaluate_keypoint_hit(q_center[0], q_center[1], q_keypoints, radius_px, geometry)
    e_parts = evaluate_keypoint_hit(e_center[0], e_center[1], e_keypoints, radius_px, geometry)
    
    has_valid_q_part = len(q_parts) > 0
    shared = set(q_parts).intersection(set(e_parts))
    is_hit = len(shared) > 0
    
    return has_valid_q_part, is_hit, list(shared)

