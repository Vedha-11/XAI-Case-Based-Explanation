"""
methods/matching.py
Implements Nearest-Neighbor Latent Matching:
1. Normalized Cosine Matching (Scale-invariant, proposed)
2. Unnormalized Euclidean Matching (Kenny et al. baseline)
"""

import numpy as np

def match_cosine(query_vec, candidate_matrix):
    """
    Finds index of candidate with maximum cosine similarity:
    S_cos(u, v) = (u . v) / (||u||_2 * ||v||_2)
    """
    # L2 normalize query
    q_norm = np.linalg.norm(query_vec)
    if q_norm > 1e-7:
        q_unit = query_vec / q_norm
    else:
        q_unit = query_vec

    # L2 normalize candidates
    c_norms = np.linalg.norm(candidate_matrix, axis=-1, keepdims=True)
    c_norms[c_norms < 1e-7] = 1.0
    c_units = candidate_matrix / c_norms

    # Dot product similarity
    sims = np.dot(c_units, q_unit)
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]
    return best_idx, best_score

def match_euclidean(query_vec, candidate_matrix):
    """
    Finds index of candidate with minimum Euclidean distance:
    D_E(u, v) = ||u - v||_2
    """
    dists = np.linalg.norm(candidate_matrix - query_vec, axis=-1)
    best_idx = np.argmin(dists)
    best_dist = dists[best_idx]
    return best_idx, best_dist

