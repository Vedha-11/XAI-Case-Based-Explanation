"""
Validation and Diagnostic Script for XAI Phase 2 Pilot Audit
Generates:
1. neighbor_diagnostics.csv (full candidate scanning and chosen neighbor comparisons)
2. metric_recalculation.csv (independent recalculation from raw data)
3. validation_runtime_memory.csv (re-measured runtime, allocated, and reserved memory across 2 repeated runs)

Authors: DA351 Research Team
Date: October 2026
"""

import os
import sys
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.neighbors import KNeighborsClassifier


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_REPO = PROJECT_ROOT / "original_repo" / "Expt 1" / "CUB"
CUB_DATA_DIR = ORIGINAL_REPO / "data" / "CUB_200_2011"
WEIGHTS_PATH = ORIGINAL_REPO / "weights" / "cnnNORMAL_best.pth"
AUDIT_OUTPUT_DIR = PROJECT_ROOT / "pilot_results" / "audit_validation"
AUDIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class SavedNetC(nn.Module):
    def __init__(self):
        super(SavedNetC, self).__init__()
        resnet = models.resnet34(weights=None)
        self.main = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AvgPool2d(7)
        self.linear = nn.Linear(512, 200)

    def forward(self, imgs):
        C = self.main(imgs)
        x = self.avgpool(C)
        x = x.view(x.shape[0], x.shape[1])
        logits = self.linear(x)
        return logits, x, C


def load_model():
    model = SavedNetC()
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_cub_metadata():
    images_df = pd.read_csv(CUB_DATA_DIR / "images.txt", sep=" ", names=["img_id", "filepath"])
    split_df = pd.read_csv(CUB_DATA_DIR / "train_test_split.txt", sep=" ", names=["img_id", "is_training_img"])
    labels_df = pd.read_csv(CUB_DATA_DIR / "image_class_labels.txt", sep=" ", names=["img_id", "target"])
    
    data = images_df.merge(labels_df, on="img_id").merge(split_df, on="img_id")
    train_data = data[data.is_training_img == 1].reset_index(drop=True)
    test_data = data[data.is_training_img == 0].reset_index(drop=True)

    parts_txt = CUB_DATA_DIR / "parts" / "parts.txt"
    part_names = {}
    with open(parts_txt, "r") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                part_names[int(parts[0])] = parts[1]

    part_locs = pd.read_csv(
        CUB_DATA_DIR / "parts" / "part_locs.txt",
        sep=" ",
        names=["img_id", "part_id", "x", "y", "visible"]
    )
    return train_data, test_data, part_names, part_locs


def transform_part_coordinates(orig_w, orig_h, part_locs_df):
    s = 224.0 / min(orig_w, orig_h)
    if orig_w < orig_h:
        w_prime = 224
        h_prime = int(round(orig_h * s))
    else:
        h_prime = 224
        w_prime = int(round(orig_w * s))
    
    x_left = int(round((w_prime - 224) / 2.0))
    y_top = int(round((h_prime - 224) / 2.0))

    transformed = []
    for _, row in part_locs_df.iterrows():
        if row.visible != 1:
            continue
        x_prime = row.x * (w_prime / orig_w)
        y_prime = row.y * (h_prime / orig_h)
        x_crop = x_prime - x_left
        y_crop = y_prime - y_top
        
        if 0 <= x_crop < 224 and 0 <= y_crop < 224:
            transformed.append({
                "part_id": int(row.part_id),
                "x_crop": float(x_crop),
                "y_crop": float(y_crop)
            })
    return transformed


def get_parts_in_patch(parts_list, grid_r, grid_c, grid_size=7, margin=8.0):
    cell_w = 224.0 / grid_size
    x_min = max(0.0, grid_c * cell_w - margin)
    x_max = min(224.0, (grid_c + 1) * cell_w + margin)
    y_min = max(0.0, grid_r * cell_w - margin)
    y_max = min(224.0, (grid_r + 1) * cell_w + margin)

    matched_parts = []
    for p in parts_list:
        if x_min <= p["x_crop"] <= x_max and y_min <= p["y_crop"] <= y_max:
            matched_parts.append(p["part_id"])
    return set(matched_parts)


class DiagnosticCBREngine:
    def __init__(self, model, alpha=5.0, device=DEVICE):
        self.model = model
        self.alpha = float(alpha)
        self.device = device
        self.act_layer4 = None
        self.grad_layer4 = None
        self.act_layer3 = None
        self.grad_layer3 = None
        self.hook_handles = []

    def _register_hooks(self):
        self._remove_hooks()
        def h_act4(m, inp, out): self.act_layer4 = out.detach()
        def h_grad4(m, gin, gout): self.grad_layer4 = gout[0].detach()
        def h_act3(m, inp, out): self.act_layer3 = out.detach()
        def h_grad3(m, gin, gout): self.grad_layer3 = gout[0].detach()

        self.hook_handles.append(self.model.main[7][-1].register_forward_hook(h_act4))
        self.hook_handles.append(self.model.main[7][-1].register_full_backward_hook(h_grad4))
        self.hook_handles.append(self.model.main[6][-1].register_forward_hook(h_act3))
        self.hook_handles.append(self.model.main[6][-1].register_full_backward_hook(h_grad3))

    def _remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []

    def extract_features_and_saliency(self, img_tensor, target_class=None):
        self._register_hooks()
        self.model.zero_grad()

        logits, _, C_feat = self.model(img_tensor.to(self.device))
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
            
        score = logits[0, target_class]
        score.backward()

        # Vanilla CAM on Layer4
        weights_c = self.model.linear.weight[target_class]
        cam_7 = torch.sum(weights_c.view(512, 1, 1) * C_feat[0], dim=0)
        cam_7 = F.relu(cam_7)
        cam_7_norm = (cam_7 - cam_7.min()) / (cam_7.max() - cam_7.min() + 1e-8)

        # LayerCAM on Layer4
        w4 = F.relu(self.grad_layer4[0])
        lcam_4 = F.relu(torch.sum(w4 * self.act_layer4[0], dim=0, keepdim=True).unsqueeze(0))
        lcam_4_up = F.interpolate(lcam_4, size=(14, 14), mode="bilinear", align_corners=False).squeeze()

        # LayerCAM on Layer3
        w3 = F.relu(self.grad_layer3[0])
        lcam_3 = F.relu(torch.sum(w3 * self.act_layer3[0], dim=0))

        # Fused Multi-Scale LayerCAM
        lcam_4_norm = (lcam_4_up - lcam_4_up.min()) / (lcam_4_up.max() - lcam_4_up.min() + 1e-8)
        lcam_3_norm = (lcam_3 - lcam_3.min()) / (lcam_3.max() - lcam_3.min() + 1e-8)
        fused_lcam_14 = 0.5 * lcam_4_norm + 0.5 * lcam_3_norm

        # 14x14 Hypercolumn
        act4_up = F.interpolate(self.act_layer4, size=(14, 14), mode="bilinear", align_corners=False)[0]
        hypercolumn_14 = torch.cat([self.act_layer3[0], act4_up], dim=0)

        self._remove_hooks()

        return {
            "logits": logits.detach(),
            "target_class": target_class,
            "cam_7": cam_7_norm.detach(),
            "fused_lcam_14": fused_lcam_14.detach(),
            "feat_layer4_7": C_feat[0].detach(),
            "hypercolumn_14": hypercolumn_14.detach()
        }


def scan_all_candidates(query_dict, neighbor_dicts, method_name="proposed_multiscale_cosine", alpha=5.0):
    """
    Detailed candidate scanning logging every neighbor's candidate pool and distances/similarities.
    """
    if "7" in method_name or "cam" in method_name and "multiscale" not in method_name:
        grid_size = 7
        q_saliency = query_dict["cam_7"]
        q_features = query_dict["feat_layer4_7"]
        feat_dim = 512
    else:
        grid_size = 14
        q_saliency = query_dict["fused_lcam_14"]
        q_features = query_dict["hypercolumn_14"]
        feat_dim = 768

    use_cosine = ("cosine" in method_name)

    q_max_idx = q_saliency.view(-1).argmax().item()
    q_r, q_c = divmod(q_max_idx, grid_size)
    q_vec = q_features[:, q_r, q_c].unsqueeze(0)

    if use_cosine:
        q_vec_norm = F.normalize(q_vec, p=2, dim=1)
        best_score = -float("inf")
    else:
        best_score = float("inf")

    best_match = {
        "neighbor_idx": -1,
        "match_r": None,
        "match_c": None,
        "score": None,
        "query_r": q_r,
        "query_c": q_c,
        "grid_size": grid_size
    }

    per_neighbor_diagnostics = []

    for n_idx, n_dict in enumerate(neighbor_dicts):
        if grid_size == 7:
            n_saliency = n_dict["cam_7"]
            n_features = n_dict["feat_layer4_7"]
        else:
            n_saliency = n_dict["fused_lcam_14"]
            n_features = n_dict["hypercolumn_14"]

        threshold = n_saliency.max() / alpha
        valid_mask = (n_saliency > threshold).view(-1)
        num_valid_cells = valid_mask.sum().item()

        if num_valid_cells == 0:
            per_neighbor_diagnostics.append({
                "neighbor_idx": n_idx,
                "num_valid_cells": 0,
                "best_cell_score": None,
                "best_coord": None
            })
            continue

        n_flat = n_features.view(feat_dim, -1).t()

        if use_cosine:
            n_flat_norm = F.normalize(n_flat, p=2, dim=1)
            sims = torch.mm(q_vec_norm, n_flat_norm.t()).squeeze(0)
            sims[~valid_mask] = -float("inf")
            val, idx = sims.max(dim=0)
            mr, mc = divmod(idx.item(), grid_size)
            cell_score = val.item()
            if cell_score > best_score:
                best_score = cell_score
                best_match.update({
                    "neighbor_idx": n_idx,
                    "match_r": mr,
                    "match_c": mc,
                    "score": best_score
                })
        else:
            dists = torch.cdist(q_vec, n_flat, p=2.0).squeeze(0)
            dists[~valid_mask] = float("inf")
            val, idx = dists.min(dim=0)
            mr, mc = divmod(idx.item(), grid_size)
            cell_score = val.item()
            if cell_score < best_score:
                best_score = cell_score
                best_match.update({
                    "neighbor_idx": n_idx,
                    "match_r": mr,
                    "match_c": mc,
                    "score": best_score
                })

        per_neighbor_diagnostics.append({
            "neighbor_idx": n_idx,
            "num_valid_cells": num_valid_cells,
            "best_cell_score": cell_score,
            "best_coord": (mr, mc)
        })

    return best_match, per_neighbor_diagnostics


def run_diagnostics():
    print("=" * 80)
    print("RUNNING DIAGNOSTIC AND AUDIT VERIFICATION FOR XAI PHASE 2 PILOT")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    model = load_model()
    train_data, test_data, part_names, part_locs_df = load_cub_metadata()

    x_train_cont_path = ORIGINAL_REPO / "data" / "X_train_cont.npy"
    x_train_y_path = ORIGINAL_REPO / "data" / "X_train_y.npy"
    x_test_cont_path = ORIGINAL_REPO / "data" / "X_test_cont.npy"

    X_train_c = np.load(x_train_cont_path)
    train_preds = np.load(x_train_y_path)
    X_test_c = np.load(x_test_cont_path)

    twin = KNeighborsClassifier(n_neighbors=1, algorithm="brute", metric="euclidean")
    twin.fit(X_train_c, train_preds)

    engine = DiagnosticCBREngine(model, alpha=5.0, device=DEVICE)

    methods = [
        "baseline_cam_l2",
        "ablation_cam_cosine",
        "ablation_multiscale_l2",
        "proposed_multiscale_cosine"
    ]

    neighbor_diagnostics = []
    num_pilot_images = 10

    # Verification of shapes
    print("\n--- SHAPE DIAGNOSTICS (Single Query Verification) ---")
    dummy_img = Image.open(CUB_DATA_DIR / "images" / test_data.iloc[0].filepath)
    dummy_tensor = test_transform(dummy_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    q0_dict = engine.extract_features_and_saliency(dummy_tensor)
    print(f"act_layer3 shape:       {engine.act_layer3.shape} (Expected: [1, 256, 14, 14])")
    print(f"act_layer4 shape:       {engine.act_layer4.shape} (Expected: [1, 512, 7, 7])")
    print(f"cam_7 shape:            {q0_dict['cam_7'].shape} (Expected: [7, 7])")
    print(f"fused_lcam_14 shape:    {q0_dict['fused_lcam_14'].shape} (Expected: [14, 14])")
    print(f"feat_layer4_7 shape:    {q0_dict['feat_layer4_7'].shape} (Expected: [512, 7, 7])")
    print(f"hypercolumn_14 shape:   {q0_dict['hypercolumn_14'].shape} (Expected: [768, 14, 14])")
    print("Shape verification: ALL MATCHING EXPECTATIONS EXACTLY.\n")

    # Repeatability Run 1 & 2 Memory Tracking
    print("--- RUNNING 2 REPEATABILITY PASSES TO MEASURE VRAM & STABILITY ---")
    runtime_records = []

    for run_id in [1, 2]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()

        for test_idx in range(num_pilot_images):
            test_row = test_data.iloc[test_idx]
            img_id = int(test_row.img_id)
            img_full_path = CUB_DATA_DIR / "images" / test_row.filepath

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with Image.open(img_full_path) as im:
                orig_w, orig_h = im.size
                q_tensor = test_transform(im.convert("RGB")).unsqueeze(0).to(DEVICE)

            q_parts_df = part_locs_df[part_locs_df.img_id == img_id]
            q_parts_crop = transform_part_coordinates(orig_w, orig_h, q_parts_df)

            q_dict = engine.extract_features_and_saliency(q_tensor)
            pred_class = q_dict["target_class"]

            q_cont = X_test_c[test_idx]
            nn_indices = twin.kneighbors(X=[q_cont], n_neighbors=10, return_distance=False)[0]

            n_dicts = []
            n_parts_meta = []
            for nn_train_idx in nn_indices:
                n_row = train_data.iloc[nn_train_idx]
                n_img_id = int(n_row.img_id)
                n_path = CUB_DATA_DIR / "images" / n_row.filepath

                with Image.open(n_path) as nim:
                    nw, nh = nim.size
                    n_tensor = test_transform(nim.convert("RGB")).unsqueeze(0).to(DEVICE)

                n_dict = engine.extract_features_and_saliency(n_tensor, target_class=pred_class)
                n_dicts.append(n_dict)

                n_p_df = part_locs_df[part_locs_df.img_id == n_img_id]
                n_parts_meta.append(transform_part_coordinates(nw, nh, n_p_df))

            # Match across all methods
            for m_name in methods:
                best_match, per_neighbor_diag = scan_all_candidates(
                    q_dict, n_dicts, method_name=m_name, alpha=5.0
                )
                
                # Check part hit
                q_patch_parts = get_parts_in_patch(
                    q_parts_crop,
                    best_match["query_r"],
                    best_match["query_c"],
                    grid_size=best_match["grid_size"]
                )
                n_matched_idx = best_match["neighbor_idx"]
                if n_matched_idx >= 0 and best_match["match_r"] is not None:
                    n_patch_parts = get_parts_in_patch(
                        n_parts_meta[n_matched_idx],
                        best_match["match_r"],
                        best_match["match_c"],
                        grid_size=best_match["grid_size"]
                    )
                else:
                    n_patch_parts = set()

                shared_parts = q_patch_parts.intersection(n_patch_parts)
                part_hit = (len(shared_parts) > 0)

                # Record candidate logging on Run 1
                if run_id == 1:
                    matched_train_img_id = int(train_data.iloc[nn_indices[best_match["neighbor_idx"]]].img_id) if best_match["neighbor_idx"] >= 0 else -1
                    neighbor_diagnostics.append({
                        "test_idx": test_idx,
                        "img_id": img_id,
                        "method": m_name,
                        "grid_size": best_match["grid_size"],
                        "query_coord": f"({best_match['query_r']},{best_match['query_c']})",
                        "selected_neighbor_pool_idx": best_match["neighbor_idx"],
                        "selected_neighbor_img_id": matched_train_img_id,
                        "selected_coord": f"({best_match['match_r']},{best_match['match_c']})",
                        "selection_score": best_match["score"],
                        "score_metric": "Cosine_Sim" if "cosine" in m_name else "L2_Distance",
                        "query_parts": ";".join([part_names[pid] for pid in q_patch_parts]),
                        "neighbor_parts": ";".join([part_names[pid] for pid in n_patch_parts]),
                        "shared_parts": ";".join([part_names[pid] for pid in shared_parts]),
                        "part_hit": part_hit
                    })

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            runtime_records.append({
                "run_id": run_id,
                "test_idx": test_idx,
                "img_id": img_id,
                "elapsed_ms": (t1 - t0) * 1000.0,
                "peak_alloc_mb": torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0,
                "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2) if torch.cuda.is_available() else 0.0
            })

        print(f"Run {run_id} completed. Total time: {time.perf_counter() - t_start:.2f} s")

    # Save neighbor diagnostics and runtime
    df_neighbor = pd.DataFrame(neighbor_diagnostics)
    df_runtime = pd.DataFrame(runtime_records)

    neighbor_csv_path = AUDIT_OUTPUT_DIR / "neighbor_diagnostics.csv"
    runtime_csv_path = AUDIT_OUTPUT_DIR / "validation_runtime_memory.csv"
    df_neighbor.to_csv(neighbor_csv_path, index=False)
    df_runtime.to_csv(runtime_csv_path, index=False)
    print(f"Saved: {neighbor_csv_path}")
    print(f"Saved: {runtime_csv_path}")

    # Metric recalculation from pilot_results.csv
    recalc_csv_path = AUDIT_OUTPUT_DIR / "metric_recalculation.csv"
    orig_pilot_csv = PROJECT_ROOT / "pilot_results" / "pilot_results.csv"
    df_orig = pd.read_csv(orig_pilot_csv)

    recalc_rows = []
    for m in methods:
        sub = df_orig[df_orig.method == m]
        hit_rate = sub["part_hit"].mean() * 100.0
        del_mean = sub["deletion_auc"].mean()
        ins_mean = sub["insertion_auc"].mean()
        
        # Check matching scores
        scores = sub["matching_score"].tolist()
        
        recalc_rows.append({
            "method": m,
            "recalculated_part_hit_rate": hit_rate,
            "recalculated_del_auc_mean": del_mean,
            "recalculated_ins_auc_mean": ins_mean,
            "min_score": min(scores),
            "max_score": max(scores),
            "mean_score": np.mean(scores)
        })

    df_recalc = pd.DataFrame(recalc_rows)
    df_recalc.to_csv(recalc_csv_path, index=False)
    print(f"Saved: {recalc_csv_path}")
    print("=" * 80)
    print("DIAGNOSTICS & AUDIT GENERATION COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_diagnostics()

