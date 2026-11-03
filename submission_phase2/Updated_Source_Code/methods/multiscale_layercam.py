"""
methods/multiscale_layercam.py
Proposed Multi-Scale LayerCAM implementation integrating layer3 (14x14) and layer4 (7x7) attributions.
Based on LayerCAM (Jiang et al., IEEE TIP 2021).
"""

import torch
import torch.nn.functional as F
import numpy as np

def compute_layercam_map(activations, gradients):
    """
    Computes LayerCAM attribution for a single layer:
    w_{k,i,j} = max(0, g_{k,i,j})
    M(i, j) = ReLU(sum_k w_{k,i,j} * A_{k,i,j})
    """
    weights = F.relu(gradients) # [1, C, H, W]
    cam = torch.sum(weights * activations, dim=1, keepdim=True) # [1, 1, H, W]
    cam = F.relu(cam)
    
    # Min-max normalization
    min_v = cam.min()
    max_v = cam.max()
    if (max_v - min_v) > 1e-7:
        cam = (cam - min_v) / (max_v - min_v)
    else:
        cam = torch.zeros_like(cam)
        
    return cam

def compute_multiscale_layercam(model, extractor, img_tensor, target_class=None, alpha=0.5):
    """
    Extracts multi-scale LayerCAM attribution across layer3 and layer4.
    Fuses maps at 14x14 resolution: M_fused = alpha * M_l3 + (1 - alpha) * upsample(M_l4).
    """
    model.eval()
    model.zero_grad()
    extractor.clear()

    # Forward pass
    logits, _, _ = model(img_tensor)
    if target_class is None:
        target_class = logits.argmax(dim=-1).item()

    # Backward pass for target class logit
    score = logits[0, target_class]
    score.backward(retain_graph=False)

    # Compute LayerCAM for layer3 (14x14) and layer4 (7x7)
    l3_act = extractor.activations['layer3']
    l3_grad = extractor.gradients['layer3']
    cam_l3 = compute_layercam_map(l3_act, l3_grad) # [1, 1, 14, 14]

    l4_act = extractor.activations['layer4']
    l4_grad = extractor.gradients['layer4']
    cam_l4 = compute_layercam_map(l4_act, l4_grad) # [1, 1, 7, 7]

    # Bilinear upsample layer4 map to 14x14
    cam_l4_up = F.interpolate(cam_l4, size=(14, 14), mode='bilinear', align_corners=False)

    # Convex fusion
    fused_cam = alpha * cam_l3 + (1.0 - alpha) * cam_l4_up
    fused_cam = fused_cam.squeeze().cpu().detach().numpy()

    # Normalize fused map
    min_v = fused_cam.min()
    max_v = fused_cam.max()
    if (max_v - min_v) > 1e-7:
        fused_cam = (fused_cam - min_v) / (max_v - min_v)

    l3_features = l3_act.cpu().detach()
    l4_features = l4_act.cpu().detach()
    
    extractor.clear()
    return fused_cam, target_class, l3_features, l4_features

