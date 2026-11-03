"""
methods/vanilla_cam.py
Baseline Class Activation Mapping (CAM) implementation as used in Kenny et al. (IJCAI 2023).
Operates strictly at the terminal layer4 (7x7 resolution).
"""

import torch
import torch.nn.functional as F
import numpy as np

def compute_vanilla_cam(model, img_tensor, target_class=None):
    """
    Computes 7x7 CAM using model linear weights and layer4 activations.
    """
    model.eval()
    with torch.no_grad():
        logits, _, C = model(img_tensor) # C: [1, 512, 7, 7]
        if target_class is None:
            target_class = logits.argmax(dim=-1).item()

        weights = model.linear.weight[target_class] # [512]
        cam = torch.zeros((7, 7), device=img_tensor.device)
        for i in range(512):
            cam += weights[i] * C[0, i, :, :]

        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

    return cam.cpu().numpy(), target_class, C.cpu().detach()

