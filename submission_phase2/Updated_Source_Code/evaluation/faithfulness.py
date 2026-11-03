"""
evaluation/faithfulness.py
Computes explanation faithfulness via the 22-step Gaussian blur Deletion and Insertion game.
Based on RISE (Petsiuk et al., BMVC 2018).
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter

def compute_deletion_insertion_auc(model, img_pil, saliency_map, target_class, device='cuda:0', num_steps=22):
    """
    Evaluates Deletion AUC and Insertion AUC by ranking pixels by saliency intensity
    and iteratively perturbing them using Gaussian blur (radius=10).
    """
    model.eval()
    W, H = img_pil.size
    
    # Upsample saliency map to image size
    saliency_img = Image.fromarray((saliency_map * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    saliency_arr = np.array(saliency_img, dtype=np.float32) / 255.0
    
    # Flatten and rank pixel indices in descending order
    ranked_indices = np.argsort(-saliency_arr.flatten())
    total_pixels = len(ranked_indices)
    step_size = total_pixels // num_steps
    
    # Create uninformative blurred baseline
    blurred_img = img_pil.filter(ImageFilter.GaussianBlur(radius=10))
    orig_np = np.array(img_pil)
    blur_np = np.array(blurred_img)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    del_probs = []
    ins_probs = []
    
    del_canvas = orig_np.copy()
    ins_canvas = blur_np.copy()
    
    with torch.no_grad():
        for step in range(num_steps + 1):
            if step > 0:
                start_idx = (step - 1) * step_size
                end_idx = min(step * step_size, total_pixels)
                pix_to_modify = ranked_indices[start_idx:end_idx]
                
                # Deletion: replace salient pixels with blur
                y_coords, x_coords = np.unravel_index(pix_to_modify, (H, W))
                del_canvas[y_coords, x_coords] = blur_np[y_coords, x_coords]
                
                # Insertion: replace blur with original pixels
                ins_canvas[y_coords, x_coords] = orig_np[y_coords, x_coords]

            # Forward pass for Deletion
            del_t = transform(Image.fromarray(del_canvas)).unsqueeze(0).to(device)
            out_del, _, _ = model(del_t)
            prob_del = torch.softmax(out_del, dim=-1)[0, target_class].item()
            del_probs.append(prob_del)
            
            # Forward pass for Insertion
            ins_t = transform(Image.fromarray(ins_canvas)).unsqueeze(0).to(device)
            out_ins, _, _ = model(ins_t)
            prob_ins = torch.softmax(out_ins, dim=-1)[0, target_class].item()
            ins_probs.append(prob_ins)

    deletion_auc = float(np.trapz(del_probs, dx=1.0 / num_steps))
    insertion_auc = float(np.trapz(ins_probs, dx=1.0 / num_steps))
    
    return deletion_auc, insertion_auc

