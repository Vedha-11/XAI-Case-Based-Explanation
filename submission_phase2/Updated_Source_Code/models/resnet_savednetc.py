"""
models/resnet_savednetc.py
Defines the SavedNetC architecture (fine-tuned ResNet-34) and hooks for multi-scale activation/gradient extraction.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class SavedNetC(nn.Module):
    """
    ResNet-34 classifier architecture matching Kenny et al. (IJCAI 2023).
    Returns (logits, x, C) where C is the layer4 feature map.
    """
    def __init__(self, num_classes=200):
        super(SavedNetC, self).__init__()
        resnet = models.resnet34(weights=None)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.avgpool = nn.AvgPool2d(7)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        l3 = self.layer3(x)  # [B, 256, 14, 14]
        l4 = self.layer4(l3) # [B, 512, 7, 7]

        pooled = self.avgpool(l4)
        flat = pooled.view(pooled.size(0), -1)
        logits = self.linear(flat)

        return logits, flat, l4

class MultiScaleFeatureExtractor:
    """
    Attaches forward and backward hooks to layer3 and layer4 to extract activations and gradients.
    """
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.gradients = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook_l3(module, input, output):
            self.activations['layer3'] = output

        def backward_hook_l3(module, grad_in, grad_out):
            self.gradients['layer3'] = grad_out[0]

        def forward_hook_l4(module, input, output):
            self.activations['layer4'] = output

        def backward_hook_l4(module, grad_in, grad_out):
            self.gradients['layer4'] = grad_out[0]

        self.hooks.append(self.model.layer3.register_forward_hook(forward_hook_l3))
        self.hooks.append(self.model.layer3.register_full_backward_hook(backward_hook_l3))
        self.hooks.append(self.model.layer4.register_forward_hook(forward_hook_l4))
        self.hooks.append(self.model.layer4.register_full_backward_hook(backward_hook_l4))

    def clear(self):
        self.activations.clear()
        self.gradients.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

