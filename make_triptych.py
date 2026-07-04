"""
make_triptych.py
================
Figure 1: Classification Evasion Triptych — imperceptible adversarial perturbation on GTSRB ResNet-50.
Three panels: Clean image, Adversarial example, Amplified perturbation.
Uses PGD attack (20 steps, α=ε/4, random start) in normalized space, smallest ε that flips prediction.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_fn

# ────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────
DATA_DIR    = 'gtsrb/dataset'
TEST_CSV    = os.path.join(DATA_DIR, 'Test.csv')
CKPT_PATH   = 'gtsrb/best_resnet50_gtsrb.pth'
OUTPUT_DIR  = 'figs'
NUM_CLASSES = 43
SEED        = 42

# ImageNet normalization (same as notebooks)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# PGD configuration
PGD_STEPS   = 20
PGD_EPS_SWEEP = [0.002, 0.005, 0.01, 0.02]  # Smallest first; stop at first that flips

# Device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')

# Per-channel bounds in normalized space (for PGD clamping)
_mean  = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
_std   = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
_lower = (0 - _mean) / _std
_upper = (1 - _mean) / _std


# ────────────────────────────────────────────────────────────────
# DATASET & MODEL LOADING
# ────────────────────────────────────────────────────────────────
class GTSRBTestDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['Path'])
        image = Image.open(img_path).convert('RGB')
        label = int(row['ClassId'])
        if self.transform:
            image = self.transform(image)
        return image, label


# Transforms: normalized for model input, raw [0,1] for display
norm_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

raw_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load datasets
full_norm = GTSRBTestDataset(TEST_CSV, DATA_DIR, transform=norm_transform)
full_raw  = GTSRBTestDataset(TEST_CSV, DATA_DIR, transform=raw_transform)

# Load model
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model = model.to(device)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

criterion = nn.CrossEntropyLoss()

print(f'Loaded model: {CKPT_PATH}')
print(f'Dataset: {len(full_norm)} images')


# ────────────────────────────────────────────────────────────────
# HELPER: Find a clean image classified correctly with high confidence
# ────────────────────────────────────────────────────────────────
def find_clean_image_high_confidence(threshold_conf=0.8):
    """Find first test image classified correctly with confidence >= threshold_conf."""
    for idx in range(len(full_norm)):
        img_norm, label = full_norm[idx]
        img_raw, _ = full_raw[idx]
        
        with torch.no_grad():
            logits = model(img_norm.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1)[0]
            pred = probs.argmax().item()
            conf = probs[pred].item()
        
        if pred == label and conf >= threshold_conf:
            print(f'Found image (idx={idx}): true_label={label}, pred={pred}, conf={conf:.4f}')
            return idx, img_norm, img_raw, label
    
    raise ValueError(f'No image found with confidence >= {threshold_conf}')


# ────────────────────────────────────────────────────────────────
# PGD ATTACK
# ────────────────────────────────────────────────────────────────
def pgd_attack(model, x, labels, eps, alpha, steps, lower, upper):
    """PGD attack in normalized space: random start + iterated FGSM + ε-ball projection.
    x: input tensor (batch_size, 3, H, W) in normalized space, requires_grad=False initially
    """
    device = x.device
    
    # Random start within ε-ball
    x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, x - eps, x + eps).detach()
    
    # Iterative steps
    for step_idx in range(steps):
        x_adv = x_adv.requires_grad_(True)
        
        with torch.enable_grad():
            logits = model(x_adv)
            loss = criterion(logits, labels)
            loss.backward()
        
        # Gradient step
        with torch.no_grad():
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv + alpha * grad_sign
            
            # Clamp to ε-ball around original x
            x_adv = torch.clamp(x_adv, x - eps, x + eps)
            
            # Clamp to normalized space bounds (per-channel)
            x_adv = torch.clamp(x_adv, lower.to(device), upper.to(device))
        
        x_adv = x_adv.detach()
    
    return x_adv


def find_flipping_eps(img_norm, img_raw, true_label):
    """Sweep eps values to find smallest ε that flips prediction."""
    img_norm = img_norm.unsqueeze(0).to(device)
    
    for eps in PGD_EPS_SWEEP:
        alpha = eps / 4
        
        with torch.no_grad():
            x_adv = pgd_attack(model, img_norm, torch.tensor([true_label]).to(device), 
                               eps, alpha, PGD_STEPS, _lower.to(device), _upper.to(device))
            pred = model(x_adv).argmax(1).item()
        
        if pred != true_label:
            print(f'Found flipping eps={eps}: prediction flipped from {true_label} to {pred}')
            return eps, x_adv.squeeze(0)
    
    raise ValueError(f'No flipping eps found in sweep {PGD_EPS_SWEEP}')


# ────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('Finding clean image with high confidence...')
print('='*60)
idx, img_norm, img_raw, true_label = find_clean_image_high_confidence(threshold_conf=0.8)

print('\n' + '='*60)
print('Sweeping epsilon to find prediction flip...')
print('='*60)
eps, x_adv_norm = find_flipping_eps(img_norm, img_raw, true_label)

# Convert to [0,1] pixel space for display and metrics
img_norm_denorm = img_norm.unsqueeze(0).to(device)  # Add batch dim, move to device
x_adv_norm_denorm = x_adv_norm.unsqueeze(0)  # already on device from pgd_attack

# Denormalize: x_01 = (x_norm * std) + mean
mean_tensor = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
std_tensor = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)

img_01 = img_norm_denorm * std_tensor + mean_tensor
x_adv_01 = x_adv_norm_denorm * std_tensor + mean_tensor

# Clip to [0,1]
img_01 = torch.clamp(img_01, 0, 1)
x_adv_01 = torch.clamp(x_adv_01, 0, 1)

# Get predictions for captions
with torch.no_grad():
    clean_pred = model(img_norm.unsqueeze(0).to(device)).argmax(1).item()
    adv_pred = model(x_adv_norm.unsqueeze(0).to(device)).argmax(1).item()

# Compute perturbation and SSIM
delta = x_adv_01 - img_01
img_np = img_01.squeeze(0).permute(1, 2, 0).cpu().numpy()
x_adv_np = x_adv_01.squeeze(0).permute(1, 2, 0).cpu().numpy()
ssim_val = ssim_fn(img_np, x_adv_np, channel_axis=2, data_range=1.0)

print(f'\nClean prediction: {clean_pred}')
print(f'Adversarial prediction: {adv_pred}')
print(f'SSIM (clean vs adversarial): {ssim_val:.6f}')
print(f'ε (normalized space): {eps:.6f}')

# ────────────────────────────────────────────────────────────────
# VISUALIZATION: THREE PANELS
# ────────────────────────────────────────────────────────────────
# Visualize perturbation: use absolute magnitude normalized within image
# Since ε is tiny (0.002), we normalize per-pixel magnitude to [0,1] for visibility
delta_mag = torch.sqrt((delta ** 2).sum(dim=1, keepdim=True)).clamp(1e-8)  # magnitude per pixel
max_mag = delta_mag.max().item()
min_mag = delta_mag.min().item()
print(f'Perturbation magnitude: min={min_mag:.6f}, max={max_mag:.6f}')

# Normalize magnitude to [0,1] within the image (shows spatial pattern of attack)
delta_vis = (delta_mag - min_mag) / (max_mag - min_mag + 1e-8)
delta_vis = delta_vis.repeat(1, 3, 1, 1)  # replicate to RGB

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Panel 1: Clean
ax = axes[0]
ax.imshow(img_01.squeeze(0).permute(1, 2, 0).cpu().numpy())
ax.set_title(f'Clean\nPred: {clean_pred}', fontsize=11, fontweight='bold')
ax.axis('off')

# Panel 2: Adversarial
ax = axes[1]
ax.imshow(x_adv_01.squeeze(0).permute(1, 2, 0).cpu().numpy())
ax.set_title(f'Adversarial (ε={eps:.4f})\nPred: {adv_pred}', fontsize=11, fontweight='bold')
ax.axis('off')

# Panel 3: Amplified Perturbation (grayscale magnitude)
ax = axes[2]
ax.imshow(delta_vis.squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='hot')
ax.set_title(f'Perturbation magnitude\n(normalized spatial pattern)\nSSIM: {ssim_val:.6f}', fontsize=11, fontweight='bold')
ax.axis('off')

plt.suptitle(f'Classification Evasion — ResNet-50 on GTSRB (PGD, {PGD_STEPS} steps, ε={eps:.4f}, SSIM={ssim_val:.6f})', 
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

# Save figure
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, 'evasion_triptych.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'\nSaved: {output_path}')
plt.close()

print('\n' + '='*60)
print('Done!')
print('='*60)
