# %% [code] {"execution":{"iopub.status.busy":"2026-02-15T12:21:12.686920Z","iopub.execute_input":"2026-02-15T12:21:12.687735Z","iopub.status.idle":"2026-02-15T12:21:12.696651Z","shell.execute_reply.started":"2026-02-15T12:21:12.687708Z","shell.execute_reply":"2026-02-15T12:21:12.695854Z"}}
import os, time, json, warnings, random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from collections import defaultdict
from pathlib import Path
from IPython.display import display, clear_output

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
import cv2

warnings.filterwarnings('ignore')

# ─── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ─── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🖥️  Device   : {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'   GPU      : {torch.cuda.get_device_name(0)}')
    print(f'   VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'   PyTorch  : {torch.__version__}')
print()

print('✅ Imports done!')

# %% [code] {"execution":{"iopub.status.busy":"2026-02-15T12:21:12.697844Z","iopub.execute_input":"2026-02-15T12:21:12.698524Z","iopub.status.idle":"2026-02-15T12:21:12.717911Z","shell.execute_reply.started":"2026-02-15T12:21:12.698502Z","shell.execute_reply":"2026-02-15T12:21:12.717113Z"}}

CFG = {
    'data_root'   : '/kaggle/input/datasets/colabonkaggle/dataset',          # root folder containing Train/ Val/ testImages/
    'train_img'   : '/Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/train/Color_Images',        # relative to data_root
    'train_mask'  : '/Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/train/Segmentation',
    'val_img'     : '/Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/val/Color_Images',
    'val_mask'    : '/Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/val/Segmentation',
    'test_img'    : '/Offroad_Segmentation_testImages/Offroad_Segmentation_testImages/Color_Images',

    'epochs'      : 40,
    'batch_size'  : 4,         
    'img_size'    : (768,768), 
    'num_workers' : 4,
    'pin_memory'  : True,

    'lr'          : 4e-5,        
    'weight_decay': 0.05,
    'drop_path_rate': 0.1,
    'warmup_epochs': 3,

    
    'amp'         : True,      
    'compile'     : False,    
    'backbone'    : 'nvidia/mit-b2',
    'save_dir'    : '/kaggle/working/runs/segformer_b2',
    'inf_target_ms': 45,         
}

os.makedirs(f"{CFG['save_dir']}/plots",       exist_ok=True)
os.makedirs(f"{CFG['save_dir']}/checkpoints", exist_ok=True)
os.makedirs(f"{CFG['save_dir']}/predictions", exist_ok=True)

print('⚙️  Config loaded!')
for k, v in CFG.items():
    print(f'   {k:15s} : {v}')

# %% [code] {"execution":{"iopub.status.busy":"2026-02-15T12:21:12.741043Z","iopub.execute_input":"2026-02-15T12:21:12.741252Z","iopub.status.idle":"2026-02-15T12:21:13.473716Z","shell.execute_reply.started":"2026-02-15T12:21:12.741236Z","shell.execute_reply":"2026-02-15T12:21:13.473016Z"}}
# ─── 10 Duality AI Classes ─────────────────────────────────────────────────────
CLASS_CONFIG = {
    100  : {'name': 'Trees',          'color': (34,  139, 34),   'idx': 0},
    200  : {'name': 'Lush Bushes',    'color': (0,   200, 0),    'idx': 1},
    300  : {'name': 'Dry Grass',      'color': (210, 180, 140),  'idx': 2},
    500  : {'name': 'Dry Bushes',     'color': (139, 90,  43),   'idx': 3},
    550  : {'name': 'Ground Clutter', 'color': (160, 82,  45),   'idx': 4},
    600  : {'name': 'Flowers',        'color': (255, 105, 180),  'idx': 5},
    700  : {'name': 'Logs',           'color': (101, 67,  33),   'idx': 6},
    800  : {'name': 'Rocks',          'color': (128, 128, 128),  'idx': 7},
    7100 : {'name': 'Landscape',      'color': (210, 180, 100),  'idx': 8},
    10000: {'name': 'Sky',            'color': (135, 206, 235),  'idx': 9},
}

NUM_CLASSES  = len(CLASS_CONFIG)
CLASS_NAMES  = [v['name']  for v in sorted(CLASS_CONFIG.values(), key=lambda x: x['idx'])]
CLASS_COLORS = [v['color'] for v in sorted(CLASS_CONFIG.values(), key=lambda x: x['idx'])]
ID_TO_IDX    = {k: v['idx']   for k, v in CLASS_CONFIG.items()}
IDX_TO_COLOR = {v['idx']: v['color'] for v in CLASS_CONFIG.values()}
COLOR_ARRAY  = np.array([IDX_TO_COLOR[i] for i in range(NUM_CLASSES)], dtype=np.uint8)

def hex_colors():
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in CLASS_COLORS]

# ─── Visualise the 10 classes ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(16, 5), facecolor='#0d1117')
fig.suptitle('Duality AI — 10 Segmentation Classes', color='white', fontsize=14, fontweight='bold')
for i, (ax, name, color) in enumerate(zip(axes.flat, CLASS_NAMES, CLASS_COLORS)):
    patch = np.full((60, 140, 3), color, dtype=np.uint8)
    ax.imshow(patch); ax.axis('off')
    ax.set_title(f'ID {list(CLASS_CONFIG.keys())[i]}\n{name}',
                 color='white', fontsize=9, pad=3)
    ax.set_facecolor('#0d1117')
plt.tight_layout()
plt.savefig(f"{CFG['save_dir']}/plots/class_palette.png", dpi=120,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
print(f'✅ {NUM_CLASSES} classes configured!')

# %% [code] {"execution":{"iopub.status.busy":"2026-02-15T12:21:13.475055Z","iopub.execute_input":"2026-02-15T12:21:13.475337Z","iopub.status.idle":"2026-02-15T12:26:54.853305Z","shell.execute_reply.started":"2026-02-15T12:21:13.475315Z","shell.execute_reply":"2026-02-15T12:26:54.852643Z"}}
import torch
import numpy as np
import cv2
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Standard ImageNet stats
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

import torchvision.transforms.functional as TF

class CleanedDualityDataset(Dataset):
    """
    Dataset with Stronger Augmentations (Scale, Rotate, Shift) + Cleaning.
    Fixes overfitting by preventing memorization.
    """
    def __init__(self, img_dir, mask_dir, augment=True):
        self.augment = augment
        self.size    = CFG['img_size']
        
        img_dir  = Path(img_dir)
        mask_dir = Path(mask_dir)
        
        self.img_paths  = sorted(list(img_dir.glob('**/*.png')) + list(img_dir.glob('**/*.jpg')))
        self.mask_paths = sorted(list(mask_dir.glob('**/*.png')))
        
        # Base transforms
        self.normalize = transforms.Normalize(IMG_MEAN, IMG_STD)

    def _remap(self, mask_np):
        out = np.full(mask_np.shape[:2], 255, dtype=np.uint8)
        for raw_id, idx in ID_TO_IDX.items():
            out[mask_np == raw_id] = idx
        return out

    def _clean_mask_data(self, mask_np):
        # Same cleaning logic as before
        kernel = np.ones((3,3), np.uint8)
        mask_clean = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
        dilation = cv2.dilate(mask_clean, kernel, iterations=1)
        erosion  = cv2.erode(mask_clean, kernel, iterations=1)
        boundary = cv2.subtract(dilation, erosion)
        mask_clean[boundary > 0] = 255
        return mask_clean

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx])

        # Resize first
        img = img.resize(self.size, resample=Image.BILINEAR)
        mask = mask.resize(self.size, resample=Image.NEAREST)

        # ── STRONG AUGMENTATION ───────────────────────────────────────────────
        if self.augment:
            # 1. Random Rotate (-20 to 20 degrees)
            if random.random() > 0.3:
                angle = random.uniform(-20, 20)
                img = TF.rotate(img, angle, interpolation=transforms.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)

            # 2. Random Scale/Zoom (0.8x to 1.2x)
            if random.random() > 0.3:
                scale = random.uniform(0.8, 1.2)
                # Affine handles the zoom-in/out
                img = TF.affine(img, angle=0, translate=(0,0), scale=scale, shear=0, interpolation=transforms.InterpolationMode.BILINEAR)
                mask = TF.affine(mask, angle=0, translate=(0,0), scale=scale, shear=0, interpolation=transforms.InterpolationMode.NEAREST)

            # 3. Flips (Standard)
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            
            # 4. Color Jitter (Image Only)
            if random.random() > 0.2:
                jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
                img = jitter(img)

        # Convert to Tensor
        img_np = np.array(img)
        mask_np = np.array(mask)
        
        mask_mapped = self._remap(mask_np)
        
        # Apply cleaning only on training
        if self.augment:
            mask_mapped = self._clean_mask_data(mask_mapped)

        img_tensor = transforms.ToTensor()(img_np)
        img_tensor = self.normalize(img_tensor)
        mask_tensor = torch.from_numpy(mask_mapped).long()
        
        return img_tensor, mask_tensor

    def __len__(self):
        return len(self.img_paths)
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255, dice_w=0.4):
        """
        Focal Loss + Dice Loss
        gamma: Focus parameter. Higher gamma (e.g., 2.0 or 4.0) makes model focus MORE on hard examples.
        """
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.dice_w = dice_w
        
        # Optional: Manual class weights (alpha) if you want to double-force rare classes
        # If alpha is None, we rely purely on Focal Loss math to find hard examples.
        self.alpha = alpha 

    def focal_loss(self, logits, targets):
        # 1. Calculate Standard Cross Entropy (raw loss)
        ce_loss = F.cross_entropy(logits, targets, reduction='none', 
                                  ignore_index=self.ignore_index, weight=self.alpha)
        
        # 2. Calculate Probabilities (pt)
        # pt is "how sure the model is". High pt = easy example. Low pt = hard example.
        pt = torch.exp(-ce_loss)
        
        # 3. Apply Focal Weighting: (1 - pt)^gamma
        # If model is 99% sure (pt=0.99), weight becomes (0.01)^2 ≈ 0.0001 (Loss ignored)
        # If model is 20% sure (pt=0.20), weight becomes (0.80)^2 ≈ 0.64 (Loss kept high)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        return focal_loss.mean()

    def dice_loss(self, logits, targets):
        prob = torch.softmax(logits, dim=1)
        mask = (targets != self.ignore_index).unsqueeze(1).float()
        
        safe_targets = targets.clone()
        safe_targets[targets == self.ignore_index] = 0
        tgt = torch.zeros_like(prob)
        tgt.scatter_(1, safe_targets.unsqueeze(1), 1)
        
        inter = (prob * tgt * mask).sum(dim=(2,3))
        union = (prob * mask).sum(dim=(2,3)) + (tgt * mask).sum(dim=(2,3))
        
        dice = (2. * inter + 1e-7) / (union + 1e-7)
        return 1 - dice.mean()

    def forward(self, logits, targets):
        # Combined Focal + Dice
        return ((1 - self.dice_w) * self.focal_loss(logits, targets) + 
                self.dice_w * self.dice_loss(logits, targets))

# Initialize with gamma=2.0 (standard) or 4.0 (aggressive focus on hard/rare stuff)
criterion = FocalLoss(gamma=2.0, ignore_index=255, dice_w=0.5)
# ─── Create datasets ──────────────────────────────────────────────────────────

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

# 1. Define Datasets (Using the CleanedDualityDataset we defined earlier)
root = CFG['data_root']

print(f"📂 Loading images from: {root}")

# We remove the try-except block so errors are visible immediately
train_ds = CleanedDualityDataset(
    f"{root}/{CFG['train_img']}", 
    f"{root}/{CFG['train_mask']}", 
    augment=True
)

val_ds = CleanedDualityDataset(
    f"{root}/{CFG['val_img']}",   
    f"{root}/{CFG['val_mask']}",   
    augment=False
)

# 2. CALCULATE SAMPLING WEIGHTS (The Anti-Overfitting / Rare Class Fix)
print("⏳ Scanning dataset to calculate class weights (this takes ~30s)...")

# Target Classes to Boost: 5:Flowers, 6:Logs, 7:Rocks
# These weights make rare classes appear 5x-20x more often
sample_weights = []

# We iterate through the dataset once to find the rare images
for i in tqdm(range(len(train_ds)), desc="Scanning for Rare Classes"):
    # Load just the mask to check contents (ignoring the image for speed)
    _, mask_tensor = train_ds[i] 
    mask_np = mask_tensor.numpy()
    
    weight = 1.0 # Default weight
    
    unique_classes = np.unique(mask_np)
    
    # HEAVY BOOST for the classes your model ignores
    if 6 in unique_classes:   # LOGS
        weight += 20.0        # Massive boost
    if 5 in unique_classes:   # FLOWERS
        weight += 10.0
    if 7 in unique_classes:   # ROCKS
        weight += 5.0
        
    sample_weights.append(weight)

# Convert to DoubleTensor for the Sampler
sample_weights = torch.DoubleTensor(sample_weights)

# Create the Sampler
# replacement=True allows us to pick the same "Log" image multiple times in one epoch
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

# 3. Create DataLoaders
# NOTE: shuffle=False is REQUIRED when using a sampler
train_loader = DataLoader(
    train_ds, 
    batch_size=CFG['batch_size'], 
    sampler=sampler,          # <--- INJECT SAMPLER HERE
    shuffle=False,            # MUST BE FALSE
    num_workers=CFG['num_workers'], 
    pin_memory=CFG['pin_memory'], 
    drop_last=True,
    persistent_workers=True
)

val_loader = DataLoader(
    val_ds, 
    batch_size=CFG['batch_size'], 
    shuffle=False,
    num_workers=CFG['num_workers'], 
    pin_memory=CFG['pin_memory'],
    persistent_workers=True
)

# 4. Print Statistics (As requested)
print(f'📦 Train samples : {len(train_ds):,}')
print(f'📦 Val samples   : {len(val_ds):,}')
print(f'📦 Train batches : {len(train_loader):,}')
print(f'📦 Val batches   : {len(val_loader):,}')
print('✅ Data Loaders Ready with Weighted Sampling!')

# %% [code] {"execution":{"iopub.status.busy":"2026-02-15T12:26:54.854498Z","iopub.execute_input":"2026-02-15T12:26:54.854936Z","iopub.status.idle":"2026-02-15T12:26:55.406187Z","shell.execute_reply.started":"2026-02-15T12:26:54.854912Z","shell.execute_reply":"2026-02-15T12:26:55.405389Z"}}
def build_segformer(device):
    """Load SegFormer-B2 from HuggingFace with Drop Path Rate fix."""
    try:
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerConfig
        )
        print(f'⬇️  Downloading {CFG["backbone"]} from HuggingFace ...')
        
        # 1. LOAD CONFIG FIRST
        config = SegformerConfig.from_pretrained(
            CFG['backbone'],
            num_labels=NUM_CLASSES,
            id2label={i: n for i, n in enumerate(CLASS_NAMES)},
            label2id={n: i for i, n in enumerate(CLASS_NAMES)},
        )
        
        # 2. INJECT DROP PATH RATE HERE (Crucial for fixing Overfitting)
        # This tells the model: "Randomly drop 10% of paths during training"
        config.drop_path_rate = CFG.get('drop_path_rate', 0.1)
        
        # 3. LOAD MODEL WITH MODIFIED CONFIG
        model = SegformerForSemanticSegmentation.from_pretrained(
            CFG['backbone'],
            config=config,                # Pass the modified config
            ignore_mismatched_sizes=True
        )
        
        model._is_segformer = True
        print(f'✅ SegFormer-B2 loaded with Drop Path Rate: {config.drop_path_rate}')
        print(f'   Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')
        
    except Exception as e:
        print(f'⚠️  HuggingFace unavailable: {e}')
        print('   Falling back to DeepLabV3+ (torchvision) ...')
        from torchvision.models.segmentation import deeplabv3_resnet101
        model = deeplabv3_resnet101(pretrained=True)
        model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, 1)
        model._is_segformer = False

    if CFG['compile'] and hasattr(torch, 'compile'):
        print('⚡ Compiling model with torch.compile ...')
        model = torch.compile(model)

    return model.to(device)


import torch.nn.utils.prune as prune

def apply_pruning_to_decoder(model, amount=0.2):
    print(f"✂️ Pruning {amount*100}% of weights from the Decoder Head...")
    for name, module in model.decode_head.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    print("✅ Pruning complete.")

# 2. Apply it RIGHT HERE
model = build_segformer(DEVICE)  # <--- Existing line
apply_pruning_to_decoder(model, amount=0.2) # <--- ADD THIS LINE
# model.to(DEVICE) # Ensure it stays on device

# %% [code] {"execution":{"iopub.status.busy":"2026-02-15T12:26:55.408046Z","iopub.execute_input":"2026-02-15T12:26:55.408329Z","iopub.status.idle":"2026-02-15T12:26:55.423803Z","shell.execute_reply.started":"2026-02-15T12:26:55.408307Z","shell.execute_reply":"2026-02-15T12:26:55.423211Z"}}
# ─── Combined CE + Dice Loss ───────────────────────────────────────────────────
class CombinedLoss(nn.Module):
    """Cross-Entropy (60%) + Dice (40%) — handles class imbalance well."""
    def __init__(self, ignore=255, dice_w=0.4):
        super().__init__()
        self.ce     = nn.CrossEntropyLoss(ignore_index=ignore)
        self.dice_w = dice_w
        self.ignore = ignore

    def dice_loss(self, logits, targets):
        prob = torch.softmax(logits, dim=1)

    # mask for valid pixels
        mask = (targets != self.ignore).unsqueeze(1).float()

    # replace ignore index (255) with 0 to avoid scatter crash
        safe_targets = targets.clone()
        safe_targets[targets == self.ignore] = 0

    # one-hot encode safely
        tgt = torch.zeros_like(prob)
        tgt.scatter_(1, safe_targets.unsqueeze(1), 1)

    # apply mask
        inter = (prob * tgt * mask).sum(dim=(0,2,3))
        union = (prob * mask).sum(dim=(0,2,3)) + (tgt * mask).sum(dim=(0,2,3))

        dice = (2. * inter + 1e-7) / (union + 1e-7)

        return 1 - dice.mean()

    def forward(self, logits, targets):
        return ((1 - self.dice_w) * self.ce(logits, targets) +
                self.dice_w * self.dice_loss(logits, targets))


# ─── Streaming IoU Metrics ─────────────────────────────────────────────────────
class SegMetrics:
    def __init__(self, n=NUM_CLASSES, ignore=255):
        self.n = n; self.ig = ignore; self.reset()

    def reset(self):
        self.confusion = np.zeros((self.n, self.n), dtype=np.int64)
        self.px_counts = np.zeros(self.n, dtype=np.int64)

    def update(self, pred, target):
        pred   = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        valid  = target != self.ig
        p, t   = pred[valid], target[valid]
        np.add.at(self.px_counts, t, 1)
        np.add.at(self.confusion.ravel(), t * self.n + p, 1)

    def iou_per_class(self):
        tp  = np.diag(self.confusion)
        denom = tp + (self.confusion.sum(0) - tp) + (self.confusion.sum(1) - tp)
        return np.where(denom > 0, tp / denom, np.nan)

    def mean_iou(self): return float(np.nanmean(self.iou_per_class()))
    def pixel_acc(self):
        return np.diag(self.confusion).sum() / max(self.confusion.sum(), 1)


# ─── Optimiser with Differential LR ───────────────────────────────────────────
criterion = CombinedLoss(ignore=255)
scaler    = GradScaler(enabled=CFG['amp'])

backbone_p, head_p = [], []
for name, p in model.named_parameters():
    (backbone_p if 'segformer.encoder' in name else head_p).append(p)

optimizer = optim.AdamW([
    {'params': backbone_p, 'lr': CFG['lr'] * 0.1},   # backbone: 10% of head LR
    {'params': head_p,     'lr': CFG['lr']},
], weight_decay=CFG['weight_decay'])

total_steps  = CFG['epochs'] * len(train_loader)
warmup_steps = CFG['warmup_epochs'] * len(train_loader)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[CFG['lr'] * 0.1, CFG['lr']],
    total_steps=total_steps,
    pct_start=warmup_steps / total_steps,
    div_factor=10, final_div_factor=100)

print('✅ Loss function   : CE (60%) + Dice (40%)')
print('✅ Optimiser       : AdamW (differential LR)')
print('✅ Scheduler       : OneCycleLR')
print(f'   Backbone LR    : {CFG["lr"]*0.1:.2e}')
print(f'   Head LR        : {CFG["lr"]:.2e}')
print(f'   Total steps    : {total_steps:,}')

# %% [code] {"execution":{"iopub.status.busy":"2026-02-15T12:26:55.424727Z","iopub.execute_input":"2026-02-15T12:26:55.424993Z","iopub.status.idle":"2026-02-15T12:26:55.451650Z","shell.execute_reply.started":"2026-02-15T12:26:55.424963Z","shell.execute_reply":"2026-02-15T12:26:55.451026Z"}}
# ─── Colour utilities ──────────────────────────────────────────────────────────
HEX = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in CLASS_COLORS]

def colorise(pred_idx):
    """Map class index array → RGB colour image."""
    return COLOR_ARRAY[pred_idx.flatten()].reshape(*pred_idx.shape, 3)

def overlay(orig, colour, alpha=0.55):
    return (orig * (1 - alpha) + colour * alpha).astype(np.uint8)


# ─── History storage ───────────────────────────────────────────────────────────
history = {
    'epoch': [], 'train_loss': [], 'val_loss': [],
    'mean_iou': [], 'val_time_ms': [], 'pixel_acc': [],
}
for i in range(NUM_CLASSES):
    history[f'iou_{i}'] = []
    history[f'px_{i}']  = []


def plot_epoch_dashboard(epoch, per_class_iou, px_counts,
                          train_loss, val_loss, mean_iou,
                          val_time_ms, save=True):
    """Full 6-panel dashboard — saved as PNG + displayed in notebook."""
    epochs = history['epoch']
    BG, SURF, BORDER = '#0d1117', '#161b22', '#30363d'
    GRID_KW = dict(color='#30363d', linestyle='--', alpha=0.5)

    fig = plt.figure(figsize=(22, 16), facecolor=BG)
    gs  = GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.36)

    def style_ax(ax):
        ax.set_facecolor(SURF)
        ax.tick_params(colors='white')
        ax.spines[:].set_color(BORDER)
        ax.grid(**GRID_KW)
        return ax

    iou_vals = np.nan_to_num(per_class_iou, nan=0.0)
    iou_colors = ['#3fb950' if v >= 0.65 else '#e3b341' if v >= 0.45 else '#f78166'
                  for v in iou_vals]

    # ── 1. Loss curves ─────────────────────────────────────────────────────────
    ax1 = style_ax(fig.add_subplot(gs[0, 0]))
    ax1.plot(epochs, history['train_loss'], color='#58a6ff', lw=2, marker='o',
             markersize=3, label='Train Loss')
    ax1.plot(epochs, history['val_loss'],   color='#f78166', lw=2, marker='s',
             markersize=3, label='Val Loss')
    ax1.set_title('Loss Curves', color='white', fontsize=11)
    ax1.set_xlabel('Epoch', color='white'); ax1.set_ylabel('Loss', color='white')
    ax1.legend(facecolor='#21262d', labelcolor='white', fontsize=9)
    if len(epochs) > 0:
        ax1.annotate(f"{history['train_loss'][-1]:.4f}",
                     (epochs[-1], history['train_loss'][-1]),
                     color='#58a6ff', fontsize=8, ha='right')
        ax1.annotate(f"{history['val_loss'][-1]:.4f}",
                     (epochs[-1], history['val_loss'][-1]),
                     color='#f78166', fontsize=8, ha='right')

    # ── 2. Mean IoU trend ──────────────────────────────────────────────────────
    ax2 = style_ax(fig.add_subplot(gs[0, 1]))
    ax2.plot(epochs, history['mean_iou'], color='#3fb950', lw=2.5,
             marker='D', markersize=3)
    ax2.fill_between(epochs, history['mean_iou'], alpha=0.15, color='#3fb950')
    best_miou = max(history['mean_iou']) if history['mean_iou'] else 0
    best_ep   = history['epoch'][history['mean_iou'].index(best_miou)] if history['mean_iou'] else 0
    ax2.axhline(best_miou, color='#f0e040', lw=1, ls=':', label=f'Best {best_miou:.4f} (ep{best_ep})')
    ax2.set_ylim(0, 1)
    ax2.set_title('Mean IoU (mIoU)', color='white', fontsize=11)
    ax2.set_xlabel('Epoch', color='white'); ax2.set_ylabel('mIoU', color='white')
    ax2.legend(facecolor='#21262d', labelcolor='white', fontsize=9)

    # ── 3. Inference time ──────────────────────────────────────────────────────
    ax3 = style_ax(fig.add_subplot(gs[0, 2]))
    bar_colors = ['#3fb950' if t < CFG['inf_target_ms'] else '#f78166'
                  for t in history['val_time_ms']]
    ax3.bar(epochs, history['val_time_ms'], color=bar_colors, width=0.7, alpha=0.85)
    ax3.axhline(CFG['inf_target_ms'], color='#f78166', lw=2, ls='--',
                label=f"{CFG['inf_target_ms']}ms target")
    if history['val_time_ms']:
        ax3.annotate(f"{history['val_time_ms'][-1]:.1f}ms",
                     (epochs[-1], history['val_time_ms'][-1] + 0.5),
                     color='white', fontsize=8, ha='center')
    ax3.set_title('Inference Time (ms/image)', color='white', fontsize=11)
    ax3.set_xlabel('Epoch', color='white'); ax3.set_ylabel('ms', color='white')
    ax3.legend(facecolor='#21262d', labelcolor='white', fontsize=9)

    # ── 4. Per-class IoU bar chart ─────────────────────────────────────────────
    ax4 = style_ax(fig.add_subplot(gs[1, :]))
    x   = np.arange(NUM_CLASSES)
    bars = ax4.bar(x, iou_vals, color=iou_colors, edgecolor=BORDER, width=0.65)
    for bar, val in zip(bars, iou_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', color='white', fontsize=9)
    ax4.set_xticks(x); ax4.set_xticklabels(CLASS_NAMES, rotation=25, ha='right',
                                            color='white', fontsize=9)
    ax4.set_ylim(0, 1.18)
    ax4.set_ylabel('IoU', color='white')
    ax4.set_title(f'Per-Class IoU — Epoch {epoch} / {CFG["epochs"]}', color='white', fontsize=11)
    ax4.axhline(mean_iou, color='#f0e040', lw=1.5, ls='--',
                label=f'mIoU = {mean_iou:.4f}')
    ax4.legend(facecolor='#21262d', labelcolor='white', fontsize=10)
    # Colour-coded legend squares
    for j, (name, color) in enumerate(zip(CLASS_NAMES, HEX)):
        ax4.bar(j, 0, color=color, label=name)

    # ── 5. Pixel distribution bar chart ───────────────────────────────────────
    ax5 = style_ax(fig.add_subplot(gs[2, :2]))
    px_pct = px_counts / (px_counts.sum() + 1e-8) * 100
    bars2 = ax5.bar(x, px_pct, color=HEX, edgecolor=BORDER, width=0.65)
    for bar, val, count in zip(bars2, px_pct, px_counts):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.1f}%\n({count//1000}k)', ha='center', va='bottom',
                 color='white', fontsize=7.5)
    ax5.set_xticks(x); ax5.set_xticklabels(CLASS_NAMES, rotation=25, ha='right',
                                            color='white', fontsize=9)
    ax5.set_ylabel('% pixels in Val set', color='white')
    ax5.set_title(f'Class Pixel Distribution (Val) — Epoch {epoch}', color='white', fontsize=11)

    # ── 6. Per-class IoU trend lines ───────────────────────────────────────────
    ax6 = style_ax(fig.add_subplot(gs[2, 2]))
    for i, name in enumerate(CLASS_NAMES):
        vals = history[f'iou_{i}']
        if vals:
            ax6.plot(epochs, vals, color=HEX[i], lw=1.5, label=name, alpha=0.9)
    ax6.set_ylim(0, 1)
    ax6.set_title('Per-Class IoU Trends', color='white', fontsize=11)
    ax6.set_xlabel('Epoch', color='white'); ax6.set_ylabel('IoU', color='white')
    ax6.legend(facecolor='#21262d', labelcolor='white', fontsize=6, ncol=2)

    fig.suptitle(
        f'Duality AI · SegFormer-B2 · Epoch {epoch}/{CFG["epochs"]}  '
        f'│  mIoU={mean_iou:.4f}  │  val_loss={val_loss:.4f}  '
        f'│  inf={val_time_ms:.1f}ms',
        color='white', fontsize=13, fontweight='bold', y=0.995)

    if save:
        path = f"{CFG['save_dir']}/plots/epoch_{epoch:03d}.png"
        plt.savefig(path, dpi=100, bbox_inches='tight', facecolor=BG)

    plt.show()
    plt.close(fig)


print('✅ Visualisation engine ready!')

# %% [code] {"execution":{"iopub.status.busy":"2026-02-15T12:26:55.452497Z","iopub.execute_input":"2026-02-15T12:26:55.452760Z","iopub.status.idle":"2026-02-15T12:26:55.469293Z","shell.execute_reply.started":"2026-02-15T12:26:55.452741Z","shell.execute_reply":"2026-02-15T12:26:55.468720Z"}}
import torch.nn.functional as F

def forward_pass(model, images, targets=None):
    """
    Unified forward pass for Hugging Face SegFormer.
    - Handles input formatting (pixel_values vs images)
    - Resizes output logits to match original image size
    - Returns internal loss if available
    """
    # 1. Run Model (Hugging Face style)
    outputs = model(pixel_values=images, labels=targets)
    
    # 2. Extract Logits
    logits = outputs.logits
    
    # 3. Resize Logits to Match Input Image Size
    # SegFormer outputs 128x128; we need 512x512
    logits = F.interpolate(
        logits, 
        size=images.shape[-2:],  # Target size (Height, Width)
        mode='bilinear', 
        align_corners=False
    )
    
    # 4. Return Logits and the Model's calculated loss (if targets provided)
    return logits, outputs.loss

print("✅ forward_pass function restored!")

# %% [code] {"execution":{"iopub.status.busy":"2026-02-15T12:26:55.470271Z","iopub.execute_input":"2026-02-15T12:26:55.470519Z","iopub.status.idle":"2026-02-15T15:49:02.578286Z","shell.execute_reply.started":"2026-02-15T12:26:55.470492Z","shell.execute_reply":"2026-02-15T15:49:02.568167Z"}}
from tqdm.auto import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, epoch):
    model.train()
    total_loss, step_count = 0.0, 0
    metrics = SegMetrics()

    pbar = tqdm(loader, desc=f"Train E{epoch}", leave=True)

    for imgs, masks in pbar:
        imgs  = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=CFG['amp']):
            logits, hf_loss = forward_pass(model, imgs, masks)
            if hf_loss is not None:
                loss = hf_loss + 0.4 * criterion.dice_loss(logits, masks)
            else:
                loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        metrics.update(logits.argmax(1), masks)
        total_loss += loss.item()
        step_count += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg": f"{total_loss/step_count:.4f}",
            "mIoU": f"{metrics.mean_iou():.4f}"
        })

    return total_loss / step_count, metrics


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATE
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    metrics    = SegMetrics()
    inf_times  = []

    pbar = tqdm(loader, desc="Validate", leave=True)

    for imgs, masks in pbar:
        imgs  = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with autocast(enabled=CFG['amp']):
            logits, hf_loss = forward_pass(model, imgs, masks)
            if hf_loss is not None:
                loss = hf_loss
            else:
                loss = criterion(logits, masks)

        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()

        inf_times.append((time.perf_counter() - t0) * 1000 / imgs.size(0))

        metrics.update(logits.argmax(1), masks)
        total_loss += loss.item()

        pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    return total_loss / len(loader), metrics, float(np.mean(inf_times))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP (With Pruning & Live Dashboard)
# ══════════════════════════════════════════════════════════════════════════════
from IPython.display import clear_output
import time
import numpy as np

# 1. SETUP MODEL & PRUNING
# ---------------------------------------------------------
# model = build_segformer(DEVICE)


# 2. INITIALIZE HISTORY (Reset for new run)
# ---------------------------------------------------------
best_miou  = 0.0
best_epoch = 0

# Clear history to avoid shape mismatch errors if re-running
history = {
    'epoch': [], 'train_loss': [], 'val_loss': [],
    'mean_iou': [], 'val_time_ms': [], 'pixel_acc': [],
}
for i in range(NUM_CLASSES):
    history[f'iou_{i}'] = []
    # Note: We need pixel counts for your dashboard
    history[f'px_{i}'] = [] 

print('='*60)
print(f'  Training SegFormer-B2  |  {CFG["epochs"]} epochs  |  Device: {DEVICE}')
print('='*60)


# 3. TRAINING LOOP
# ---------------------------------------------------------
for epoch in range(1, CFG['epochs'] + 1):
    
    # ── Train ─────────────────────────────────────────
    tr_loss, tr_metrics = train_one_epoch(
        model, train_loader, optimizer, scheduler, criterion, scaler, epoch)

    # ── Validate ──────────────────────────────────────
    vl_loss, vl_metrics, inf_ms = validate(model, val_loader, criterion)

    # ── Calculate Metrics ─────────────────────────────
    mean_iou = vl_metrics.mean_iou()
    pix_acc  = vl_metrics.pixel_acc()
    per_class_iou = vl_metrics.iou_per_class()
    
    # Get pixel counts for your dashboard (assuming metrics has this, or we simulate it)
    # If your SegMetrics doesn't have .pixel_counts(), we'll use a placeholder
    try:
        px_counts = vl_metrics.total_area_inter  # often stores total pixels per class
    except:
        px_counts = np.zeros(NUM_CLASSES) # Fallback

    # ── Update History (CRITICAL for plotting) ────────
    history['epoch'].append(epoch)
    history['train_loss'].append(tr_loss)
    history['val_loss'].append(vl_loss)
    history['mean_iou'].append(mean_iou)
    history['pixel_acc'].append(pix_acc)
    history['val_time_ms'].append(inf_ms)
    
    for i, iou_val in enumerate(per_class_iou):
        history[f'iou_{i}'].append(float(np.nan_to_num(iou_val)))
        # Store pixel counts if available, else 0
        count = px_counts[i] if i < len(px_counts) else 0
        history[f'px_{i}'].append(count)

    # ── Save Checkpoint ───────────────────────────────
    if mean_iou > best_miou:
        best_miou  = mean_iou
        best_epoch = epoch
        torch.save(model.state_dict(), f"{CFG['save_dir']}/checkpoints/best_model.pth")
    
    # ── LIVE DASHBOARD ────────────────────────────────
    clear_output(wait=True) # Clears the previous plot
    
    # Call YOUR existing visualization engine
    plot_epoch_dashboard(
        epoch=epoch,
        per_class_iou=per_class_iou,
        px_counts=px_counts,
        train_loss=tr_loss,
        val_loss=vl_loss,
        mean_iou=mean_iou,
        val_time_ms=inf_ms,
        save=True
    )
    
    print(f"Epoch {epoch} Complete | mIoU: {mean_iou:.4f} | Best: {best_miou:.4f}")

print(f'\n🏆 Training Finished. Best mIoU: {best_miou:.4f} @ Epoch {best_epoch}')

# %% [code]
#Code block to save pytorch model after training online on kaggle 
import shutil
from IPython.display import FileLink

source_path = "/kaggle/working/checkpoints/best_model.pth" 


output_filename = "my_model_backup"


shutil.make_archive(output_filename, 'zip', root_dir='/kaggle/working')

print("CLICK BELOW TO SAVE YOUR MODEL:")
display(FileLink(f'{output_filename}.zip'))