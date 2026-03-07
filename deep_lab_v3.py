# %% [code] {"id":"8JpdEfqGrwjs","outputId":"09a6fdf4-d509-42d9-b2fc-22da8f9ddec3","execution":{"iopub.status.busy":"2026-02-15T16:57:57.251752Z","iopub.execute_input":"2026-02-15T16:57:57.252321Z","iopub.status.idle":"2026-02-15T16:57:57.259823Z","shell.execute_reply.started":"2026-02-15T16:57:57.252292Z","shell.execute_reply":"2026-02-15T16:57:57.259151Z"}}
# Import libraries
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# %% [code] {"id":"sCjppadksI9h","outputId":"ebf7dd78-49b7-4ba5-e582-6d87b67e80b8","execution":{"iopub.status.busy":"2026-02-15T16:57:57.261037Z","iopub.execute_input":"2026-02-15T16:57:57.261720Z","iopub.status.idle":"2026-02-15T16:57:57.278035Z","shell.execute_reply.started":"2026-02-15T16:57:57.261698Z","shell.execute_reply":"2026-02-15T16:57:57.277357Z"}}
# Configuration
class Config:
    # Data
    train_dir = "/kaggle/input/datasets/vedanshtyagi28/offroad-segmentation/Offroad_Segmentation_Training_Dataset"
    test_dir = "/content/drive/MyDrive/Offroad_Segmentation_testImages"
    img_size = (512, 512)  # (height, width)

    # Model
    model_type = 'deeplabv3_resnet50'  # Options: deeplabv3_resnet50, deeplabv3_mobilenet, smp_deeplabv3
    num_classes = 10
    pretrained = True

    # Training
    # Lowered default batch size to reduce GPU memory usage
    batch_size = 4
    num_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-4
    # Use 0 on Windows to avoid worker/process pickling hangs; increase on Linux if desired
    num_workers = 0

    # Mixed precision and gradient accumulation to reduce GPU memory footprint
    use_amp = True
    grad_accum_steps = 1

    # Loss
    loss_type = 'combined'  # Options: dice, focal, iou, combined, dice_focal, ce
    use_class_weights = False

    # Directories
    save_dir = '/kaggle/working/checkpoints'
    log_dir = '/kaggle/working/logs'
    output_dir = '/kaggle/working/results'

    # Training control
    early_stopping = 20

config = Config()

# Create directories
os.makedirs(config.save_dir, exist_ok=True)
os.makedirs(config.log_dir, exist_ok=True)
os.makedirs(config.output_dir, exist_ok=True)

print("Configuration:")
print(f"  Model: {config.model_type}")
print(f"  Image size: {config.img_size}")
print(f"  Batch size: {config.batch_size}")
print(f"  Epochs: {config.num_epochs}")
print(f"  Loss: {config.loss_type}")

# %% [code] {"id":"42tD9hm6sRo0","outputId":"5d8e3e09-353f-4cdc-bc32-86e21562eb2c","execution":{"iopub.status.busy":"2026-02-15T16:57:57.317248Z","iopub.execute_input":"2026-02-15T16:57:57.317739Z","iopub.status.idle":"2026-02-15T16:57:57.427625Z","shell.execute_reply.started":"2026-02-15T16:57:57.317717Z","shell.execute_reply":"2026-02-15T16:57:57.426882Z"}}
# Class mapping from hackathon document
CLASS_MAPPING = {
    100: 0,    # Trees
    200: 1,    # Lush Bushes
    300: 2,    # Dry Grass
    500: 3,    # Dry Bushes
    550: 4,    # Ground Clutter
    600: 5,    # Flowers
    700: 6,    # Logs
    800: 7,    # Rocks
    7100: 8,   # Landscape
    10000: 9,  # Sky
}

REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

CLASS_NAMES = [
    'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes', 'Ground Clutter',
    'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

# Color palette for visualization
COLOR_PALETTE = np.array([
    [0, 255, 0],      # Trees - Green
    [0, 128, 0],      # Lush Bushes - Dark Green
    [255, 255, 0],    # Dry Grass - Yellow
    [139, 69, 19],    # Dry Bushes - Brown
    [160, 82, 45],    # Ground Clutter - Sienna
    [255, 0, 255],    # Flowers - Magenta
    [165, 42, 42],    # Logs - Brown
    [128, 128, 128],  # Rocks - Gray
    [210, 180, 140],  # Landscape - Tan
    [135, 206, 235],  # Sky - Light Blue
])

# Display class information
fig, ax = plt.subplots(figsize=(12, 3))
for i, (name, color) in enumerate(zip(CLASS_NAMES, COLOR_PALETTE)):
    ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color/255.))
    ax.text(i+0.5, 0.5, name, ha='center', va='center', fontsize=9, rotation=90)
ax.set_xlim(0, 10)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Class Colors', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [code] {"id":"IULWtIQqsTUK","outputId":"d0e268eb-3923-4eb3-82dc-6ac7aa76b0da","execution":{"iopub.status.busy":"2026-02-15T16:57:57.428876Z","iopub.execute_input":"2026-02-15T16:57:57.429108Z","iopub.status.idle":"2026-02-15T16:57:57.441710Z","shell.execute_reply.started":"2026-02-15T16:57:57.429087Z","shell.execute_reply":"2026-02-15T16:57:57.440941Z"}}
class DesertTerrainDataset(Dataset):
    """Dataset for desert terrain segmentation"""

    def __init__(self, root_dir, split,transform=None, img_size=(512, 512)):
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split

        self.image_dir = os.path.join(root_dir, 'Color_Images')
        self.mask_dir = os.path.join(root_dir, 'Segmentation')

        self.image_files = sorted([f for f in os.listdir(self.image_dir)
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])

        if transform is None:
            self.transform = self.get_default_transforms(split)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and mask
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path))

        # Convert mask to class indices
        mask = self.convert_mask_to_class_indices(mask)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.long()

    def convert_mask_to_class_indices(self, mask):
        """Convert mask values to class indices (0-9)"""
        output_mask = np.zeros_like(mask, dtype=np.int64)
        for original_value, class_idx in CLASS_MAPPING.items():
            output_mask[mask == original_value] = class_idx
        return output_mask

    def get_default_transforms(self, split):
        """Get augmentation transforms"""
        if split == 'train':
            return A.Compose([
                A.Resize(height=512, width=512),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5), # Good for top-down terrain
                A.RandomRotate90(p=0.5),
                
                # Geometric distortions (Important if repeating images!)
                A.ElasticTransform(p=0.3),
                A.GridDistortion(p=0.3),
                
                # Color jitter to simulate different times of day
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(height=self.img_size[0], width=self.img_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

print("✓ Dataset class defined")

# %% [code] {"id":"Gzz9A1qMslUA","outputId":"058d6933-424c-4fd3-e4fe-6d49c6a198cf","execution":{"iopub.status.busy":"2026-02-15T16:57:57.442721Z","iopub.execute_input":"2026-02-15T16:57:57.442973Z","iopub.status.idle":"2026-02-15T16:57:57.456673Z","shell.execute_reply.started":"2026-02-15T16:57:57.442952Z","shell.execute_reply":"2026-02-15T16:57:57.455972Z"}}
class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ model for semantic segmentation"""

    def __init__(self, num_classes=10, backbone='resnet50', pretrained=True):
        super(DeepLabV3Plus, self).__init__()

        self.num_classes = num_classes
        self.backbone = backbone

        if backbone == 'resnet50':
            self.model = deeplabv3_resnet50(pretrained=pretrained)
        elif backbone == 'mobilenet_v3':
            self.model = deeplabv3_mobilenet_v3_large(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Replace classifier for our classes
        in_channels = self.model.classifier[0].convs[0][0].in_channels
        self.model.classifier = DeepLabHead(in_channels, num_classes)

        # Update auxiliary classifier
        if hasattr(self.model, 'aux_classifier'):
            aux_in_channels = self.model.aux_classifier[0].in_channels
            self.model.aux_classifier = nn.Conv2d(aux_in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape[-2:]
        output = self.model(x)

        if isinstance(output, dict):
            output = output['out']

        if output.shape[-2:] != input_shape:
            output = F.interpolate(output, size=input_shape, mode='bilinear', align_corners=False)

        return output

print("✓ Model class defined")

# %% [code] {"id":"iM5gtoLasp6o","outputId":"2501473c-cfab-4837-90f5-0780a1c9f03c","execution":{"iopub.status.busy":"2026-02-15T16:57:57.458307Z","iopub.execute_input":"2026-02-15T16:57:57.458485Z","iopub.status.idle":"2026-02-15T16:57:57.475003Z","shell.execute_reply.started":"2026-02-15T16:57:57.458470Z","shell.execute_reply":"2026-02-15T16:57:57.474440Z"}}
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Define the Standard Dice Loss (Required helper) ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        num_classes = predictions.shape[1]
        # Apply Softmax to get probabilities
        predictions = F.softmax(predictions, dim=1)

        # Convert targets to One-Hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Flatten for calculation
        predictions = predictions.reshape(predictions.size(0), num_classes, -1)
        targets_one_hot = targets_one_hot.reshape(targets_one_hot.size(0), num_classes, -1)

        # Calculate intersection and union
        intersection = (predictions * targets_one_hot).sum(dim=2)
        union = predictions.sum(dim=2) + targets_one_hot.sum(dim=2)

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Return 1 - Dice (because we want to minimize loss)
        return 1 - dice_score.mean()

# --- 2. Define the Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            reduction='none', 
            ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
            
        return focal_loss.mean()

# --- 3. Combine Them ---
class CombinedFocalDiceLoss(nn.Module):
    def __init__(self, focal_weight=0.5, dice_weight=0.5, gamma=2.0, alpha=None):
        super(CombinedFocalDiceLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        loss_focal = self.focal(inputs, targets)
        loss_dice = self.dice(inputs, targets)
        return (self.focal_weight * loss_focal) + (self.dice_weight * loss_dice), loss_focal,loss_dice

print("✓ Loss functions (Dice + Focal) defined successfully")

# %% [code] {"id":"RN44SbFKsuD3","outputId":"47169553-9ad8-4f94-92f2-0f641047b000","execution":{"iopub.status.busy":"2026-02-15T16:57:57.475732Z","iopub.execute_input":"2026-02-15T16:57:57.475977Z","iopub.status.idle":"2026-02-15T16:57:57.489888Z","shell.execute_reply.started":"2026-02-15T16:57:57.475949Z","shell.execute_reply":"2026-02-15T16:57:57.489254Z"}}
def visualize_prediction(image, mask, prediction, class_names=CLASS_NAMES, palette=COLOR_PALETTE):
    """Visualize image, ground truth, and prediction"""

    # Denormalize image if needed
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    # Convert masks to RGB
    def mask_to_rgb(mask_indices):
        if isinstance(mask_indices, torch.Tensor):
            mask_indices = mask_indices.cpu().numpy()
        rgb = np.zeros((*mask_indices.shape, 3), dtype=np.uint8)
        for class_idx, color in enumerate(palette):
            rgb[mask_indices == class_idx] = color
        return rgb

    mask_rgb = mask_to_rgb(mask)
    pred_rgb = mask_to_rgb(prediction)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(mask_rgb)
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(pred_rgb)
    axes[2].set_title('Prediction', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # Overlay
    overlay = (image * 255 * 0.5 + pred_rgb * 0.5).astype(np.uint8)
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay', fontsize=14, fontweight='bold')
    axes[3].axis('off')

    plt.tight_layout()
    return fig


def plot_training_curves(train_losses, val_losses, val_ious):
    """Plot training curves"""
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # IoU
    axes[1].plot(epochs, val_ious, 'g-', linewidth=2)
    best_iou = max(val_ious)
    axes[1].axhline(y=best_iou, color='r', linestyle='--', label=f'Best: {best_iou:.4f}', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('IoU', fontsize=12)
    axes[1].set_title('Validation IoU', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

print("✓ Visualization functions defined")

# %% [code] {"id":"WJa-2rMvsxyN","outputId":"f70f3d90-43b2-4a03-fea1-57904fa1acb9","execution":{"iopub.status.busy":"2026-02-15T16:57:57.490766Z","iopub.execute_input":"2026-02-15T16:57:57.491042Z","iopub.status.idle":"2026-02-15T16:57:58.703336Z","shell.execute_reply.started":"2026-02-15T16:57:57.491015Z","shell.execute_reply":"2026-02-15T16:57:58.702397Z"}}
# Check if data directory exists
if not os.path.exists(config.train_dir):
    print(f"⚠️  Data directory not found: {config.train_dir}")
    print("Please update the DATA_DIR variable in the Configuration cell.")
else:
    # Create datasets
    train_dataset = DesertTerrainDataset(
        root_dir=os.path.join(config.train_dir, "train"),
        img_size=config.img_size,
        split='train'
    )

    val_dataset = DesertTerrainDataset(
        root_dir=os.path.join(config.train_dir, "val"),
        img_size=config.img_size,
        split='val'
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Visualize some samples
    print("\nVisualizing training samples...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(min(6, len(train_dataset))):
        image, mask = train_dataset[i]

        # Denormalize
        img = image.numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        # Convert mask to RGB
        mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_idx, color in enumerate(COLOR_PALETTE):
            mask_rgb[mask.numpy() == class_idx] = color

        # Overlay
        overlay = (img * 255 * 0.6 + mask_rgb * 0.4).astype(np.uint8)

        axes[i].imshow(overlay)
        axes[i].set_title(f'Sample {i+1}', fontsize=12)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# %% [code] {"id":"U1Gp6sVa3W7T","outputId":"2820a653-0f5d-4605-82f8-b7593e23e3bd","execution":{"iopub.status.busy":"2026-02-15T16:57:58.704302Z","iopub.execute_input":"2026-02-15T16:57:58.704556Z","iopub.status.idle":"2026-02-15T16:57:59.473683Z","shell.execute_reply.started":"2026-02-15T16:57:58.704508Z","shell.execute_reply":"2026-02-15T16:57:59.473021Z"}}
# Create model
print(f"Creating {config.model_type} model...")

if config.model_type == 'deeplabv3_resnet50':
    model = DeepLabV3Plus(num_classes=config.num_classes, backbone='resnet50', pretrained=config.pretrained)
elif config.model_type == 'deeplabv3_mobilenet':
    model = DeepLabV3Plus(num_classes=config.num_classes, backbone='mobilenet_v3', pretrained=config.pretrained)
else:
    raise ValueError(f"Unknown model type: {config.model_type}")

model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✓ Model created successfully")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# Test forward pass (use eval mode to avoid BatchNorm errors with batch_size=1)
dummy_input = torch.randn(1, 3, config.img_size[0], config.img_size[1]).to(device)
model.eval()  # switch to eval so BatchNorm/Dropout don't require batch statistics
with torch.no_grad():
    dummy_output = model(dummy_input)
model.train()  # restore training mode
print(f"  Input shape: {dummy_input.shape}")
print(f"  Output shape: {dummy_output.shape}")

# %% [code] {"id":"k1J-xVs13qEo","outputId":"bae44a48-3424-498b-9032-389d67bf791f","execution":{"iopub.status.busy":"2026-02-15T16:57:59.474453Z","iopub.execute_input":"2026-02-15T16:57:59.474731Z","iopub.status.idle":"2026-02-15T17:00:54.982308Z","shell.execute_reply.started":"2026-02-15T16:57:59.474704Z","shell.execute_reply":"2026-02-15T17:00:54.981597Z"}}
# Create dataloaders

from torch.utils.data import WeightedRandomSampler

def get_sampler(dataset):
    print("Computing sample weights for balancing...")
    sample_weights = []
    
    # Rare classes we want to boost (Indices from your mapping)
    # 6=Logs, 7=Rocks, 3=Dry Bushes
    RARE_CLASSES = [6, 7, 5] 
    
    for i in tqdm(range(len(dataset))):
        _, mask = dataset[i] # mask is tensor
        
        unique_classes = torch.unique(mask)
        
        weight = 1.0
        # If image contains a rare class, boost its weight!
        for cls in RARE_CLASSES:
            if cls in unique_classes:
                weight += 10.0  # Boost this image 10x
                
        sample_weights.append(weight)

    sample_weights = torch.DoubleTensor(sample_weights)
    
    # Create the sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True # Allows picking the same image multiple times
    )
    
    return sampler

# usage:
train_sampler = get_sampler(train_dataset)


train_loader = DataLoader(
    train_dataset, 
    batch_size=config.batch_size, 
    sampler=train_sampler, 
    shuffle=False,         
    num_workers=config.num_workers
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=True
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# Setup loss function
criterion = CombinedFocalDiceLoss(
    focal_weight=0.5, 
    dice_weight=0.5, 
    gamma=3.0  # Higher gamma (e.g., 3.0 or 4.0) forces focus on harder examples
)

# Setup optimizer and scheduler
optimizer = optim.Adam(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=5,
)

print("\n✓ Training setup complete")

# %% [code] {"id":"wbZx7yCE34sd","outputId":"274ba83c-da94-4b46-d778-5c06f49ec5d7","execution":{"iopub.status.busy":"2026-02-15T17:00:54.984953Z","iopub.execute_input":"2026-02-15T17:00:54.985189Z","iopub.status.idle":"2026-02-15T17:00:55.001565Z","shell.execute_reply.started":"2026-02-15T17:00:54.985167Z","shell.execute_reply":"2026-02-15T17:00:55.000799Z"}}
import torch.nn as nn

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None, grad_accum_steps=1):
    model.train()  # Set model to training mode

    # Explicitly set BatchNorm layers to eval mode
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            module.eval()

    running_loss = 0.0
    running_focal = 0.0
    running_dice = 0.0

    pbar = tqdm(train_loader, desc="Training")

    optimizer.zero_grad()

    for step, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Mixed precision if scaler provided
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(images)
            loss, focal ,dice = criterion(outputs, masks)
            loss = loss / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation step
        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        # Keep running sums (rescale loss back)
        running_loss += loss.item() * grad_accum_steps
        running_focal += focal.item()
        running_dice += dice.item()

        # Free references to help GPU memory
        del outputs
        torch.cuda.empty_cache()

        pbar.set_postfix({'loss': f"{running_loss/((step+1)):.4f}"})

    epoch_loss = running_loss / len(train_loader)
    epoch_focal = running_focal / len(train_loader)
    epoch_dice = running_dice / len(train_loader)

    return epoch_loss, epoch_focal, epoch_dice

import torch
import torch.nn.functional as F

# 1. Define the TTA Strategies
# Format: {'name': str, 'forward': func, 'inverse': func}
TTA_TRANSFORMS = [
    # A. Original Image (Baseline)
    {
        'name': 'Original',
        'forward': lambda x: x,
        'inverse': lambda x: x
    },

    # B. Horizontal Flip
    {
        'name': 'Horizontal Flip',
        'forward': lambda x: torch.flip(x, dims=[3]), 
        'inverse': lambda x: torch.flip(x, dims=[3])
    },

    # C. Vertical Flip (Excellent for top-down terrain)
    {
        'name': 'Vertical Flip',
        'forward': lambda x: torch.flip(x, dims=[2]), 
        'inverse': lambda x: torch.flip(x, dims=[2])
    },

    # D. Rotate 90 (Counter-Clockwise)
    {
        'name': 'Rotate 90',
        'forward': lambda x: torch.rot90(x, k=1, dims=[2, 3]),
        'inverse': lambda x: torch.rot90(x, k=-1, dims=[2, 3])
    }
]

# 2. The Prediction Function
def predict_with_tta(model, images):
    """
    Applies TTA to a batch of images:
    1. Augments the batch (flips/rotates).
    2. Runs the model.
    3. Inverts the augmentation on the prediction.
    4. Averages the results.
    """
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for tta in TTA_TRANSFORMS:
            # Apply Transform
            aug_images = tta['forward'](images)
            
            # Predict
            output = model(aug_images)
            probs = F.softmax(output, dim=1) # Convert to probability
            
            # Invert Transform (Flip the prediction back)
            original_oriented_probs = tta['inverse'](probs)
            
            all_probs.append(original_oriented_probs)
    
    # Average all predictions
    avg_probs = torch.stack(all_probs).mean(dim=0)
    return avg_probs


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0

    # Initialize global intersection and union counters
    total_intersection = torch.zeros(num_classes).to(device)
    total_union = torch.zeros(num_classes).to(device)

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)

            avg_probs = model(images)
            loss, _, _ = criterion(avg_probs, masks)
            total_loss += loss.item()

            # Get predictions
            predictions = torch.argmax(avg_probs, dim=1)

            # Calculate Intersection and Union for this batch
            # Flatten to 1D for easy calculation
            preds_flat = predictions.view(-1)
            masks_flat = masks.view(-1)

            for cls in range(num_classes):
                # Create boolean masks for the current class
                pred_cls = (preds_flat == cls)
                mask_cls = (masks_flat == cls)

                # Accumulate counts
                intersection = (pred_cls & mask_cls).sum()
                union = (pred_cls | mask_cls).sum()

                total_intersection[cls] += intersection
                total_union[cls] += union

    # --- END OF EPOCH CALCULATION ---

    # Calculate IoU globally (safe from batch poisoning)
    # Add small epsilon to avoid 0/0 if a class is truly missing from the WHOLE dataset
    epsilon = 1e-6
    class_iou = total_intersection / (total_union + epsilon)

    # Convert to numpy for logging
    class_iou = class_iou.cpu().numpy()
    mean_iou = np.mean(class_iou)

    return total_loss / len(loader), mean_iou, class_iou

print("✓ Training functions defined")

# %% [code] {"id":"EE_cnwjGdg1C","outputId":"6e216449-2c2a-4977-c86a-1579fecd8ba4","execution":{"iopub.status.busy":"2026-02-15T17:00:55.002554Z","iopub.execute_input":"2026-02-15T17:00:55.002801Z"}}
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60 + "\n")

# Check GPU availability (re-adding for robustness)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training state
if device.type == 'cuda':
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

best_iou = 0.0
train_losses = []
val_losses = []
val_ious = []


# Setup AMP scaler if enabled
scaler = None
if device.type == 'cuda' and getattr(config, 'use_amp', False):
    scaler = torch.cuda.amp.GradScaler()
    print("Using mixed precision (AMP)")

# Training loop
for epoch in range(config.num_epochs):
    print(f"\nEpoch {epoch+1}/{config.num_epochs}")
    print("-" * 60)

    try:
        # Train
        train_loss, train_focal, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, grad_accum_steps=getattr(config, 'grad_accum_steps', 1)
        )

        # Validate
        val_loss, mean_iou, class_iou = validate(model, val_loader, criterion, device, config.num_classes)

        # Update learning rate
        scheduler.step(mean_iou)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ious.append(mean_iou)

        # Print results
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} (Dice: {train_focal:.4f}, CE: {train_dice:.4f})")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val IoU: {mean_iou:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        print(f"\n  Per-class IoU:")
        for name, iou in zip(CLASS_NAMES, class_iou):
            if not np.isnan(iou):
                print(f"    {name:20s}: {iou:.4f}")

        # Save best model
        if mean_iou > best_iou:
            best_iou = mean_iou
            os.makedirs(config.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'config': config
            }, os.path.join(config.save_dir, 'best_model.pth'))
            print(f"\n  ✓ Saved best model (IoU: {best_iou:.4f})")

        # Early stopping
        if config.early_stopping > 0 and len(val_ious) > config.early_stopping:
            recent_ious = val_ious[-config.early_stopping:]
            if all(iou <= best_iou for iou in recent_ious):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

        # Plot curves every 5 epochs
        if (epoch + 1) % 5 == 0:
            fig = plot_training_curves(train_losses, val_losses, val_ious)
            plt.show()

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print('\nCUDA out of memory during training. Suggestions:')
            print(' - Reduce `config.batch_size` (current:', config.batch_size, ')')
            print(' - Reduce `config.img_size` (e.g., 320x320)')
            print(' - Set `config.use_amp = True` to enable mixed precision')
            print(' - Increase `config.grad_accum_steps` to simulate larger batches')
            print(' - Use a smaller backbone (mobilenet_v3)')
            torch.cuda.empty_cache()
            break
        else:
            raise

print("\n" + "="*60)
print(f"TRAINING COMPLETE - Best IoU: {best_iou:.4f}")
print("="*60)

# %% [code]
#Code block to save pytorch model after training online on kaggle
# import shutil
# from IPython.display import FileLink

# source_path = "/kaggle/working/checkpoints/best_model.pth" 

# output_filename = "my_model_backup"

# shutil.make_archive(output_filename, 'zip', root_dir='/kaggle/working')

# print("CLICK BELOW TO SAVE YOUR MODEL:")
# display(FileLink(f'{output_filename}.zip'))
torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'config': config
            }, "deep_lab_v3.pth")

# %% [code] {"id":"34r2NndPwUfA"}
import numpy as np
from PIL import Image
import os

# 1. Get a single mask file
mask_dir = os.path.join(config.train_dir, "train", "Segmentation")
# Pick a few files to be sure
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg'))])[:3]

print(f"Checking {len(mask_files)} masks for pixel values...\n")

for m_file in mask_files:
    path = os.path.join(mask_dir, m_file)
    raw_mask = np.array(Image.open(path))
    unique_vals = np.unique(raw_mask)

    print(f"File: {m_file}")
    print(f"  > Found Pixel Values: {unique_vals}")

    # Check if these values exist in your mapping
    matches = [v for v in unique_vals if v in CLASS_MAPPING]
    mismatches = [v for v in unique_vals if v not in CLASS_MAPPING]

    print(f"  > Matches in Map: {matches}")
    print(f"  > NOT in Map (Converted to Trees!): {mismatches}\n")

print("Expected Keys from your code:", list(CLASS_MAPPING.keys()))

# %% [code]
  # Load best model
  checkpoint_path = os.path.join(config.save_dir, 'best_model.pth')
  if os.path.exists(checkpoint_path):
      # Explicitly set weights_only=False to allow loading of numpy objects
      checkpoint = torch.load(checkpoint_path, weights_only=False)
      model.load_state_dict(checkpoint['model_state_dict'])
      print(f"Loaded best model (IoU: {checkpoint['best_iou']:.4f})")

  model.eval()

  # Visualize predictions
  print("\nVisualizing predictions on validation set...")

  num_samples = min(6, len(val_dataset))
  indices = np.random.choice(len(val_dataset), num_samples, replace=False)

  for idx in indices:
      image, mask = val_dataset[idx]

      # Predict
      with torch.no_grad():
          output = model(image.unsqueeze(0).to(device))
          prediction = torch.argmax(output, dim=1).squeeze(0)

      # Visualize
      fig = visualize_prediction(image, mask, prediction)
      plt.savefig(os.path.join(config.output_dir, f'val_prediction_{idx}.png'), dpi=150, bbox_inches='tight')
      plt.show()

      # Calculate IoU for this sample
      sample_iou, _ = calculate_iou(prediction.unsqueeze(0), mask.unsqueeze(0).to(device), config.num_classes)
      print(f"Sample {idx} IoU: {sample_iou:.4f}")
      print()

# %% [code]
import time

# Measure inference speed
model.eval()

dummy_input = torch.randn(1, 3, config.img_size[0], config.img_size[1]).to(device)

# Warmup
for _ in range(10):
    with torch.no_grad():
        _ = model(dummy_input)

# Measure
num_iterations = 100
start_time = time.time()

for _ in range(num_iterations):
    with torch.no_grad():
        _ = model(dummy_input)

if device.type == 'cuda':
    torch.cuda.synchronize()

end_time = time.time()

avg_time = (end_time - start_time) / num_iterations * 1000  # ms

print("\nInference Speed:")
print("="*40)
print(f"Average time per image: {avg_time:.2f} ms")
print(f"FPS: {1000/avg_time:.2f}")

if avg_time < 50:
    print("✓ Meets the <50ms requirement!")
else:
    print("⚠️  Slower than 50ms target")
    print("Consider using MobileNetV3 or smaller image size")
print("="*40)