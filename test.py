"""
Submission / Inference Script
Adapted for SegFormer B1 (Student Model)
Generates predictions in the format required for the Hackathon.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import argparse
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

# ============================================================================
# Configuration & Color Palette
# ============================================================================

# The Hackathon's Standard Class IDs (Target Output)
# 0: Background, 1: Trees, 2: Lush Bushes, ...
TARGET_PALETTE = np.array([
    [0, 0, 0],        # 0: Background
    [34, 139, 34],    # 1: Trees
    [0, 255, 0],      # 2: Lush Bushes
    [210, 180, 140],  # 3: Dry Grass
    [139, 90, 43],    # 4: Dry Bushes
    [128, 128, 0],    # 5: Ground Clutter
    [139, 69, 19],    # 6: Logs
    [128, 128, 128],  # 7: Rocks
    [160, 82, 45],    # 8: Landscape
    [135, 206, 235],  # 9: Sky
], dtype=np.uint8)

def mask_to_color(mask):
    """Convert a class mask (0-9) to a colored RGB image for visualization."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(10):
        color_mask[mask == class_id] = TARGET_PALETTE[class_id]
    return color_mask

# ============================================================================
# Dataset (Inference Only)
# ============================================================================

class InferenceDataset(Dataset):
    def __init__(self, data_dir, img_size=(512, 512)):
        # Handle both structure types: 'data_dir/Color_Images' OR just 'data_dir'
        if os.path.exists(os.path.join(data_dir, 'Color_Images')):
            self.image_dir = os.path.join(data_dir, 'Color_Images')
        else:
            self.image_dir = data_dir
            
        self.files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg'))])
        
        # SegFormer-specific normalization (ImageNet stats)
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        img_path = os.path.join(self.image_dir, file_name)
        
        # Load and convert to RGB
        image = Image.open(img_path).convert("RGB")
        original_size = image.size # (W, H)
        
        # Apply transforms
        input_tensor = self.transform(image)
        
        return input_tensor, file_name, original_size

# ============================================================================
# Main Inference Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='SegFormer Submission Script')
    parser.add_argument('--model_path', type=str, required=True,default="best_elite_model.pth" help='Path to your .pth file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to test images folder', default = )
    parser.add_argument('--output_dir', type=str, default='./submission_output', help='Where to save results')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'masks_color'), exist_ok=True)

    # 2. Load Model (SegFormer B1)
    print(f"Loading SegFormer B1 from {args.model_path}...")
    
    # Initialize architecture
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b1", 
        num_labels=10, 
        ignore_mismatched_sizes=True
    )
    
    # Load Weights
    checkpoint = torch.load(args.model_path, map_location=device)
    # Handle 'model_state_dict' wrapper if present
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()

    # 3. Data Loader
    test_set = InferenceDataset(args.data_dir)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Found {len(test_set)} images. Starting inference...")

    # 4. Inference Loop
    with torch.no_grad():
        for images, file_names, original_sizes in tqdm(test_loader):
            images = images.to(device)
            
            # Predict
            outputs = model(images)
            logits = outputs.logits # (B, 10, 128, 128)
            
            # Iterate batch
            for i in range(len(file_names)):
                # Resize logits to original image size
                w, h = original_sizes[0][i].item(), original_sizes[1][i].item()
                
                logit = logits[i].unsqueeze(0) # (1, 10, 128, 128)
                logit = F.interpolate(logit, size=(h, w), mode='bilinear', align_corners=False)
                
                pred = torch.argmax(logit, dim=1).squeeze().cpu().numpy().astype(np.uint8)
                
                # --- CRITICAL: REMAP CLASSES ---
                # Your model trained with 0=Trees, but Hackathon expects 1=Trees
                # We apply the shift based on your training mapping:
                # Model 0 (Trees)      -> Output 1
                # Model 1 (Lush Bush)  -> Output 2
                # Model 2 (Dry Grass)  -> Output 3
                # Model 3 (Dry Bush)   -> Output 4
                # Model 8 (Landscape)  -> Output 8 (No change)
                # Model 9 (Sky)        -> Output 9 (No change)
                
                final_mask = np.zeros_like(pred)
                final_mask[pred == 0] = 1
                final_mask[pred == 1] = 2
                final_mask[pred == 2] = 3
                final_mask[pred == 3] = 4
                final_mask[pred == 8] = 8
                final_mask[pred == 9] = 9
                # Classes 4,5,6,7 were missing in training, so they won't appear
                
                # Save Raw Mask (0-9 IDs)
                base_name = os.path.splitext(file_names[i])[0]
                save_path = os.path.join(args.output_dir, 'masks', f'{base_name}.png')
                cv2.imwrite(save_path, final_mask)
                
                # Save Color Visualization
                color_viz = mask_to_color(final_mask)
                cv2.imwrite(
                    os.path.join(args.output_dir, 'masks_color', f'{base_name}_color.png'),
                    cv2.cvtColor(color_viz, cv2.COLOR_RGB2BGR)
                )

    print(f"\nDone! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

