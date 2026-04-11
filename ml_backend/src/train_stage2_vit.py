"""
SOTA Stage-2 ViT Training Pipeline
====================================
Key improvements over original:
  1. TransformSubset wrapper — each split gets its own transforms (fixes the
     bug where train augmentation leaked into val images)
  2. Heavy augmentation: RandomResizedCrop, Perspective, GaussianBlur,
     Affine, Grayscale, Erasing — simulates real-world phone photos
  3. WeightedRandomSampler — counteracts the 76:300 class imbalance
  4. Label smoothing (0.1) — prevents overconfident preds on edge cases
  5. Cosine Annealing LR — better convergence on small datasets
  6. Gradient clipping (max_norm=1.0) — stabilises tiny-batch training
  7. Batch size 16 (or 8 if OOM) — less noisy gradients than BS=1
  8. Mixed precision (if CUDA) — 2× speed on GPU
  9. Early stopping — patience-based to avoid overfitting
"""

import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset, WeightedRandomSampler
from transformers import SwinForImageClassification
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/vit_dataset"))
RUNS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../runs/vit"))

IMG_SIZE       = 224
BATCH_SIZE     = 16        # Raised from 1; use 8 if OOM
EPOCHS         = 30
LEARNING_RATE  = 2e-5
WEIGHT_DECAY   = 0.01
LABEL_SMOOTH   = 0.1       # Label smoothing factor
GRAD_CLIP      = 1.0       # Max gradient norm
PATIENCE       = 7         # Early stopping patience (epochs without val improvement)
TRAIN_SPLIT    = 0.8


# ==========================================
# TRANSFORM SUBSET WRAPPER
# ==========================================
class TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset so each split can have its own transforms
    without mutating the parent dataset."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        # Expose indices for WeightedRandomSampler compatibility
        self.indices = subset.indices
        self.dataset = subset.dataset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# ==========================================
# AUGMENTATION PIPELINES
# ==========================================
train_transforms = transforms.Compose([
    # --- Geometric ---
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(degrees=20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    # --- Color ---
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    # --- Tensor ---
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),  # Cutout-style
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ==========================================
# HELPERS
# ==========================================
def create_run_directory():
    os.makedirs(RUNS_DIR, exist_ok=True)
    run_num = 1
    while os.path.exists(os.path.join(RUNS_DIR, f"train{run_num if run_num > 1 else ''}")):
        run_num += 1
    run_name = f"train{run_num if run_num > 1 else ''}"
    run_path = os.path.join(RUNS_DIR, run_name)
    weights_path = os.path.join(run_path, "weights")
    os.makedirs(weights_path, exist_ok=True)
    return run_path, weights_path


def build_weighted_sampler(dataset):
    """Build a WeightedRandomSampler to over-sample the minority class."""
    # Collect all labels from the underlying subset indices
    labels = []
    for idx in dataset.indices:
        _, label = dataset.dataset[idx]
        labels.append(label)
    labels = torch.tensor(labels)

    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


# ==========================================
# MAIN
# ==========================================
def main():
    print("🚀 Initializing SOTA Stage 2 (ViT) Training Pipeline...\n")

    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler() if use_amp else None
    print(f"💻 Device: {device}  |  Mixed Precision: {use_amp}")

    run_path, weights_path = create_run_directory()
    print(f"📁 Output: {run_path}\n")

    # 2. Dataset — load WITHOUT transforms (TransformSubset applies them)
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Dataset not found at {DATA_DIR}")

    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=None)
    class_names = full_dataset.classes
    n_total = len(full_dataset)
    train_size = int(TRAIN_SPLIT * n_total)
    val_size = n_total - train_size

    # Fixed seed split for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Wrap with per-split transforms
    train_dataset = TransformSubset(train_subset, train_transforms)
    val_dataset   = TransformSubset(val_subset, val_transforms)

    # Class-balanced sampler for training
    sampler = build_weighted_sampler(train_subset)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Print class distribution
    all_labels = [full_dataset.targets[i] for i in train_subset.indices]
    for i, name in enumerate(class_names):
        count = all_labels.count(i)
        print(f"  {name}: {count} images")
    print(f"✅ Train: {train_size} | Val: {val_size}\n")

    # 3. Model
    print("🧠 Loading Swin-Base Transformer...")
    model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-base-patch4-window7-224",
        num_labels=2,
        id2label={0: "action_required", 1: "no_action"},
        label2id={"action_required": 0, "no_action": 1},
        ignore_mismatched_sizes=True
    )
    model.to(device)

    # 4. Optimizer, Scheduler, Loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    # 5. Training Loop
    best_acc = 0.0
    patience_counter = 0
    start_time = time.time()

    print("\n🔥 Starting Training...\n")

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}  (LR: {scheduler.get_last_lr()[0]:.2e})")
        print("-" * 40)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            pbar = tqdm(dataloader, desc=f"  {phase.capitalize():>5}", leave=False)

            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch_n = inputs.size(0)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if use_amp and phase == 'train':
                        with torch.amp.autocast(device_type='cuda'):
                            logits = model(inputs).logits
                            loss = criterion(logits, labels)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        logits = model(inputs).logits
                        loss = criterion(logits, labels)
                        if phase == 'train':
                            loss.backward()
                            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                            optimizer.step()

                _, preds = torch.max(logits, 1)
                running_loss += loss.item() * batch_n
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += batch_n

                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples

            print(f"  [{phase.upper()}] Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            # Save best + early stopping
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    patience_counter = 0
                    best_path = os.path.join(weights_path, "best.pth")
                    torch.save(model.state_dict(), best_path)
                    print(f"  ⭐ New Best Val Acc! Saved to best.pth")
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print(f"\n⏹️  Early stopping triggered (no improvement for {PATIENCE} epochs)")
                        break

        # Step the scheduler
        scheduler.step()

        if patience_counter >= PATIENCE:
            break

        print()

    # Save last
    last_path = os.path.join(weights_path, "last.pth")
    torch.save(model.state_dict(), last_path)

    elapsed = time.time() - start_time
    print("\n" + "=" * 40)
    print(f"🎉 Done in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"🏆 Best Val Acc: {best_acc:.4f}")
    print(f"💾 Weights: {weights_path}")
    print("=" * 40)


if __name__ == "__main__":
    main()