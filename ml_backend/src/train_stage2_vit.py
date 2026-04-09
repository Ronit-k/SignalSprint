import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from transformers import SwinForImageClassification
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
# Paths relative to the 'src' folder
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/vit_dataset"))
RUNS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../runs/vit"))

IMG_SIZE = 224
BATCH_SIZE = 16    # Reduce to 8 if you hit a CUDA Out of Memory error
EPOCHS = 15
LEARNING_RATE = 2e-5

def create_run_directory():
    """Creates a YOLO-style run directory (e.g., runs/vit/train/weights)"""
    os.makedirs(RUNS_DIR, exist_ok=True)
    
    # Find the next run number
    run_num = 1
    while os.path.exists(os.path.join(RUNS_DIR, f"train{run_num if run_num > 1 else ''}")):
        run_num += 1
        
    run_name = f"train{run_num if run_num > 1 else ''}"
    run_path = os.path.join(RUNS_DIR, run_name)
    weights_path = os.path.join(run_path, "weights")
    
    os.makedirs(weights_path, exist_ok=True)
    return run_path, weights_path

def main():
    print("🚀 Initializing Stage 2 (HTD-ViT) Training Pipeline...\n")
    
    # 1. Setup Device & Directories
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Compute Device: {device}")
    
    run_path, weights_path = create_run_directory()
    print(f"📁 Output Directory: {run_path}\n")

    # 2. Data Augmentation & Loading
    print("🔄 Preparing Data Loaders...")
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Dataset not found at {DATA_DIR}. Please run extraction first.")

    full_dataset = datasets.ImageFolder(root=DATA_DIR)
    class_names = full_dataset.classes
    print(f"✅ Found Classes: {class_names}")

    # 80/20 Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply transforms
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"✅ Train Setup: {train_size} images | Val Setup: {val_size} images\n")

    # 3. Model Initialization (Swin Transformer)
    print("🧠 Downloading/Loading Swin Transformer...")
    model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-base-patch4-window7-224",
        num_labels=2,
        id2label={0: "action_required", 1: "no_action"},
        label2id={"action_required": 0, "no_action": 1},
        ignore_mismatched_sizes=True
    )
    model.to(device)

    # 4. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    best_acc = 0.0
    start_time = time.time()

    print("\n🔥 Starting Training Loop...\n")

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print("-" * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Set up the progress bar
            pbar = tqdm(dataloader, desc=f"{phase.capitalize():>5} Phase", leave=False)
            
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).logits
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar with live loss
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"[{phase.upper()}] Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            # Save the best model logic
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_path = os.path.join(weights_path, "best.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"⭐ New Best Validation Accuracy! Weights saved to best.pth")

        print("") # Empty line between epochs

    # Save the final epoch weights as last.pth
    last_model_path = os.path.join(weights_path, "last.pth")
    torch.save(model.state_dict(), last_model_path)
    
    time_elapsed = time.time() - start_time
    print("=" * 40)
    print(f"🎉 Training Complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"🏆 Best Validation Accuracy: {best_acc:.4f}")
    print(f"💾 Models saved to: {weights_path}")
    print("=" * 40)

if __name__ == "__main__":
    main()