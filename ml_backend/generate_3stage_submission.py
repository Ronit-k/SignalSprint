import pickle
import os

print("Packaging model_3stage.pkl with:")
print("  Stage 1: models/best_yolo_v2.pt")
print("  Stage 2: models/yolov8m.pt")
print("  Stage 3: models/best_vit_v4.pth")
print()

yolo_stage1_path = "models/best_yolo_v2.pt"
yolo_stage2_path = "models/yolov8m.pt"
vit_path = "models/best_vit_v4.pth"

# --- Validation ---
for path in [yolo_stage1_path, yolo_stage2_path, vit_path]:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found!")
        exit(1)

# --- Read raw bytes for each model ---
with open(yolo_stage1_path, "rb") as f:
    yolo_stage1_bytes = f.read()
print(f"✅ Loaded Stage 1 YOLO:  {len(yolo_stage1_bytes) / 1e6:.1f} MB")

with open(yolo_stage2_path, "rb") as f:
    yolo_stage2_bytes = f.read()
print(f"✅ Loaded Stage 2 YOLO:  {len(yolo_stage2_bytes) / 1e6:.1f} MB")

with open(vit_path, "rb") as f:
    vit_bytes = f.read()
print(f"✅ Loaded Stage 3 ViT:   {len(vit_bytes) / 1e6:.1f} MB")

# --- Swin-Base config (air-gap safe: no HuggingFace download at inference time) ---
vit_config_dict = {
    "architectures": ["SwinForImageClassification"],
    "depths": [2, 2, 18, 2],
    "drop_path_rate": 0.1,
    "embed_dim": 128,
    "hidden_act": "gelu",
    "hidden_size": 1024,
    "id2label": {
        "0": "action_required",
        "1": "no_action"
    },
    "image_size": 224,
    "label2id": {
        "action_required": 0,
        "no_action": 1
    },
    "model_type": "swin",
    "num_channels": 3,
    "num_heads": [4, 8, 16, 32],
    "num_layers": 4,
    "patch_size": 4,
    "window_size": 7
}

# --- Bundle ---
data = {
    "yolo_stage1": yolo_stage1_bytes,   # Custom DMC-Gate detector
    "yolo_stage2": yolo_stage2_bytes,   # COCO context annotator
    "vit": vit_bytes,                   # Fine-tuned Swin classifier
    "vit_config": vit_config_dict
}

output_path = "export/model.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "wb") as f:
    pickle.dump(data, f)

total_mb = (len(yolo_stage1_bytes) + len(yolo_stage2_bytes) + len(vit_bytes)) / 1e6
print(f"\n🏆 Successfully generated {output_path}  ({total_mb:.1f} MB total)")
