import pickle
import os

print("Packaging model.pkl with models/", yolo_path, "and models/", vit_path)

yolo_path = "models/best_yolo_v2.pt"
vit_path = "models/best_vit_v5.pth"
print ("using", yolo_path, "and", vit_path)

if not os.path.exists(yolo_path):
    print(f"ERROR: {yolo_path} not found!")
    exit(1)

if not os.path.exists(vit_path):
    print(f"ERROR: {vit_path} not found!")
    exit(1)

with open(yolo_path, "rb") as f:
    yolo_bytes = f.read()

with open(vit_path, "rb") as f:
    vit_bytes = f.read()

config_dict = {
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

data = {
    "yolo": yolo_bytes,
    "vit": vit_bytes,
    "vit_config": config_dict
}

output_path = "export/model.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "wb") as f:
    pickle.dump(data, f)
print(f"Successfully generated {output_path}")
