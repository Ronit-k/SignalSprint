"""
Generate model.pkl with CLIP Safety Net
=========================================
Bundles: YOLO + Swin ViT + CLIP vision encoder + pre-computed text embeddings

The CLIP text encoder is ONLY used here (at bundle time) to pre-compute
text embeddings for our dustbin-detection prompts. At inference, only the
CLIP vision encoder runs — no text encoder needed, no HuggingFace download.
"""

import pickle
import os
import io
import torch
from transformers import CLIPModel, CLIPTokenizer, CLIPConfig

# ── Paths ───────────────────────────────────────────
yolo_path = "models/best_yolo_v2.pt"
vit_path  = "models/best_vit_v6.pth"

for p in [yolo_path, vit_path]:
    if not os.path.exists(p):
        print(f"ERROR: {p} not found!")
        exit(1)

# ── 1. Read YOLO bytes ──────────────────────────────
with open(yolo_path, "rb") as f:
    yolo_bytes = f.read()
print(f"✅ YOLO:  {len(yolo_bytes) / 1e6:.1f} MB")

# ── 2. Read ViT bytes ───────────────────────────────
with open(vit_path, "rb") as f:
    vit_bytes = f.read()
print(f"✅ ViT:   {len(vit_bytes) / 1e6:.1f} MB")

# ── 3. Swin config (air-gap safe) ───────────────────
vit_config_dict = {
    "architectures": ["SwinForImageClassification"],
    "depths": [2, 2, 18, 2],
    "drop_path_rate": 0.1,
    "embed_dim": 128,
    "hidden_act": "gelu",
    "hidden_size": 1024,
    "id2label": {"0": "action_required", "1": "no_action"},
    "image_size": 224,
    "label2id": {"action_required": 0, "no_action": 1},
    "model_type": "swin",
    "num_channels": 3,
    "num_heads": [4, 8, 16, 32],
    "num_layers": 4,
    "patch_size": 4,
    "window_size": 7
}

# ── 4. CLIP: Load, pre-compute text embeddings, bundle ──
print("\n🧠 Loading CLIP ViT-B/32 for text embedding pre-computation...")
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name, use_safetensors=True)
clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
clip_model.eval()

# Positive prompts: things that ARE authorized dustbins
positive_texts = [
    "a dustbin",
    "a garbage bin",
    "a trash can",
    "a waste bin",
    "a blue dustbin on the ground",
    "a green dustbin on the ground",
    "a cyan dustbin",
    "a metal cage dustbin",
    "a metal stand elevated dustbin",
    "a dustbin with garbage around it",
    "a trash bin with spilled garbage",
    "an overflowing dustbin",
]

# Negative prompts: things that are NOT dustbins
negative_texts = [
    "an empty road",
    "a wall",
    "a building",
    "trees and grass",
    "a red postbox",
    "a poster on a wall",
    "a park bench",
    "a flower pot",
    "a water drum",
    "people walking",
    "a parked vehicle",
    "fallen dry leaves on ground",
]

n_positive = len(positive_texts)
all_texts = positive_texts + negative_texts

print(f"   Pre-computing embeddings for {len(all_texts)} prompts ({n_positive} positive)...")

inputs = clip_tokenizer(all_texts, padding=True, return_tensors="pt")
with torch.no_grad():
    text_output = clip_model.text_model(**inputs)
    text_features = clip_model.text_projection(text_output.pooler_output)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

print(f"   Text features shape: {text_features.shape}")

# Save CLIP vision model weights to bytes
clip_vision_buf = io.BytesIO()
torch.save(clip_model.vision_model.state_dict(), clip_vision_buf)
clip_vision_bytes = clip_vision_buf.getvalue()
print(f"✅ CLIP vision encoder: {len(clip_vision_bytes) / 1e6:.1f} MB")

# Save CLIP visual projection weights
clip_proj_buf = io.BytesIO()
torch.save(clip_model.visual_projection.state_dict(), clip_proj_buf)
clip_proj_bytes = clip_proj_buf.getvalue()

# Save CLIP config
clip_config_dict = clip_model.config.to_dict()

# ── 5. Bundle everything ────────────────────────────
data = {
    "yolo": yolo_bytes,
    "vit": vit_bytes,
    "vit_config": vit_config_dict,
    # CLIP components
    "clip_vision": clip_vision_bytes,
    "clip_projection": clip_proj_bytes,
    "clip_config": clip_config_dict,
    "clip_text_features": text_features,  # pre-computed, shape [24, 512]
    "clip_n_positive": n_positive,
}

output_path = "export/model.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "wb") as f:
    pickle.dump(data, f)

total_mb = (len(yolo_bytes) + len(vit_bytes) + len(clip_vision_bytes)) / 1e6
print(f"\n🏆 Generated {output_path}  (~{total_mb:.0f} MB)")
print(f"   CLIP text embeddings: {text_features.shape}")
print(f"   Positive prompts: {n_positive}, Negative prompts: {len(negative_texts)}")
