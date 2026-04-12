"""
2-Stage SOTA Predict Pipeline + CLIP Safety Net
=================================================
Stage 1 — DMC-Gate: Custom YOLO detects bins, filters traps, extracts halo crops

  ┌─ Bins found ──→ Stage 2 (ViT + TTA) ──→ decision
  └─ No bins ────→ CLIP Safety Net (zero-shot bin detection)
                     ├─ "dustbin present" → Full image → ViT → decision
                     └─ "no dustbin"      → return 0

Stage 2 — Contextual Observer: Swin Transformer + TTA + softmax thresholding

CLIP Safety Net catches YOLO false negatives by computing cosine similarity
between the image and pre-computed text embeddings for dustbin vs non-dustbin
prompts. Only the CLIP vision encoder runs at inference (text embeddings
are pre-baked into model.pkl).
"""

import os
import io
import cv2
import pickle
import tempfile
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import SwinConfig, SwinForImageClassification, CLIPVisionModel, CLIPConfig
from ultralytics import YOLO

# ──────────────────────────────────────────────
# HYPERPARAMETERS
# ──────────────────────────────────────────────
YOLO_CONF           = 0.25   # Stage 1 YOLO confidence
ACTION_THRESHOLD    = 0.3    # ViT softmax P(action) must exceed this
TTA_ENABLED         = True   # Test-Time Augmentation
CLIP_BIN_THRESHOLD  = 0.1    # CLIP: min (pos - neg) similarity gap to confirm bin presence

CLASS_NAMES = {
    0: "bin_caged",
    1: "bin_elevated",
    2: "bin_ground",
    3: "trap_object"
}

# CLIP image preprocessing (matches openai/clip-vit-base-patch32 training)
CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
])


def load_model():
    """Load YOLO + ViT + CLIP from model.pkl."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_pkl_path = os.path.join(current_dir, "model.pkl")

    with open(model_pkl_path, "rb") as f:
        data = pickle.load(f)

    device = torch.device("cpu")

    # ── Stage 1: YOLO ─────────────────────────────
    yolo_temp_path = os.path.join(tempfile.gettempdir(), 'temp_yolo.pt')
    with open(yolo_temp_path, 'wb') as f:
        f.write(data['yolo'])
    yolo_model = YOLO(yolo_temp_path)

    # ── Stage 2: Swin Transformer ─────────────────
    vit_config = SwinConfig.from_dict(data['vit_config'])
    vit_model = SwinForImageClassification(vit_config)
    vit_bytes_io = io.BytesIO(data['vit'])
    vit_model.load_state_dict(torch.load(vit_bytes_io, map_location=device))
    vit_model.to(device)
    vit_model.eval()

    vit_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ── CLIP Safety Net ───────────────────────────
    clip_config = CLIPConfig.from_dict(data['clip_config'])
    clip_vision = CLIPVisionModel(clip_config.vision_config)
    clip_vision_bytes = io.BytesIO(data['clip_vision'])
    clip_vision.vision_model.load_state_dict(torch.load(clip_vision_bytes, map_location=device))
    clip_vision.to(device)
    clip_vision.eval()

    # Load visual projection layer (maps vision features → shared CLIP space)
    clip_proj_bytes = io.BytesIO(data['clip_projection'])
    clip_projection = torch.nn.Linear(
        clip_config.vision_config.hidden_size,
        clip_config.projection_dim,
        bias=False
    )
    clip_projection.load_state_dict(torch.load(clip_proj_bytes, map_location=device))
    clip_projection.to(device)
    clip_projection.eval()

    # Pre-computed text embeddings [n_prompts, 512]
    clip_text_features = data['clip_text_features'].to(device)
    clip_n_positive = data['clip_n_positive']

    return {
        "yolo": yolo_model,
        "vit": vit_model,
        "device": device,
        "vit_transform": vit_transform,
        "clip_vision": clip_vision,
        "clip_projection": clip_projection,
        "clip_text_features": clip_text_features,
        "clip_n_positive": clip_n_positive,
    }


# ──────────────────────────────────────────────
# CLIP Safety Net
# ──────────────────────────────────────────────
def _clip_detects_bin(models, bgr_img):
    """
    Zero-shot check: does this image contain a dustbin?
    Computes cosine similarity between the image and pre-computed
    positive (dustbin) vs negative (non-dustbin) text embeddings.
    Returns True if positive prompts score significantly higher.
    """
    clip_vision     = models["clip_vision"]
    clip_projection = models["clip_projection"]
    text_features   = models["clip_text_features"]
    n_pos           = models["clip_n_positive"]
    device          = models["device"]

    # BGR → RGB → PIL → CLIP transform
    img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pixel_values = CLIP_TRANSFORM(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        vision_output = clip_vision(pixel_values=pixel_values)
        # Use the [CLS] token output (pooler_output)
        pooled = vision_output.pooler_output
        # Project to shared embedding space
        image_features = clip_projection(pooled)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Cosine similarity with all text embeddings
        similarities = (image_features @ text_features.T).squeeze(0)  # [n_prompts]

    # Average similarity to positive vs negative prompts
    pos_sim = similarities[:n_pos].mean().item()
    neg_sim = similarities[n_pos:].mean().item()

    # The gap determines confidence that a dustbin is present
    gap = pos_sim - neg_sim
    return gap > CLIP_BIN_THRESHOLD


# ──────────────────────────────────────────────
# TTA
# ──────────────────────────────────────────────
def _tta_transforms(pil_img):
    views = [pil_img]
    views.append(pil_img.transpose(Image.FLIP_LEFT_RIGHT))
    views.append(pil_img.transpose(Image.FLIP_TOP_BOTTOM))
    w, h = pil_img.size
    for ratio in (0.85, 0.90):
        dw = int(w * (1 - ratio) / 2)
        dh = int(h * (1 - ratio) / 2)
        views.append(pil_img.crop((dw, dh, w - dw, h - dh)))
    return views


def _get_action_probability(vit_model, pil_img, transform, device, use_tta=True):
    views = _tta_transforms(pil_img) if use_tta else [pil_img]
    all_probs = []
    for view in views:
        tensor = transform(view).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = vit_model(tensor).logits
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs[0, 0].item())
    return sum(all_probs) / len(all_probs)


# ──────────────────────────────────────────────
# Main predict
# ──────────────────────────────────────────────
def predict(models, image_path):
    """
    Full pipeline: YOLO → (CLIP fallback if no bins) → ViT → decision.
    Returns 1 (intimate DMC) or 0 (no action).
    """
    yolo_model    = models["yolo"]
    vit_model     = models["vit"]
    device        = models["device"]
    vit_transform = models["vit_transform"]

    # ── STAGE 1: YOLO ─────────────────────────────
    results = yolo_model.predict(source=image_path, conf=YOLO_CONF, verbose=False)
    result  = results[0]

    original_img = result.orig_img
    if original_img is None:
        original_img = cv2.imread(image_path)
    if original_img is None:
        return 0

    img_h, img_w = original_img.shape[:2]
    vit_inputs = []

    for box in result.boxes:
        class_id   = int(box.cls[0].item())
        class_name = CLASS_NAMES.get(class_id, "unknown")

        if class_name == "trap_object":
            continue

        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        bbox_w, bbox_h = x2 - x1, y2 - y1
        pad_x, pad_y = bbox_w * 1.5, bbox_h * 1.5

        crop_x1 = int(max(0,     x1 - pad_x))
        crop_y1 = int(max(0,     y1 - pad_y))
        crop_x2 = int(min(img_w, x2 + pad_x))
        crop_y2 = int(min(img_h, y2 + pad_y))

        crop = original_img[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size > 0:
            vit_inputs.append(crop)

    # ── CLIP SAFETY NET: catch YOLO false negatives ──
    if not vit_inputs:
        # YOLO found nothing. Ask CLIP: "is there really a dustbin here?"
        if _clip_detects_bin(models, original_img):
            # CLIP says YES — YOLO missed it. Send full image to ViT.
            vit_inputs = [original_img]
        else:
            # CLIP agrees: no dustbin. Safe to return 0.
            return 0

    # ── STAGE 2: ViT + TTA + Confidence Threshold ──
    action_probs = []
    for crop in vit_inputs:
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        prob = _get_action_probability(
            vit_model, pil_img, vit_transform, device, use_tta=TTA_ENABLED
        )
        action_probs.append(prob)

    # ── AGGREGATION ───────────────────────────────
    max_prob = max(action_probs)
    if max_prob > ACTION_THRESHOLD:
        return 1
    return 0
