"""
2-Stage SOTA Predict Pipeline
==============================
Stage 1 — DMC-Gate Detector: Custom YOLOv8 (best_yolo_v2.pt)
           Detects bins, filters traps, extracts 150% halo crops

Stage 2 — Contextual Observer: Fine-tuned Swin Transformer
           Classifies each crop → action_required / no_action
           with softmax confidence thresholding + Test-Time Augmentation (TTA)

Aggregation — Confidence-gated MAX rule with TTA-averaged probabilities
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
from transformers import SwinConfig, SwinForImageClassification
from ultralytics import YOLO

# ──────────────────────────────────────────────
# HYPERPARAMETERS (tune these to dial in accuracy)
# ──────────────────────────────────────────────
YOLO_CONF         = 0.25   # Raised from 0.1 to filter noisy detections
ACTION_THRESHOLD   = 0.3   # Softmax P(action) must exceed this to flag
TTA_ENABLED        = True   # Test-Time Augmentation for robustness
PLOT_EVERY_N       = 33     # Auto-save confidence plot every N predictions (0 = disabled)
PLOT_SAVE_PATH     = "confidence_plot.png"

# ── Internal state for auto-plotting (do not edit) ────────
_prediction_log = []  # accumulates (filename, confidence, decision)

CLASS_NAMES = {
    0: "bin_caged",
    1: "bin_elevated",
    2: "bin_ground",
    3: "trap_object"
}


def load_model():
    """Load all models from model.pkl (expected in same directory)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_pkl_path = os.path.join(current_dir, "model.pkl")

    with open(model_pkl_path, "rb") as f:
        data = pickle.load(f)

    # Stage 1: YOLO
    yolo_temp_path = os.path.join(tempfile.gettempdir(), 'temp_yolo.pt')
    with open(yolo_temp_path, 'wb') as f:
        f.write(data['yolo'])
    yolo_model = YOLO(yolo_temp_path)

    # Stage 2: Swin Transformer (air-gap safe via bundled config)
    device = torch.device("cpu")
    vit_config = SwinConfig.from_dict(data['vit_config'])
    vit_model = SwinForImageClassification(vit_config)

    vit_bytes_io = io.BytesIO(data['vit'])
    vit_model.load_state_dict(torch.load(vit_bytes_io, map_location=device))
    vit_model.to(device)
    vit_model.eval()

    # Pre-build transforms once (not per-image)
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return {
        "yolo": yolo_model,
        "vit": vit_model,
        "device": device,
        "transform": base_transform,
    }


def _tta_transforms(pil_img):
    """
    Test-Time Augmentation: generate multiple deterministic views of the crop.
    Returns a list of PIL images to run through the ViT independently.
    The predictions are averaged for a smoother, more robust score.
    """
    views = [pil_img]                                            # 1. Original
    views.append(pil_img.transpose(Image.FLIP_LEFT_RIGHT))       # 2. H-flip
    views.append(pil_img.transpose(Image.FLIP_TOP_BOTTOM))       # 3. V-flip

    # 4-5. Slight crops (centre 85% and 90%) — simulates scale jitter
    w, h = pil_img.size
    for ratio in (0.85, 0.90):
        dw = int(w * (1 - ratio) / 2)
        dh = int(h * (1 - ratio) / 2)
        cropped = pil_img.crop((dw, dh, w - dw, h - dh))
        views.append(cropped)

    return views


def _get_action_probability(vit_model, pil_img, transform, device, use_tta=True):
    """
    Run the ViT on a single crop and return P(action_required).
    If TTA is enabled, averages softmax probabilities across augmented views.
    """
    if use_tta:
        views = _tta_transforms(pil_img)
    else:
        views = [pil_img]

    all_probs = []
    for view in views:
        tensor = transform(view).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = vit_model(tensor).logits
            probs = F.softmax(logits, dim=1)
            action_prob = probs[0, 0].item()  # class 0 = action_required
        all_probs.append(action_prob)

    # Average across TTA views
    return sum(all_probs) / len(all_probs)


def _log_and_return(image_path, decision, confidence):
    """Log the prediction and trigger auto-plot if counter hit. Always returns decision."""
    if PLOT_EVERY_N > 0:
        img_name = os.path.basename(image_path)
        _prediction_log.append((img_name, confidence, decision))
        if len(_prediction_log) >= PLOT_EVERY_N:
            _save_confidence_plot()
    return decision


def predict(models, image_path):
    """
    Full 2-stage pipeline on a single image.
    Returns 1 (intimate DMC) or 0 (no action).
    """
    yolo_model = models["yolo"]
    vit_model  = models["vit"]
    device     = models["device"]
    transform  = models["transform"]

    # ── STAGE 1: DMC-Gate Selective Detection ──────────────────
    results = yolo_model.predict(source=image_path, conf=YOLO_CONF, verbose=False)
    result  = results[0]

    original_img = result.orig_img
    if original_img is None:
        original_img = cv2.imread(image_path)
    if original_img is None:
        return _log_and_return(image_path, 0, 0.0)

    img_h, img_w = original_img.shape[:2]
    vit_inputs = []

    for box in result.boxes:
        class_id   = int(box.cls[0].item())
        class_name = CLASS_NAMES.get(class_id, "unknown")

        # SINK LOGIC: skip trap objects
        if class_name == "trap_object":
            continue

        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # 150% Halo (kept as-is per user request)
        pad_x = bbox_w * 1.5
        pad_y = bbox_h * 1.5

        crop_x1 = int(max(0,     x1 - pad_x))
        crop_y1 = int(max(0,     y1 - pad_y))
        crop_x2 = int(min(img_w, x2 + pad_x))
        crop_y2 = int(min(img_h, y2 + pad_y))

        crop = original_img[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size > 0:
            vit_inputs.append(crop)

    # No authorized bin → DMC not concerned
    if not vit_inputs:
        return _log_and_return(image_path, 0, 0.0)

    # ── STAGE 2: ViT with TTA + Confidence Thresholding ──────
    action_probs = []
    for crop in vit_inputs:
        # BGR → RGB → PIL
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        prob = _get_action_probability(
            vit_model, pil_img, transform, device, use_tta=TTA_ENABLED
        )
        action_probs.append(prob)

    # ── AGGREGATION: Confidence-gated MAX ─────────────────────
    max_prob = max(action_probs)
    decision = 1 if max_prob > ACTION_THRESHOLD else 0

    return _log_and_return(image_path, decision, max_prob)


def _save_confidence_plot():
    """Auto-save the accumulated prediction log as a confidence bar chart, then reset."""
    global _prediction_log
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names        = [r[0] for r in _prediction_log]
    confidences  = [r[1] for r in _prediction_log]
    decisions    = [r[2] for r in _prediction_log]
    colors       = ['#e74c3c' if d == 1 else '#2ecc71' for d in decisions]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.6), 6))
    bars = ax.bar(range(len(names)), confidences, color=colors,
                  edgecolor='#2c3e50', linewidth=0.8, alpha=0.85)

    ax.axhline(y=ACTION_THRESHOLD, color='#e67e22', linestyle='--', linewidth=2,
               label=f'Threshold = {ACTION_THRESHOLD}')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=65, ha='right', fontsize=8)
    ax.set_ylabel('P(action_required)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Test Images', fontsize=12, fontweight='bold')
    ax.set_title('ViT Confidence per Image  (Red=DMC Alert, Green=No Action)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    for bar, conf in zip(bars, confidences):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{conf:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=150)
    plt.close()
    print(f"\n📊 Auto-plot saved to: {PLOT_SAVE_PATH} ({len(_prediction_log)} images)")

    _prediction_log = []  # reset for next batch


def predict_with_confidence(models, image_path):
    """
    Same as predict() but returns (decision, max_confidence).
    Used by plot_confidences() for diagnostic visualization.
    """
    yolo_model = models["yolo"]
    vit_model  = models["vit"]
    device     = models["device"]
    transform  = models["transform"]

    results = yolo_model.predict(source=image_path, conf=YOLO_CONF, verbose=False)
    result  = results[0]

    original_img = result.orig_img
    if original_img is None:
        original_img = cv2.imread(image_path)
    if original_img is None:
        return 0, 0.0

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

    if not vit_inputs:
        return 0, 0.0

    action_probs = []
    for crop in vit_inputs:
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        prob = _get_action_probability(
            vit_model, pil_img, transform, device, use_tta=TTA_ENABLED
        )
        action_probs.append(prob)

    max_prob = max(action_probs)
    decision = 1 if max_prob > ACTION_THRESHOLD else 0
    return decision, max_prob


def plot_confidences(models, image_dir, save_path="confidence_plot.png"):
    """
    Run predict on all images in image_dir and save a bar chart:
      x-axis = image filenames
      y-axis = max P(action_required) confidence
      bar color = red if flagged (>threshold), green if not
      horizontal dashed line = ACTION_THRESHOLD
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import glob

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.*")))
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    names = []
    confidences = []
    decisions = []

    for img_path in image_paths:
        basename = os.path.basename(img_path)
        decision, conf = predict_with_confidence(models, img_path)
        names.append(basename)
        confidences.append(conf)
        decisions.append(decision)
        icon = "🚨" if decision == 1 else "✅"
        print(f"{icon} {basename}: conf={conf:.4f} → {decision}")

    # ── Plot ──────────────────────────────────────────────────
    colors = ['#e74c3c' if d == 1 else '#2ecc71' for d in decisions]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.6), 6))

    bars = ax.bar(range(len(names)), confidences, color=colors, edgecolor='#2c3e50',
                  linewidth=0.8, alpha=0.85)

    # Threshold line
    ax.axhline(y=ACTION_THRESHOLD, color='#e67e22', linestyle='--', linewidth=2,
               label=f'Threshold = {ACTION_THRESHOLD}')

    # Labels
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=65, ha='right', fontsize=8)
    ax.set_ylabel('P(action_required)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Test Images', fontsize=12, fontweight='bold')
    ax.set_title('ViT Confidence per Image  (Red = DMC Alert, Green = No Action)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Annotate confidence values on bars
    for bar, conf in zip(bars, confidences):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{conf:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n📊 Plot saved to: {save_path}")

