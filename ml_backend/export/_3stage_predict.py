"""
3-Stage SOTA Predict Pipeline
==============================
Stage 1 — DMC-Gate Detector     : Custom YOLOv8 (best_yolo_v2.pt)
Stage 2 — COCO Context Annotator: YOLOv8m (yolov8m.pt) — plots boxes onto crop
Stage 3 — Contextual Observer   : Fine-tuned Swin Transformer (best_vit_v4.pth)
           + Softmax confidence thresholding + TTA

Aggregation — Confidence-gated MAX with TTA-averaged probabilities
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
# HYPERPARAMETERS
# ──────────────────────────────────────────────
YOLO_CONF          = 0.25
COCO_CONF          = 0.25
ACTION_THRESHOLD   = 0.55
TTA_ENABLED        = True

CLASS_NAMES = {
    0: "bin_caged",
    1: "bin_elevated",
    2: "bin_ground",
    3: "trap_object"
}


def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_pkl_path = os.path.join(current_dir, "model.pkl")

    with open(model_pkl_path, "rb") as f:
        data = pickle.load(f)

    device = torch.device("cpu")

    # Stage 1: Custom YOLO
    yolo_s1_tmp = os.path.join(tempfile.gettempdir(), "temp_yolo_stage1.pt")
    with open(yolo_s1_tmp, "wb") as f:
        f.write(data["yolo_stage1"])
    yolo_stage1 = YOLO(yolo_s1_tmp)

    # Stage 2: COCO YOLO
    yolo_s2_tmp = os.path.join(tempfile.gettempdir(), "temp_yolo_stage2.pt")
    with open(yolo_s2_tmp, "wb") as f:
        f.write(data["yolo_stage2"])
    yolo_stage2 = YOLO(yolo_s2_tmp)

    # Stage 3: Swin Transformer
    vit_config = SwinConfig.from_dict(data["vit_config"])
    vit_model = SwinForImageClassification(vit_config)
    vit_bytes_io = io.BytesIO(data["vit"])
    vit_model.load_state_dict(torch.load(vit_bytes_io, map_location=device))
    vit_model.to(device)
    vit_model.eval()

    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return {
        "yolo_stage1": yolo_stage1,
        "yolo_stage2": yolo_stage2,
        "vit": vit_model,
        "device": device,
        "transform": base_transform,
    }


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


def predict(models, image_path):
    yolo_stage1 = models["yolo_stage1"]
    yolo_stage2 = models["yolo_stage2"]
    vit_model   = models["vit"]
    device      = models["device"]
    transform   = models["transform"]

    # ── STAGE 1 ───────────────────────────────
    results = yolo_stage1.predict(source=image_path, conf=YOLO_CONF, verbose=False)
    result  = results[0]

    original_img = result.orig_img
    if original_img is None:
        original_img = cv2.imread(image_path)
    if original_img is None:
        return 0

    img_h, img_w = original_img.shape[:2]

    halo_crops = []
    for box in result.boxes:
        class_id   = int(box.cls[0].item())
        class_name = CLASS_NAMES.get(class_id, "unknown")
        if class_name == "trap_object":
            continue

        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        bbox_w, bbox_h  = x2 - x1, y2 - y1
        pad_x, pad_y    = bbox_w * 1.5, bbox_h * 1.5

        crop_x1 = int(max(0,     x1 - pad_x))
        crop_y1 = int(max(0,     y1 - pad_y))
        crop_x2 = int(min(img_w, x2 + pad_x))
        crop_y2 = int(min(img_h, y2 + pad_y))

        crop = original_img[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size > 0:
            halo_crops.append(crop)

    if not halo_crops:
        return 0

    # ── STAGE 2: COCO Context ─────────────────
    annotated_crops = []
    for crop in halo_crops:
        coco_results = yolo_stage2.predict(crop, conf=COCO_CONF, verbose=False)
        annotated_crops.append(coco_results[0].plot())

    # ── STAGE 3: ViT + TTA + Confidence ───────
    action_probs = []
    for crop in annotated_crops:
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        prob = _get_action_probability(
            vit_model, pil_img, transform, device, use_tta=TTA_ENABLED
        )
        action_probs.append(prob)

    # ── AGGREGATION ───────────────────────────
    return 1 if max(action_probs) > ACTION_THRESHOLD else 0
