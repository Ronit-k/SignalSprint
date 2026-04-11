"""
3-Stage Pipeline Predict Script
================================
Stage 1 — DMC-Gate Detector     : Custom YOLOv8 (best_yolo_v2.pt)
             - Detects bin_caged / bin_elevated / bin_ground / trap_object
             - Filters out trap_object (sink logic)
             - Extracts 150% halo crops around each valid bin

Stage 2 — COCO Context Annotator: YOLOv8m pretrained on COCO (yolov8m.pt)
             - Runs on each halo crop
             - Plots general-object bounding boxes (spills, trash, etc.)
               directly onto the crop pixels via result.plot()

Stage 3 — Contextual Observer   : Fine-tuned Swin Transformer (best_vit_v4.pth)
             - Receives the visually annotated crop from Stage 2
             - Classifies: action_required (1) or no_action (0)

Aggregation — OR rule: if ANY bin returns action_required → final = 1
"""

import os
import io
import cv2
import pickle
import tempfile
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import SwinConfig, SwinForImageClassification
from ultralytics import YOLO

# Class map for the custom Stage-1 YOLO model
CLASS_NAMES = {
    0: "bin_caged",
    1: "bin_elevated",
    2: "bin_ground",
    3: "trap_object"
}


def load_model():
    """
    Deserialize model_3stage.pkl (expected alongside this script in the ZIP)
    and return a dict with the three ready-to-run model objects.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_pkl_path = os.path.join(current_dir, "model.pkl")

    with open(model_pkl_path, "rb") as f:
        data = pickle.load(f)

    device = torch.device("cpu")

    # --- Stage 1: Custom DMC-Gate YOLO ---
    yolo_s1_tmp = os.path.join(tempfile.gettempdir(), "temp_yolo_stage1.pt")
    with open(yolo_s1_tmp, "wb") as f:
        f.write(data["yolo_stage1"])
    yolo_stage1 = YOLO(yolo_s1_tmp)

    # --- Stage 2: COCO Context YOLO ---
    yolo_s2_tmp = os.path.join(tempfile.gettempdir(), "temp_yolo_stage2.pt")
    with open(yolo_s2_tmp, "wb") as f:
        f.write(data["yolo_stage2"])
    yolo_stage2 = YOLO(yolo_s2_tmp)

    # --- Stage 3: Swin Transformer (air-gap safe: config from pkl, no HF download) ---
    vit_config = SwinConfig.from_dict(data["vit_config"])
    vit_model = SwinForImageClassification(vit_config)

    vit_bytes_io = io.BytesIO(data["vit"])
    vit_model.load_state_dict(torch.load(vit_bytes_io, map_location=device))
    vit_model.to(device)
    vit_model.eval()

    return {
        "yolo_stage1": yolo_stage1,
        "yolo_stage2": yolo_stage2,
        "vit": vit_model,
        "device": device,
    }


def predict(models, image_path):
    """
    Run the full 3-stage pipeline on a single image.

    Returns
    -------
    int : 1 if action is required (overflow / spill detected), 0 otherwise.
    """
    yolo_stage1 = models["yolo_stage1"]
    yolo_stage2 = models["yolo_stage2"]
    vit_model   = models["vit"]
    device      = models["device"]

    # ------------------------------------------------------------------ #
    # STAGE 1 — DMC-Gate Selective Detection
    # ------------------------------------------------------------------ #
    results = yolo_stage1.predict(source=image_path, conf=0.1, verbose=False)
    result  = results[0]

    original_img = result.orig_img
    if original_img is None:
        original_img = cv2.imread(image_path)
    if original_img is None:
        return 0  # Cannot read image, safe default

    img_h, img_w = original_img.shape[:2]

    # Collect halo crops for each valid (non-trap) detection
    halo_crops = []
    for box in result.boxes:
        class_id   = int(box.cls[0].item())
        class_name = CLASS_NAMES.get(class_id, "unknown")

        # SINK LOGIC: skip trap objects immediately
        if class_name == "trap_object":
            continue

        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # 150% Halo — 1.5× the bbox dim as padding on every side
        pad_x = bbox_w * 1.5
        pad_y = bbox_h * 1.5

        crop_x1 = int(max(0,     x1 - pad_x))
        crop_y1 = int(max(0,     y1 - pad_y))
        crop_x2 = int(min(img_w, x2 + pad_x))
        crop_y2 = int(min(img_h, y2 + pad_y))

        crop = original_img[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size > 0:
            halo_crops.append(crop)

    # No authorized bin found — DMC not concerned
    if not halo_crops:
        return 0

    # ------------------------------------------------------------------ #
    # STAGE 2 — COCO Context Annotation
    # ------------------------------------------------------------------ #
    # Run the COCO model on each halo crop and bake the detected bounding
    # boxes (spills, loose trash, etc.) directly into the pixel data using
    # result.plot() so the ViT can reason visually about the context.
    annotated_crops = []
    for crop in halo_crops:
        coco_results    = yolo_stage2.predict(crop, conf=0.25, verbose=False)
        annotated_crop  = coco_results[0].plot()  # BGR NumPy array with boxes painted on
        annotated_crops.append(annotated_crop)

    # ------------------------------------------------------------------ #
    # STAGE 3 — The Contextual Observer (Swin Transformer)
    # ------------------------------------------------------------------ #
    vit_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    bin_decisions = []
    for crop in annotated_crops:
        # BGR → RGB → PIL (must match training-time preprocessing)
        img_rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img  = Image.fromarray(img_rgb)

        input_tensor = vit_transforms(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits            = vit_model(input_tensor).logits
            predicted_class_id = torch.argmax(logits, dim=1).item()

        # Safely resolve the label (handles int or string keys in id2label)
        predicted_label = vit_model.config.id2label.get(
            int(predicted_class_id),
            vit_model.config.id2label.get(str(predicted_class_id), "no_action")
        )

        bin_decisions.append(1 if "action_required" in predicted_label else 0)

    # ------------------------------------------------------------------ #
    # AGGREGATION — OR Rule
    # ------------------------------------------------------------------ #
    return 1 if any(d == 1 for d in bin_decisions) else 0
