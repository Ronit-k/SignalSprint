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

CLASS_NAMES = {
    0: "bin_caged",
    1: "bin_elevated",
    2: "bin_ground",
    3: "trap_object"
}

def load_model():
    # model.pkl is guaranteed to be in the same directory per the ZIP submission rules
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_pkl_path = os.path.join(current_dir, "model.pkl")

    with open(model_pkl_path, "rb") as f:
        data = pickle.load(f)

    # 1. The YOLO Loading Path: Use tempfile.gettempdir() for cross-platform safety
    yolo_temp_path = os.path.join(tempfile.gettempdir(), 'temp_yolo.pt')
    with open(yolo_temp_path, 'wb') as f:
        f.write(data['yolo'])

    # Load Stage 1: YOLO
    yolo_model = YOLO(yolo_temp_path)

    # 3. The AIR-GAP Fix: Initialize Swin via the dumped config dictionary for a totally offline run
    device = torch.device("cpu")
    vit_config = SwinConfig.from_dict(data['vit_config'])
    vit_model = SwinForImageClassification(vit_config)

    # 2. The ViT Memory Management: Load custom trained weights natively from RAM (BytesIO)
    vit_bytes_io = io.BytesIO(data['vit'])
    vit_model.load_state_dict(torch.load(vit_bytes_io, map_location=device))
    vit_model.to(device)
    vit_model.eval()

    return {
        "yolo": yolo_model,
        "vit": vit_model,
        "device": device
    }

def predict(models, image_path):
    yolo_model = models["yolo"]
    vit_model = models["vit"]
    device = models["device"]

    # Stage 1: DMC-Gate Selective Detection
    results = yolo_model.predict(source=image_path, conf=0.5, verbose=False)
    result = results[0]

    original_img = result.orig_img
    if original_img is None:
        original_img = cv2.imread(image_path)
    if original_img is None:
        # Fallback if image cannot be read
        return 0

    img_h, img_w = original_img.shape[:2]
    vit_inputs = []

    for box in result.boxes:
        # Extract classification info
        class_id = int(box.cls[0].item())
        class_name = CLASS_NAMES.get(class_id, "unknown")

        # THE SINK LOGIC: Ignore trap objects immediately
        if class_name == "trap_object":
            continue

        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # Halo Math (Add 150% padding on all 4 sides)
        pad_x = bbox_w * 1.5
        pad_y = bbox_h * 1.5

        crop_x1 = int(max(0, x1 - pad_x))
        crop_y1 = int(max(0, y1 - pad_y))
        crop_x2 = int(min(img_w, x2 + pad_x))
        crop_y2 = int(min(img_h, y2 + pad_y))

        # Extract the padded image slice
        crop_img = original_img[crop_y1:crop_y2, crop_x1:crop_x2]

        if crop_img.size > 0:
            vit_inputs.append(crop_img)

    # Output 0.0 immediately if no authorized bin is detected
    if not vit_inputs:
        return 0

    # ViT transforms
    vit_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Stage 2: The Contextual Observer (Process Crops)
    bin_decisions = []
    for crop in vit_inputs:
        # Convert BGR -> RGB -> PIL Image to match training data
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        input_tensor = vit_transforms(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = vit_model(input_tensor).logits
            predicted_class_id = torch.argmax(outputs, dim=1).item()

        # Safely checks for integer key, falls back to string key, then defaults to "no_action"
        predicted_label = vit_model.config.id2label.get(
            int(predicted_class_id), 
            vit_model.config.id2label.get(str(predicted_class_id), "no_action")
        )
        
        if "action_required" in predicted_label or predicted_label == "action_required":
            bin_decisions.append(1)
        else:
            bin_decisions.append(0)

    # Stage 3: The Logic Head (Probabilistic Calibration via MAX Rule)
    # Trigger the DMC if any single bin is critically overflowing
    if any(decision == 1 for decision in bin_decisions):
        return 1
        
    return 0
