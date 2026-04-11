import os
import cv2
import glob
from ultralytics import YOLO

def process_and_save(model, src_folder, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    images = glob.glob(os.path.join(src_folder, '*.*'))
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(f"Processing {len(images)} images from {src_folder}...")
    
    if not images:
        print(f"⚠️ No images found in {src_folder}")
        return

    for img_path in images:
        filename = os.path.basename(img_path)
        dest_path = os.path.join(dest_folder, filename)
        
        # Run inference
        # conf=0.25 is a common default, adjusts as needed
        results = model.predict(source=img_path, conf=0.25, verbose=False)
        
        # results[0].plot() returns a numpy array representing the image with bounding boxes
        annotated_img = results[0].plot()
        
        # Save annotated image
        cv2.imwrite(dest_path, annotated_img)
        
    print(f"✅ Completed saving {len(images)} images to {dest_folder}\n")

if __name__ == "__main__":
    # Base paths setup assuming script runs from ml_backend/src
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # Go up to ml_backend
    
    model_path = os.path.join(project_root, "models", "yolov8m.pt")
    
    old_action_dir = os.path.join(project_root, "data", "vit_dataset", "old_action_required")
    old_no_action_dir = os.path.join(project_root, "data", "vit_dataset", "old_no_action")
    
    new_action_dir = os.path.join(project_root, "data", "vit_dataset", "action_required")
    new_no_action_dir = os.path.join(project_root, "data", "vit_dataset", "no_action")
    
    print(f"Loading YOLO model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"❌ Error: YOLO model not found at {model_path}")
        exit(1)
        
    model = YOLO(model_path)
    
    # Process "action_required"
    print("--- Starting Processing: Action Required (Intimate DMC) ---")
    process_and_save(model, old_action_dir, new_action_dir)
    
    # Process "no_action"
    print("--- Starting Processing: No Action (Ignore) ---")
    process_and_save(model, old_no_action_dir, new_no_action_dir)
    
    print("✨ All preprocessing finished successfully! Your visual YOLO-annotated images are ready for ViT training.")
