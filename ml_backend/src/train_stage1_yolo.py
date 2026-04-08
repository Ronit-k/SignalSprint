from ultralytics import YOLO

def train_bin_detector():
    # 1. Load a pre-trained YOLO11 model
    # 'yolo11n.pt' is Nano (fastest). Use 'yolo11s.pt' (Small) for a bit more accuracy.
    model = YOLO("yolo11m.pt") 

    print("🚀 Starting YOLO11 Training...")

    # 2. Train the model
    results = model.train(
        data="ml_backend/data/yolo_dataset/yolo_dataset.yaml",    # Path to your YAML file
        epochs=100,               # 100 is a good baseline to see how it converges
        imgsz=512,                # Standard YOLO image size
        batch=16,                 # Adjust based on your GPU RAM (8, 16, 32)
        device="0",               # Use "0" for GPU, or "cpu" if you don't have a GPU
        project="IITK_Bin_Vision",# Main folder for saved models
        name="run_v1_base",       # Name of this specific training run
        
        # --- Advanced Parameters for your specific dataset ---
        patience=20,              # Early stopping if no improvement for 20 epochs
        save=True,                # Save the best weights
        plots=True,               # Generate confusion matrices and loss curves
        workers=8                 # CPU workers for data loading (adjust based on your CPU)
    )

    print(f"\n✅ Training Complete! Best model saved to: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    train_bin_detector()