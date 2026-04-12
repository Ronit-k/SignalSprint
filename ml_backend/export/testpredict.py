import os
import sys
import glob
import csv

# Import the evaluator's functions from your predict.py
from _2stage_predict import load_model, predict
# from _2stage_predict_modified import load_model, predict
# from predict import load_model, predict

def main():
    print("🚀 Initializing Pipeline Evaluation Test...")
    print("1. Testing load_model()...")
    try:
        models = load_model()
        print("✅ Models loaded successfully from model.pkl!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        sys.exit(1)

    # Resolve paths to test images assuming script is in ml_backend/export
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_images_dir = os.path.abspath(os.path.join(current_dir, "../test_data/images"))
    
    if not os.path.exists(test_images_dir):
        print(f"⚠️ Test directory not found at: {test_images_dir}")
        print("Attempting alternate path...")
        test_images_dir = os.path.abspath(os.path.join(current_dir, "../test_data"))
        if not os.path.exists(test_images_dir):
            sys.exit(1)

    test_data_dir = os.path.dirname(test_images_dir) if os.path.basename(test_images_dir) == "images" else test_images_dir
    labels_csv_path = os.path.join(test_data_dir, "labels.csv")
    ground_truths = {}
    if os.path.exists(labels_csv_path):
        try:
            with open(labels_csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # In case headers have spaces or differ
                    img_name = row.get('img_name', '').strip()
                    label_str = row.get('label', '').strip()
                    if img_name and label_str.strip().isdigit():
                        ground_truths[img_name] = int(label_str.strip())
            print(f"✅ Loaded {len(ground_truths)} ground truth labels from {labels_csv_path}")
        except Exception as e:
            print(f"⚠️ Could not read labels.csv: {e}")
    else:
        print(f"⚠️ No labels.csv found at {labels_csv_path}")

    image_paths = glob.glob(os.path.join(test_images_dir, "*.*"))
    # Filter to actual images
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not image_paths:
        print(f"❌ No test images found in {test_images_dir}")
        sys.exit(1)

    print(f"\n2. Testing predict() on up to 5 images from {test_images_dir}...")
    
    # Process up to 5 images
    for img_path in image_paths[0:50]:
        print("-" * 50)
        img_basename = os.path.basename(img_path)
        print(f"🖼️ Testing Image: {img_basename}")
        try:
            prediction = predict(models, img_path)
            
            icon = "🚨 (Intimate DMC)" if prediction == 1 else "✅ (Ignore)"
            print(f"🏆 Final Output: {prediction} {icon}")
            
            if ground_truths and img_basename in ground_truths:
                truth = ground_truths[img_basename]
                correct = (prediction == truth)
                res_icon = "🟢 CORRECT" if correct else "🔴 INCORRECT"
                print(f"📌 Ground Truth: {truth} -> {res_icon}")
                
        except Exception as e:
            print(f"❌ Crash during predict() for {img_basename}: {e}")

if __name__ == "__main__":
    main()
