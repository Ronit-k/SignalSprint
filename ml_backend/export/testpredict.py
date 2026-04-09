import os
import sys
import glob

# Import the evaluator's functions from your predict.py
from predict import load_model, predict

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

    image_paths = glob.glob(os.path.join(test_images_dir, "*.*"))
    # Filter to actual images
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not image_paths:
        print(f"❌ No test images found in {test_images_dir}")
        sys.exit(1)

    print(f"\n2. Testing predict() on up to 5 images from {test_images_dir}...")
    
    # Process up to 5 images
    for img_path in image_paths[:50]:
        print("-" * 50)
        print(f"🖼️ Testing Image: {os.path.basename(img_path)}")
        try:
            prediction = predict(models, img_path)
            
            icon = "🚨 (Intimate DMC)" if prediction == 1 else "✅ (Ignore)"
            print(f"🏆 Final Output: {prediction} {icon}")
        except Exception as e:
            print(f"❌ Crash during predict() for {os.path.basename(img_path)}: {e}")

if __name__ == "__main__":
    main()
