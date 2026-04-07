import os
from PIL import Image
import pillow_heif

# Enable HEIC support
pillow_heif.register_heif_opener()

folder = "/media/ronit/SharedVolume/SignalSprint/ml_backend/data/raw_images/new"

for file in os.listdir(folder):
    if file.lower().endswith(".heic"):
        heic_path = os.path.join(folder, file)
        jpg_path = os.path.join(folder, os.path.splitext(file)[0] + ".jpg")

        try:
            # Open HEIC
            img = Image.open(heic_path)

            # Resize to 512x512
            img = img.resize((512, 512), Image.LANCZOS)

            # Convert to RGB (important for JPG)
            img = img.convert("RGB")

            # Save as JPG
            img.save(jpg_path, "JPEG", quality=95)

            # Delete original HEIC
            os.remove(heic_path)

            print(f"Converted: {file} → {os.path.basename(jpg_path)}")

        except Exception as e:
            print(f"Error processing {file}: {e}")