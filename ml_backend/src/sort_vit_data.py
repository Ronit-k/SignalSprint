import cv2
import os
import glob

def generate_and_sort_crops():
    # --- PATHS ---
    source_dataset = os.path.expanduser("../data/temp_clean_export")
    target_dataset = os.path.expanduser("../data/vit_dataset")
    
    action_dir = os.path.join(target_dataset, "action_required")
    no_action_dir = os.path.join(target_dataset, "no_action")
    os.makedirs(action_dir, exist_ok=True)
    os.makedirs(no_action_dir, exist_ok=True)

    splits = ["train", "valid", "test"]
    
    # Trackers for your final report
    stats = {
        "action_saved": 0,
        "no_action_saved": 0,
        "traps_ignored": 0,
        "images_skipped_no_tag": 0
    }

    print("🚀 Starting Strict 50% Halo Cropping...\n")

    for split in splits:
        label_dir = os.path.join(source_dataset, split, "labels")
        image_dir = os.path.join(source_dataset, split, "images")

        if not os.path.exists(label_dir):
            continue

        for label_file in glob.glob(os.path.join(label_dir, "*.txt")):
            base_name = os.path.basename(label_file).replace(".txt", ".jpg")
            image_path = os.path.join(image_dir, base_name)

            if not os.path.exists(image_path):
                continue

            img = cv2.imread(image_path)
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            with open(label_file, "r") as f:
                lines = f.readlines()

            # 1. Determine State (Must have 4 or 5)
            is_action = False
            is_no_action = False
            
            for line in lines:
                if not line.strip(): continue
                class_id = line.strip().split()[0]
                if class_id == "4":
                    is_action = True
                elif class_id == "5":
                    is_no_action = True

            # If the image has NO state tag, skip it entirely
            if not is_action and not is_no_action:
                stats["images_skipped_no_tag"] += 1
                continue

            # Route priority: If an image has both for some reason, prioritize 'action'
            target_folder = action_dir if is_action else no_action_dir
            folder_name = "action_required" if is_action else "no_action"

            # 2. Extract Bins (Classes 0, 1, 2 only)
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if not parts: continue
                
                class_id = parts[0]
                
                # Ignore Traps (3) and the State Tags (4, 5)
                if class_id == "3":
                    stats["traps_ignored"] += 1
                    continue
                    
                # ONLY process Caged (0), Elevated (1), Ground (2)
                if class_id in ["0", "1", "2"]:
                    _, x_center, y_center, bbox_w, bbox_h = map(float, parts)
                    
                    # Convert YOLO to Pixels
                    x_c_pix = x_center * img_w
                    y_c_pix = y_center * img_h
                    w_pix = bbox_w * img_w
                    h_pix = bbox_h * img_h

                    # 50% Halo Math (Expand 25% on all 4 sides)
                    halo_pad_x = w_pix * 0.25
                    halo_pad_y = h_pix * 0.25

                    # Calculate Crop Box and Clamp to image edges
                    x_min = int(max(0, x_c_pix - (w_pix / 2) - halo_pad_x))
                    x_max = int(min(img_w, x_c_pix + (w_pix / 2) + halo_pad_x))
                    y_min = int(max(0, y_c_pix - (h_pix / 2) - halo_pad_y))
                    y_max = int(min(img_h, y_c_pix + (h_pix / 2) + halo_pad_y))

                    crop_img = img[y_min:y_max, x_min:x_max]
                    
                    if crop_img.size == 0:
                        print(f"⚠️ Warning: Invalid crop calculated for {base_name}, line {i}")
                        continue

                    # Save the crop
                    crop_filename = f"{base_name.replace('.jpg', '')}_crop_{i}.jpg"
                    save_path = os.path.join(target_folder, crop_filename)
                    cv2.imwrite(save_path, crop_img)
                    
                    # Log the success
                    if is_action:
                        stats["action_saved"] += 1
                    else:
                        stats["no_action_saved"] += 1
                        
                    print(f"✅ Cropped Class {class_id} from {base_name} -> Saved as {crop_filename} in [{folder_name}]")

    # Print the Final Report
    print("\n" + "="*50)
    print("📊 ViT DATASET GENERATION REPORT")
    print("="*50)
    print(f"📁 action_required/ : {stats['action_saved']} crops successfully stored")
    print(f"📁 no_action/       : {stats['no_action_saved']} crops successfully stored")
    print("-" * 50)
    print(f"🛡️  Traps Ignored    : {stats['traps_ignored']} trap bounding boxes skipped")
    print(f"⚠️  Images Skipped   : {stats['images_skipped_no_tag']} images skipped (Missing class 4 or 5 tag)")
    print("="*50)

if __name__ == "__main__":
    generate_and_sort_crops()