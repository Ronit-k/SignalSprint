import cv2
import os
import glob
import tkinter as tk
from PIL import Image, ImageTk

def ask_user_for_crop(crop_img, window_title="Multi-Bin Detected!"):
    """
    Displays the crop using Tkinter and waits for the user to press '1' or '0'.
    Returns True for Action (1), False for No Action (0).
    """
    choice = [None]  # Use a list to store the mutable state inside the callback

    root = tk.Tk()
    root.title(window_title)
    
    # Convert OpenCV BGR format to RGB for PIL/Tkinter
    img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Optional: Resize if the crop is massive so it fits nicely on screen
    max_size = 500
    if pil_img.width > max_size or pil_img.height > max_size:
        pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
    tk_img = ImageTk.PhotoImage(pil_img)

    # UI Elements
    img_label = tk.Label(root, image=tk_img)
    img_label.pack(padx=20, pady=10)

    instruction = tk.Label(
        root, 
        text="Press '1' for Action Required\nPress '0' for No Action", 
        font=("Arial", 14, "bold")
    )
    instruction.pack(pady=10)

    # Key press handler
    def on_key(event):
        if event.char == '1':
            choice[0] = True
            root.destroy()
        elif event.char == '0':
            choice[0] = False
            root.destroy()

    root.bind('<Key>', on_key)
    
    # Force window to the front so you don't have to click it every time
    root.focus_force() 
    root.mainloop()

    return choice[0]

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
        "action_saved_auto": 0,
        "no_action_saved_auto": 0,
        "action_saved_manual": 0,
        "no_action_saved_manual": 0,
        "traps_ignored": 0,
        "images_skipped_no_tag": 0
    }

    print("🚀 Starting Strict 50% Halo Cropping & Sorting...\n")

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

            # 1. Determine Image-Level State
            is_action = False
            is_no_action = False
            
            for line in lines:
                if not line.strip(): continue
                class_id = line.strip().split()[0]
                if class_id == "4":
                    is_action = True
                elif class_id == "5":
                    is_no_action = True

            # Skip if missing image-level tags
            if not is_action and not is_no_action:
                stats["images_skipped_no_tag"] += 1
                continue

            # 2. Collect all valid crops for this image FIRST
            collected_crops = []

            for i, line in enumerate(lines):
                parts = line.strip().split()
                if not parts: continue
                
                class_id = parts[0]
                
                if class_id == "3":
                    stats["traps_ignored"] += 1
                    continue
                    
                if class_id in ["0", "1", "2"]:
                    _, x_center, y_center, bbox_w, bbox_h = map(float, parts)
                    
                    x_c_pix, y_c_pix = x_center * img_w, y_center * img_h
                    w_pix, h_pix = bbox_w * img_w, bbox_h * img_h

                    halo_pad_x, halo_pad_y = w_pix * 0.25, h_pix * 0.25

                    x_min = int(max(0, x_c_pix - (w_pix / 2) - halo_pad_x))
                    x_max = int(min(img_w, x_c_pix + (w_pix / 2) + halo_pad_x))
                    y_min = int(max(0, y_c_pix - (h_pix / 2) - halo_pad_y))
                    y_max = int(min(img_h, y_c_pix + (h_pix / 2) + halo_pad_y))

                    crop_img = img[y_min:y_max, x_min:x_max]
                    
                    if crop_img.size > 0:
                        crop_filename = f"{base_name.replace('.jpg', '')}_crop_{i}.jpg"
                        collected_crops.append({
                            "img": crop_img,
                            "filename": crop_filename,
                            "class_id": class_id
                        })
                    else:
                        print(f"⚠️ Warning: Invalid crop calculated for {base_name}, line {i}")

            # 3. Routing Logic (Auto vs. Manual)
            if len(collected_crops) == 1:
                # AUTO ROUTE: Trust the CSV / Image Tag
                crop = collected_crops[0]
                target_folder = action_dir if is_action else no_action_dir
                save_path = os.path.join(target_folder, crop["filename"])
                cv2.imwrite(save_path, crop["img"])
                
                if is_action:
                    stats["action_saved_auto"] += 1
                else:
                    stats["no_action_saved_auto"] += 1
                print(f"✅ AUTO-Sorted {crop['filename']}")

            elif len(collected_crops) > 1:
                # MANUAL ROUTE: Multi-Bin Detected!
                print(f"\n🛑 Multi-Bin Detected in {base_name}! ({len(collected_crops)} bins). Waiting for input...")
                
                for crop in collected_crops:
                    # Fire up Tkinter
                    user_wants_action = ask_user_for_crop(crop["img"], window_title=f"Sorting: {crop['filename']}")
                    
                    # Handle if user closes window without pressing a key
                    if user_wants_action is None:
                        print(f"⚠️ Window closed without input. Defaulting to overall image state.")
                        user_wants_action = is_action

                    # Save based on user input
                    target_folder = action_dir if user_wants_action else no_action_dir
                    save_path = os.path.join(target_folder, crop["filename"])
                    cv2.imwrite(save_path, crop["img"])

                    if user_wants_action:
                        stats["action_saved_manual"] += 1
                        print(f"   👤 Manual -> [Action Required] -> {crop['filename']}")
                    else:
                        stats["no_action_saved_manual"] += 1
                        print(f"   👤 Manual -> [No Action]       -> {crop['filename']}")

    # Print the Final Report
    print("\n" + "="*50)
    print("📊 ViT DATASET GENERATION REPORT")
    print("="*50)
    print(f"🤖 Auto-Sorted (Action)    : {stats['action_saved_auto']}")
    print(f"🤖 Auto-Sorted (No Action) : {stats['no_action_saved_auto']}")
    print("-" * 50)
    print(f"👤 Manually Sorted (Action): {stats['action_saved_manual']}")
    print(f"👤 Manually Sorted (No Act): {stats['no_action_saved_manual']}")
    print("-" * 50)
    print(f"🛡️  Traps Ignored          : {stats['traps_ignored']}")
    print(f"⚠️  Images Skipped         : {stats['images_skipped_no_tag']}")
    print("="*50)

if __name__ == "__main__":
    generate_and_sort_crops()