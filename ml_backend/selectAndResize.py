import os
import tkinter as tk
from PIL import Image, ImageTk

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif'}

def process_images(base_dir="."):
    print("Starting image processing...")
    print("Commands:")
    print("  [Enter] : Convert to .jpg (if needed) and center-crop to 1:1")
    print("  00      : Delete the image")
    print("  s       : Skip to next image")
    print("  q       : Quit the script")
    print("-" * 50)

    # Initialize a hidden background Tkinter instance to manage our windows
    root = tk.Tk()
    root.withdraw() 

    for root_dir, dirs, files in os.walk(base_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            
            if ext in VALID_EXTENSIONS:
                file_path = os.path.join(root_dir, file)

                try:
                    # Open the original image for processing
                    img = Image.open(file_path)
                    
                    # Create a temporary image just for viewing so it fits on screen
                    view_img = img.copy()
                    view_img.thumbnail((800, 800)) # Scales down for preview only
                    
                    # Create a Python-controlled window
                    viewer_window = tk.Toplevel(root)
                    viewer_window.title(f"Preview: {file}")
                    
                    # Convert to Tkinter-compatible photo
                    tk_photo = ImageTk.PhotoImage(view_img)
                    label = tk.Label(viewer_window, image=tk_photo)
                    label.pack()
                    
                    # Bring window to the front, then render it without blocking the terminal
                    viewer_window.lift()
                    viewer_window.attributes('-topmost', True)
                    viewer_window.after_idle(viewer_window.attributes, '-topmost', False)
                    viewer_window.update() 

                    print(f"\nViewing: {file_path}")
                    user_input = input("Action ([Enter]=Crop/Convert, '00'=Delete, 's'=Skip, 'q'=Quit): ")

                    # IMMEDIATELY close the photo window
                    viewer_window.destroy()
                    root.update()

                    # ACTION 1: Delete
                    if user_input == "00":
                        img.close()
                        os.remove(file_path)
                        print(f"[-] Deleted: {file_path}")

                    # ACTION 2: Convert to JPG and crop 1:1
                    elif user_input == "":
                        width, height = img.size
                        min_dim = min(width, height)

                        left = (width - min_dim) // 2
                        top = (height - min_dim) // 2
                        right = left + min_dim
                        bottom = top + min_dim

                        img_cropped = img.crop((left, top, right, bottom))
                        img_rgb = img_cropped.convert("RGB")

                        new_file_path = os.path.splitext(file_path)[0] + ".jpg"
                        img_rgb.save(new_file_path, "JPEG", quality=95)
                        img.close()

                        if file_path != new_file_path:
                            os.remove(file_path)
                            print(f"[+] Converted and Cropped -> Saved as {os.path.basename(new_file_path)}")
                        else:
                            print(f"[+] Cropped to 1:1 -> Overwrote {os.path.basename(new_file_path)}")

                    # ACTION 3: Quit
                    elif user_input.lower() == 'q':
                        img.close()
                        root.destroy()
                        print("Exiting script...")
                        return

                    # ACTION 4: Skip
                    else:
                        img.close()
                        print("[~] Skipped.")

                except Exception as e:
                    print(f"[!] Error processing {file_path}: {e}")

    # Clean up Tkinter when done
    root.destroy()

if __name__ == "__main__":
    process_images(os.getcwd() + "/raw_images/new/screenshotsordownloads")
