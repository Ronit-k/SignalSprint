import os
import glob

def sanitize_dataset():
    # Update this path if your dataset is located elsewhere
    dataset_path = os.path.expanduser("../data/yolo_dataset")
    
    # Note: Roboflow typically names the validation folder 'valid'
    splits = ["train", "valid", "test"]

    # Counters for the summary report
    counts = {
        "0": {"name": "bin_caged", "count": 0},
        "1": {"name": "bin_elevated", "count": 0},
        "2": {"name": "bin_ground", "count": 0},
        "3": {"name": "trap_object", "count": 0},
        "4": {"name": "z_action", "count": 0},
        "5": {"name": "z_no_action", "count": 0}
    }

    files_modified = 0

    for split in splits:
        label_dir = os.path.join(dataset_path, split, "labels")
        if not os.path.exists(label_dir):
            print(f"Warning: Directory not found -> {label_dir}")
            continue
            
        print(f"Scanning {split}/labels...")
        
        for label_file in glob.glob(os.path.join(label_dir, "*.txt")):
            with open(label_file, "r") as f:
                lines = f.readlines()
                
            new_lines = []
            file_needs_rewrite = False
            
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                    
                class_id = parts[0]
                
                # Tally the counts
                if class_id in counts:
                    counts[class_id]["count"] += 1
                
                # Keep classes 0, 1, 2, 3. Discard 4 and 5.
                if class_id in ["0", "1", "2", "3"]:
                    new_lines.append(line)
                elif class_id in ["4", "5"]:
                    file_needs_rewrite = True
                    
            # If we found action/no_action tags, rewrite the file without them
            if file_needs_rewrite:
                with open(label_file, "w") as f:
                    f.writelines(new_lines)
                files_modified += 1

    # Print the final report
    print("\n" + "="*50)
    print("📊 DATASET SANITIZATION REPORT")
    print("="*50)
    print(f"Files modified: {files_modified}\n")
    print("Class Counts (Found before deletion):")
    for cid, data in counts.items():
        status = "-> [KEPT]" if cid in ["0", "1", "2", "3"] else "-> [DELETED]"
        # Formatting to keep the columns aligned
        print(f"  Class {cid} ({data['name'].ljust(12)}): {str(data['count']).rjust(5)} {status}")
    print("="*50)
    print("\n✅ Cleanup complete!")

if __name__ == "__main__":
    sanitize_dataset()