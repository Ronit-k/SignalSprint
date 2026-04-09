# Signal Sprint: SOTA "DMC-Intelligence" Pipeline

This repository contains the State-of-the-Art (SOTA) hierarchical computer vision pipeline and Flutter interface for the **Signal Sprint** competition. 

This pipeline is engineered to solve the complex visual overlap between "clean bins" and "active spills" by separating localization (Stage 1) from spatial reasoning (Stage 2) to prevent multi-class model confusion.

## 🧠 Architecture Overview
The algorithm determines the probability of intimately calling the DMC using a three-stage approach:

1. **Stage 1 (DMC-Gate): Selective Detection** * **Model:** `YOLOv11n`
   * **Role:** Detects authorized bins (`bin_ground`, `bin_cage`, `bin_elevated`) and explicitly rejects unauthorized objects (`trap_object`). 
   * **SOTA Trick:** Utilizes entirely unlabeled `null` images during training to actively suppress false positives on empty scattered garbage. Outputs $0.0$ immediately if no authorized bin is detected.

2. **Stage 2 (Contextual Observer): Spatial Reasoning**
   * **Model:** `HTD-ViT` (Hierarchical Token Decomposition Vision Transformer)
   * **Role:** Analyzes a localized 50% "halo" crop around the detected bin. It classifies the semantic textures and geometric relationships, distinguishing manmade garbage spills (`action_required`) from natural leaves and contained overflow (`no_action`).

3. **Stage 3 (Logic Head): Probabilistic Calibration**
   * **Model:** `MSW-Net Stacking` & The MAX Rule
   * **Role:** Calibrates the raw ViT outputs into precise $0.0 - 1.0$ probabilities based on DMC constraints. 
   * **Inference Logic:** If multiple bins are detected in a single frame, the pipeline evaluates them independently and aggregates the final score using the **MAX Rule** (triggering the DMC if *any* single bin is critically overflowing).

## 🛠️ The Dual-Stage Dataset Pipeline
To train this architecture without triggering "Multi-Bin Poisoning," this repository utilizes a custom data-engineering workflow:
* **Stage 1 Prep:** Bounding boxes are drawn purely for object type, alongside image-level state tags.
* **Stage 2 Prep (`smart_sort_crops.py`):** A custom "Human-in-the-Loop" OpenCV script. It parses dataset CSVs to automatically route single-bin crops (90% of data) into ViT training folders, while pausing to explicitly ask the user to manually verify multi-bin crops (10% of data), guaranteeing absolute dataset purity.

## 📂 Repository Structure

```text
signal-sprint-sota/
├── .gitignore
├── README.md                 
│
├── ml_backend/               # Python/Conda domain for model training and inference
│   ├── data/
|   |   |__ heicToJpg.py      # Converts HEIC images to JPG
|   |   |__ selectAndResize.py# Selects and resizes images
│   │   ├── raw_images/       # Initial 2,000 unaugmented 1:1 captures (ignored in Git)
│   │   ├── temp_clean_export/# Temporary unaugmented Roboflow export (ignored in Git)
│   │   ├── yolo_dataset/     # Augmented Roboflow export for Stage 1 (ignored in Git)
│   │   └── vit_dataset/      # Stage 2 classification dataset
│   │       ├── unsorted_crops/   # Temporary holding for extracted halo crops
│   │       ├── action_required/  # 1.0 Probability (Spills & overflows)
│   │       └── no_action/        # 0.0 Probability (Clean, natural litter, full but contained)
│   │
│   ├── notebooks/            # Jupyter notebooks for EDA and pipeline testing
│   ├── src/
│   │   ├── train_stage1_yolo.py  # Ultralytics fine-tuning script
│   │   ├── train_stage2_vit.py   # PyTorch training script
│   │   ├── generate_and_sort_vit_crops.py # Automated extraction of Stage 2 50%-halo crops
│   │   └── inference.py          # The final logic combining all stages (MAX Rule)
│   │
│   ├── models/               # Saved weights (.pt/pth)
│   ├── export/
│   │   └── signal_sprint_pipeline.pkl  # Final 80% submission file
│   │
│   └── requirements.txt      # Conda/pip dependencies
│
└── flutter_app/              # Bonus 20% domain
    ├── lib/
    │   ├── main.dart
    │   ├── screens/          # Capture/Upload UI
    │   └── services/         # API calls to the local Python backend
    ├── pubspec.yaml
    └── android/              # Native configs