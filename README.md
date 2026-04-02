# Signal Sprint

This repository contains the State-of-the-Art (SOTA) hierarchical computer vision pipeline and Flutter interface for the **Signal Sprint** competition.

## 🧠 Architecture Overview
The algorithm determines the probability of intimately calling the DMC using a three-stage approach:
1. **Stage 1 (DMC-Gate):** `YOLOv11n` object detection to verify the presence of authorized bins (Ground, Cage, Elevated) and explicitly reject unauthorized bins/postboxes.
2. **Stage 2 (Contextual Observer):** `HTD-ViT` (Hierarchical Token Decomposition Vision Transformer) to classify semantic textures and geometric relationships, distinguishing manmade garbage spills from natural leaves and contained overflow.
3. **Stage 3 (Logic Head):** `MSW-Net Stacking` for precise $0.0 - 1.0$ probability calibration.

## 📂 Repository Structure

```text
signal-sprint-sota/
├── .gitignore
├── README.md                 
│
├── ml_backend/               # Python/Conda domain for model training and inference
│   ├── data/
│   │   ├── raw_images/       # Initial 2,000 unaugmented 1:1 captures (ignored in Git)
│   │   ├── temp_clean_export/# Temporary unaugmented Roboflow export (ignored in Git)
│   │   ├── yolo_dataset/     # Augmented Roboflow export for Stage 1 (ignored in Git)
│   │   └── vit_dataset/      # Stage 2 classification dataset
│   │       ├── unsorted_crops/   # Temporary holding for extracted halo crops
│   │       ├── action_required/  # 1.0 Probability (Spills & overflows)
│   │       └── no_action/        # 0.0 Probability (Clean, natural litter, full but contained)
│   │
│   ├── notebooks/            # Jupyter notebooks for EDA and testing
│   ├── src/
│   │   ├── train_yolo.py     # Ultralytics training script
│   │   ├── train_vit.py      # PyTorch training script
│   │   ├── stacker.py        # MSW-Net logic head script
│   │   ├── generate_vit_crops.py # Automated extraction of Stage 2 crops
│   │   └── inference.py      # The final logic combining all stages
│   │
│   ├── models/               # Saved weights (.pt)
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