# Litter Detection — University of Exeter Research Project

An end-to-end litter detection and geospatial mapping pipeline built on YOLOv8, developed as part of a research project at the University of Exeter.

## Overview

This project trains and deploys a custom YOLO object detection model to identify litter in images captured in the field. Detections are then plotted on an interactive map using GPS coordinates, enabling spatial analysis of litter distribution.

## Repository Contents

| File | Description |
|------|-------------|
| `Research_Project_Complete_Pipeline.ipynb` | Full pipeline: dataset preparation, YOLO format conversion, model inference, and heatmap generation |
| `Yolo26x_litter.pt` | Trained YOLOv8x model weights fine-tuned for litter detection |

## Pipeline Steps

1. **Dataset Import** — Loads a COCO-annotated litter dataset from Google Drive into the Colab environment.
2. **Dataset Conversion** — Converts COCO JSON annotations to YOLO format (normalised bounding boxes), splitting into train / val / test subsets (540 / 116 / 116 images).
3. **Model Inference** — Runs `Yolo26x_litter.pt` on field images with a confidence threshold of 0.5.
4. **Heatmap Visualisation** — Aggregates detection coordinates and renders an interactive Folium heatmap to identify litter hotspots.

## Getting Started

### Prerequisites

```bash
pip install ultralytics folium
```

### Running the Notebook

1. Open `Research_Project_Complete_Pipeline.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Upload or mount `Yolo26x_litter.pt` to `/content/`.
3. Provide your dataset (images + COCO `annotations.json` + `train_val_test_distribution_file.json`) in a Google Drive folder at `MyDrive/yolo_dataset/`.
4. Run all cells sequentially.

### Running Inference on Your Own Images

```python
from ultralytics import YOLO

model = YOLO("Yolo26x_litter.pt")
results = model.predict(source="your_image.jpg", conf=0.5, save=True)
```

### Generating a Location Heatmap

Pass a list of images with GPS coordinates to `run_prediction_pipeline`:

```python
images = [
    {"image_path": "img1.jpg", "latitude": 50.6010, "longitude": -3.4290},
    {"image_path": "img2.jpg", "latitude": 50.6020, "longitude": -3.4300},
]
run_prediction_pipeline(images, model_path="Yolo26x_litter.pt")
```

## Model

- **Architecture:** YOLOv8x
- **Classes:** 1 (`litter`)
- **Training data:** 540 images (COCO-format annotations, converted to YOLO)
- **Validation / Test split:** 116 images each
- **Inference confidence threshold:** 0.5

## Dataset

The dataset follows COCO annotation format with a single `rubbish` category, remapped to `litter` (class ID `0`) during conversion. Images were captured in and around Exeter, UK.

## License

This project is for academic research purposes. Please contact the author before reusing the model weights or dataset.
