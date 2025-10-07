# Multi-Domain Feature Fusion for Deepfake Detection

This repository contains the implementation for the undergraduate final year project **"Multi-Domain Feature Fusion for Deepfake Detection"**, submitted to the **Department of Electronics & Computer Engineering**, **Pulchowk Campus**, **IOE**, **Tribhuvan University**.


## Abstract

The rapid advancement of generative AI has made deepfake creation more accessible and sophisticated, posing a significant threat to information integrity.  
This project implements and evaluates a deepfake detection system that **fuses features from three distinct domains** to enhance detection robustness and generalization.

We compare the individual performance of **spatial**, **frequency**, and **semantic** feature-based classifiers against a **unified fusion model** that leverages their combined strengths.

### 🔍 Feature Extraction Domains
- **Spatial Domain:** Utilizes a pre-trained **XceptionNet** to capture fine-grained, pixel-level artifacts and texture inconsistencies.  
- **Frequency Domain:** Applies **Fast Fourier Transform (FFT)** to analyze the frequency spectrum of images, identifying unnatural periodic patterns introduced during generation.  
- **Semantic Domain:** Employs the **DINOv2 Vision Transformer** to extract high-level semantic features, detecting logical inconsistencies like unnatural facial geometry or expressions.

---

## Project Pipeline

The project follows a **four-stage pipeline**:

1. **Metadata Generation**  
   Scans dataset directories to create a master CSV file (`ffpp_metadata.csv`) cataloging all video paths and corresponding labels (real/fake).

2. **Preprocessing**  
   Extracts frames from videos, detects and crops faces using **MTCNN**, and saves them as individual images.

3. **Feature Extraction**  
   Processes cropped faces through the three pipelines (Spatial, Frequency, Semantic) to generate and save numerical feature vectors (`.npy` files).

4. **Training & Evaluation**  
   Trains separate MLP classifiers on each individual feature set and on the concatenated (fused) feature set, evaluates their performance, and prints a comparison table.


##  Project File Structure

```
├── FaceForensics++/              # Contains dataset videos
├── download_and_prepare_ffpp.py  # 1. Generates metadata
├── preprocess.py                 # 2. Extracts faces
├── feature_extractors.py         # 3. Generates feature vectors
├── models_and_dataloaders.py     # Defines model and dataset loaders
├── train_evaluate.py             # 4. Trains, evaluates, compares models
└── requirements.txt              # Dependencies
```

---