# COVID-19 Chest X-ray Classification

This project aims to classify chest X-ray images as either COVID-19 positive or Non-COVID using transfer learning with pre-trained CNNs. The dataset contains X-ray images. Models used include MobileNetV2 and EfficientNetB0. My best model EfficientNetB0 achieved 80% test accuracy, though struggled with COVID-19 sensitivity.

## Overview

* The task is to detect COVID‑19 from chest X‑ray images using deep learning. I frame this as a binary image classification problem with two classes:
  * COVID‑19
  * Non‑COVID
 
The approach uses transfer learning with pre‑trained CNNs (MobileNetV2, EfficientNetB0), fine‑tuning, and class weighting to address dataset imbalance.
My final EfficientNetB0 model achieved 80% test accuracy despite significant class imbalance.

## Summary of Workdone

### Data

* Data:
  * Type: CGrayscale/monochrome chest X-ray images (resized to 224×224).
  * Size: 372 images in total (328 usable after cleaning).
  * Instances (after preprocessing):
    * Train: 249 images
    * Validation: 52 images
    * Test: 50 images
  * Class Distribution:
    * COVID‑19: 265 images
    * Non‑COVID: 63 images
  * Class Distribution Visualization:
  * Sample Raw Images (COVID vs Non‑COVID):
<img width="1189" height="590" alt="download" src="https://github.com/user-attachments/assets/e6ca1368-0487-4fc9-9a74-c2c048ba3644" />
  Figure 1: Raw chest X‑ray images showing COVID‑19 and Non‑COVID cases.

#### Preprocessing / Clean up

* Removed images without labels or corrupted files.
* Mapped metadata to images by filename.
* Resized all images to 224×224 pixels.
* Normalized pixel values to range [0,1].
* Applied data augmentation: random flips, rotations, and zooms.

<img width="1097" height="784" alt="download-1" src="https://github.com/user-attachments/assets/17e40b7c-7875-43fe-a24c-e78c75495068" />
Figure 2: Preprocessed chest X‑ray images after resizing, normalization, and augmentation.


### Problem Formulation

* Input: 224×224 chest X-ray image.
* Output: Binary label (1 = COVID‑19, 0 = Non-COVID).
* Task: Binary image classification.
* Metrics: Accuracy, Precision, Recall, F1‑score.

### Models Tried

1. MobileNetV2
   * Used as baseline transfer learning model.
   * Base model frozen, custom dense layers added.
   * Performance plateaued at 79% accuracy.
2. EfficientNetB0
   * Similar transfer learning setup.
   * Better generalization.
   * Final test accuracy: 80%
  
### Training

* Hardware: Google Colab GPU.
* Epochs: 10 (baseline), 5 (fine‑tuning).
* Batch Size: 32.
* Optimizer: Adam.
* Class weighting applied to handle imbalance.
Training & Validation Curves (EfficientNetB0 Final Run):
<img width="1183" height="484" alt="download-4" src="https://github.com/user-attachments/assets/8d817490-bfc9-463a-b4ce-f143fbbb8b98" />
Figure 3: Training vs validation accuracy and loss.

1. Left Graph (Accuracy over Epochs):
   * The training accuracy quickly increases and then remains constant around 82%.
   * The validation accuracy starts high 83% and stays stable, showing no overfitting
    
2. Right Graph (Loss over Epochs):
   * Training loss drops sharply at first, then levels off with minor fluctuations.
   * Validation loss steadily decreases and stabilizes, indicating good generalization.
### Performance Evaluation

* Confusion Matrix:

<img width="528" height="479" alt="download-3" src="https://github.com/user-attachments/assets/e617879c-314b-431a-a8bd-e63540465f07" />

* Classification Report:

<img width="530" height="126" alt="Screenshot 2025-08-06 at 12 50 27 AM" src="https://github.com/user-attachments/assets/b2ec987a-819b-4719-8855-92ba33f5fd2d" />

This report shows that while the model correctly identified all Non-COVID cases, it failed to detect any COVID-19 cases, indicating poor performance on the minority class despite a high overall accuracy.

* Grad‑CAM Heatmaps:

<img width="639" height="324" alt="download-2" src="https://github.com/user-attachments/assets/96eb83da-67c6-4d62-9d72-92b5e711eb7c" />

Figure 4: Grad‑CAM visualizations highlighting regions influencing model predictions.


### Conclusions

* Strengths: Stable training, no overfitting, decent overall accuracy.
* Weaknesses: Extremely poor COVID‑19 recall fails to identify positives reliably.
* Main Cause: Severe class imbalance + small dataset size.
* Implication: In real medical contexts, missing COVID‑19 cases can be dangerous recall must be prioritized.


### Future Work

* Collect a larger, more balanced dataset.
* Explore other architectures (ResNet50, DenseNet).
* Apply advanced augmentation or synthetic data generation.
* Test on real‑world hospital datasets.

## How to reproduce results

* Download the dataset from Kaggle.
* Run the notebook covid_xray_classification.ipynb.
* Follow preprocessing, training, and evaluation cells.

### Overview of files in repository

* data_preprocessing.ipynb – Cleans and preprocesses dataset.
* train_mobilenetv2.ipynb – MobileNetV2 baseline training.
* train_efficientnetb0.ipynb – EfficientNetB0 final training and evaluation.
* utils.py – Helper functions for preprocessing and visualization.

### Software Setup
* Python 3.10+
* TensorFlow 2.x
* scikit‑learn
* Matplotlib, NumPy, Pillow


## Citations

* PBachr, R. COVID Chest X‑Ray Dataset. Kaggle.
* Tan, M., Le, Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
* Kaggle Data: https://www.kaggle.com/datasets/bachrr/covid-chest-xray/data
