\# Lung Disease Classification using Deep Learning



This project focuses on classifying lung diseases from medical images using deep learning with PyTorch.  

The goal is to build an end-to-end image classification pipeline that can assist in early diagnosis by automatically analyzing chest X-ray images.

This repository uses an EfficientNet-B0 model to classify X-rays into four categories: COVID19, Normal, Pneumonia, and Tuberculosis.


---



\## Problem Statement

Lung diseases such as pneumonia, tuberculosis, and COVID-19 can be life-threatening if not diagnosed early. Manual analysis of medical images is time-consuming and prone to human error.  

This project aims to develop a \*\*deep learning-based image classification system\*\* that can accurately identify lung diseases from medical images.



---



\## Dataset

\- Medical imaging dataset (Chest X-ray)

\- Classes: 'COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS'

\- Images resized and normalized before training



>  Dataset is \*\*not included\*\* in this repository due to size limitations.

---



\##  Approach

1\. Data preprocessing and augmentation

2\. Convolutional Neural Network (CNN) based classification

3\. Training using PyTorch

4\. Model evaluation using accuracy and loss metrics



---








\##  Results

\- Training and validation accuracy plotted

\- Confusion matrix for performance analysis




---



## How to run
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Place your data in a `./data` folder organized by class.
4. Run training: `python train.py`


