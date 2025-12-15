AI-Based Blood Group Prediction and Classification Using Image Processing
Overview

This project implements an AI-driven pipeline to predict and classify blood groups from blood sample images. The system processes raw image data, extracts meaningful visual features, and applies machine learning or deep learning models to learn patterns associated with different blood groups.

The focus of this project is on the processing flow, model learning, and prediction logic, rather than on application deployment.

How the System Works
1. Image Acquisition

Blood sample images are collected from a predefined dataset. Each image is labeled with its corresponding blood group, which is essential for supervised learning. Images may vary in size, lighting conditions, and noise levels.

2. Image Preprocessing

Raw images are not directly suitable for model training. Preprocessing improves image quality and ensures uniformity across the dataset.

Preprocessing steps include:

Resizing images to a fixed dimension

Converting images to grayscale (if required)

Noise removal using filters

Normalization of pixel values

Contrast enhancement to highlight important regions

These steps reduce unwanted variations and improve feature learning.

3. Region of Interest (ROI) Extraction

If required, specific regions of the blood image are isolated to remove background information. This step ensures the model focuses only on meaningful blood patterns instead of irrelevant visual noise.

4. Feature Extraction

Two approaches can be used:

Traditional Image Processing Approach

Texture features

Edge detection outputs

Color intensity distributions

Deep Learning Approach

Convolutional Neural Networks (CNNs) automatically learn hierarchical features

Lower layers capture edges and textures

Higher layers capture complex patterns relevant to blood group differentiation

Feature extraction converts visual data into numerical representations that the model can learn from.

5. Model Training

The extracted features are fed into a classification model.

Machine learning models learn decision boundaries between blood groups

Deep learning models learn features and classification jointly

The dataset is split into training and testing sets

Loss functions guide the optimization process

Model weights are updated iteratively to minimize prediction error

6. Model Evaluation

The trained model is evaluated using unseen test images.

Evaluation metrics include:

Accuracy

Precision

Recall

Confusion matrix

This step verifies how well the model generalizes to new data.

7. Prediction Phase

For a new input image:

The same preprocessing steps are applied

Features are extracted

The trained model predicts the most probable blood group

The output is generated based on learned visual patterns.

Technical Highlights

End-to-end image processing and AI pipeline

Consistent preprocessing for reliable learning

Feature-driven and data-driven classification

Modular design for easy experimentation

Tech Stack

Python

OpenCV

TensorFlow / Keras

Scikit-learn

NumPy, Pandas

Matplotlib

Project Status

Academic / Learning Project
