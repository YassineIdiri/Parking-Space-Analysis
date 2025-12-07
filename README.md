![parking](https://github.com/user-attachments/assets/6a4a3652-e2d3-4502-b0c0-251c2bf8d0c9)
## 🚀 Getting Started

### Install
Clone the repository:
``` bash
pip install opencv-python
pip install numpy
pip install scikit-learn
```

The project aims to classify images of parking spaces as either "occupied" or "free" using machine learning techniques. It utilizes image processing and feature extraction methods, specifically Local Binary Patterns (LBP) and Histogram of Oriented Gradients (HOG), combined with a Random Forest classifier to achieve this goal. The core of the project lies in extracting both texture and shape-based features from images to build an accurate classification model.

### 📄 Report

![park1](https://github.com/user-attachments/assets/77d61ccd-86ed-48c0-9e79-d9807ca450a0)

### ⚙️ Technologies Used

Python: The programming language used for implementing the entire project.
OpenCV: A computer vision library for image processing tasks such as reading, resizing, and converting images to grayscale.
scikit-image: A library that provides efficient implementations for image processing functions like Local Binary Patterns (LBP) and Histogram of Oriented Gradients (HOG).
NumPy: A fundamental package for numerical operations and handling arrays.
scikit-learn: A machine learning library used for classification, hyperparameter tuning (GridSearchCV), normalization, and evaluation metrics.

### 🔎 Approach

Image Feature Extraction:

Local Binary Patterns (LBP): Extracts texture information by identifying local patterns in grayscale images. An LBP histogram is computed to represent texture-based features.
Histogram of Oriented Gradients (HOG): Extracts shape-based features by analyzing the gradients in different directions within an image.
Data Normalization: The features extracted using LBP and HOG are normalized using MinMaxScaler to bring all values within the range [0, 1].

Classification:

A Random Forest Classifier is used to classify the features into "occupied" or "free". The classifier’s hyperparameters are optimized using GridSearchCV to find the best settings for n_estimators, max_depth, and min_samples_split.
Model Evaluation:

The model is evaluated on test images using accuracy, precision, recall, F1-score, and a confusion matrix.

### 📈 Outcome 

The project achieves a solid level of accuracy on the test set, showing that combining LBP and HOG features with a Random Forest classifier is effective for this task.
This approach is effective for scenarios where differentiating between occupied and free parking spaces is required based on image data.
