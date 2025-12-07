![parking](https://github.com/user-attachments/assets/6a4a3652-e2d3-4502-b0c0-251c2bf8d0c9)
## 🚀 Getting Started

### Install
Clone the repository:
``` bash
pip install opencv-python
pip install numpy
pip install scikit-learn
```
The project aims to classify images of parking spaces as either “occupied” or “free” using machine learning techniques. It relies on image processing and feature extraction based on Local Binary Patterns (LBP) to capture the texture of each parking space. The extracted LBP features are then used to train a classification model to accurately distinguish between occupied and free spots.

### 📄 Report

![park1](https://github.com/user-attachments/assets/77d61ccd-86ed-48c0-9e79-d9807ca450a0)

### ⚙️ Technologies Used

⚙️ Technologies Used

Python – main programming language used for the implementation.

OpenCV – for image processing tasks such as loading images, resizing, grayscale conversion, and ROI extraction.

scikit-image – for computing Local Binary Patterns (LBP) and generating LBP histograms.

NumPy – for numerical operations and array manipulation.

scikit-learn – for splitting data, training the classifier, and evaluating performance.

### 🔎 Approach

Image Feature Extraction (LBP Only)
Images of parking spaces are preprocessed (grayscale, normalization, resizing).

Local Binary Patterns (LBP) are computed for each image to capture texture information.

An LBP histogram is generated for each image and used as the feature vector for classification.

Classification
A simple ML classifier (e.g., SVM, KNN, or Logistic Regression) is trained using the LBP histograms.

The dataset is split into training and testing sets to evaluate generalization.

Model Evaluation
The model is evaluated using metrics such as accuracy and a confusion matrix.

The results show how well the system distinguishes between “occupied” and “free” parking spaces.

### 📈 Outcome

The system classifies parking spaces based on texture patterns extracted using LBP, achieving a good level of accuracy given the simplicity of the method. This demonstrates the effectiveness of texture-based image descriptors for lightweight computer vision tasks such as parking occupancy detection.
