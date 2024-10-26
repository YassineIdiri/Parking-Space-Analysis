import os
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import local_binary_pattern, hog
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Parameters
IMAGE_SIZE = (128, 128)
LBP_POINTS = 8
LBP_RADIUS = 1
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

def compute_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

def compute_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm='L2-Hys',
        visualize=False,
        transform_sqrt=True,
        feature_vector=True
    )
    return hog_features

def load_images_from_folder(folder, use_lbp=True, use_hog=True):
    features = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img_resized = cv2.resize(img, IMAGE_SIZE)
                    feature_list = []
                    if use_lbp:
                        lbp_features = compute_lbp(img_resized)
                        feature_list.append(lbp_features)
                    if use_hog:
                        hog_features = compute_hog(img_resized)
                        feature_list.append(hog_features)
                    combined_features = np.hstack(feature_list)
                    features.append(combined_features)
                    labels.append(label)
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    # Load data
    train_features, train_labels = load_images_from_folder('images/train')
    test_features, test_labels = load_images_from_folder('images/test')

    # Normalize features
    scaler = MinMaxScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    # Optionally skip PCA

    # Try Random Forest Classifier
    parameters = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5, n_jobs=-1)
    clf.fit(train_features, train_labels)
    print(f"Best parameters found: {clf.best_params_}")

    # Evaluate on test data
    predictions = clf.predict(test_features)
    accuracy = np.mean(predictions == test_labels) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    # Detailed evaluation
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, predictions))
