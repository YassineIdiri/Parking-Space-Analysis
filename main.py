import os
import cv2
import numpy as np
from sklearn.metrics import pairwise_distances

# Define the size to which all images will be resized
IMAGE_SIZE = (128, 128)  # Example size, change as needed

def compute_lbp(image):
    """Compute Local Binary Pattern representation of the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_image = np.zeros_like(gray, dtype=np.uint8)
    for r in range(1, gray.shape[0]-1):
        for c in range(1, gray.shape[1]-1):
            center = gray[r, c]
            binary_string = ''
            for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:
                binary_string += '1' if gray[r+dr, c+dc] > center else '0'
            lbp_value = int(binary_string, 2)
            lbp_image[r, c] = lbp_value
    return lbp_image.flatten()

def load_images_from_folder(folder):
    """Load images from a folder and return them in a list."""
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img_resized = cv2.resize(img, IMAGE_SIZE)  # Resize the image
                    images.append(compute_lbp(img_resized))
                    labels.append(label)
    return np.array(images), np.array(labels)

def classify_images(test_images, train_images, train_labels):
    """Classify the test images using Euclidean distance."""
    results = []
    for test_img in test_images:
        distances = pairwise_distances(test_img.reshape(1, -1), train_images)
        nearest_idx = np.argmin(distances)
        results.append(train_labels[nearest_idx])
    return results

def evaluate_model(test_images, test_labels, train_images, train_labels):
    """Evaluate the model and calculate the accuracy."""
    predictions = classify_images(test_images, train_images, train_labels)
    accuracy = np.sum(predictions == test_labels) / len(test_labels) * 100
    return accuracy

# Main function to run the above code
if __name__ == "__main__":
    # Load training data
    train_images, train_labels = load_images_from_folder('images/train')  # Path to your images folder

    # Load test data
    test_images, test_labels = load_images_from_folder('images/test')  # Path to your test images folder

    # Evaluate the model
    accuracy = evaluate_model(test_images, test_labels, train_images, train_labels)

    print(f"Accuracy: {accuracy:.2f}%")
