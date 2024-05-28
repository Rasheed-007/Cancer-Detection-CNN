import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Function to preprocess image
def preprocess_image(image_path, target_size=(100, 100)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    return image

# Load the saved model
model = load_model('brain_cancer_cnn_model.h5')

# Path to cancerous and non-cancerous test image folders
cancerous_folder = r"R:\Internship Project\BRAIN\new_dataset\archive\Testing\meningioma"
non_cancerous_folder = r"R:\Internship Project\BRAIN\new_dataset\archive\Testing\notumor"

# Initialize variables for accuracy calculation
total_test_samples = 0
correct_predictions = 0

# Iterate through cancerous images
for filename in os.listdir(cancerous_folder):
    if filename.endswith('.jpg'):
        image_path = os.path.join(cancerous_folder, filename)
        # Preprocess the test image
        test_image = preprocess_image(image_path)
        # Perform inference
        prediction = model.predict(np.expand_dims(test_image, axis=0))
        # Increment total test samples
        total_test_samples += 1
        # Check prediction and update correct predictions
        if prediction[0][0] > 0.5:
            correct_predictions += 1

# Iterate through non-cancerous images
for filename in os.listdir(non_cancerous_folder):
    if filename.endswith('.jpg'):
        image_path = os.path.join(non_cancerous_folder, filename)
        # Preprocess the test image
        test_image = preprocess_image(image_path)
        # Perform inference
        prediction = model.predict(np.expand_dims(test_image, axis=0))
        # Increment total test samples
        total_test_samples += 1
        # Check prediction and update correct predictions
        if prediction[0][0] <= 0.5:
            correct_predictions += 1

# Calculate test accuracy
test_accuracy = correct_predictions / total_test_samples
print("Test Accuracy:", test_accuracy)





