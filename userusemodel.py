import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Function to preprocess image
def preprocess_image(image_path, target_size=(100, 100)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    return image

# Load the saved model
model = load_model('brain_cancer_cnn_model.h5')

# Function to predict tumor or not
def predict_tumor(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    # Perform inference
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    if prediction[0][0] > 0.5:
        return "Tumor"
    else:
        return "No Tumor"

# Function to handle button click event
def open_file_dialog():
    file_path = filedialog.askopenfilename()
    if file_path:
        prediction = predict_tumor(file_path)
        messagebox.showinfo("Prediction Result", f"The model predicts: {prediction}")

# Create Tkinter window
root = tk.Tk()
root.title("Brain Tumor Detection")

# Create button to open file dialog
button = tk.Button(root, text="Select Image", command=open_file_dialog)
button.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
