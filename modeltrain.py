import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def preprocess_image(image_path, target_size=(100, 100)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    return image


positive_folder = r"R:\Internship Project\BRAIN\new_dataset\archive\Training\meningioma"
negative_folder = r"R:\Internship Project\BRAIN\new_dataset\archive\Training\notumor"

positive_images = [os.path.join(positive_folder, img) for img in os.listdir(positive_folder) if img.endswith('.jpg')]
negative_images = [os.path.join(negative_folder, img) for img in os.listdir(negative_folder) if img.endswith('.jpg')]


positive_features = np.array([preprocess_image(img) for img in positive_images])
negative_features = np.array([preprocess_image(img) for img in negative_images])


positive_labels = np.ones(len(positive_features))
negative_labels = np.zeros(len(negative_features))


X = np.concatenate((positive_features, negative_features), axis=0)
y = np.concatenate((positive_labels, negative_labels), axis=0)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
false_positive_rate = conf_matrix[0, 1] / (conf_matrix[0, 1] + conf_matrix[0, 0])

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("False Positive Rate:", false_positive_rate)

# Save the model
model.save('brain_cancer_cnn_model.h5')
