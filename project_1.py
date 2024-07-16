import opencv
import os
import numpy as np
import cv2

# Define folder paths
train_folder = '/home/pavaniankam/Desktop/paathu/lion_train_data/'
val_folder = '/home/pavaniankam/Desktop/paathu/lion_validation_data/'
test_folder = '/home/pavaniankam/Desktop/paathu/lion_test_data/'
# Image size for resizing
image_size = (250, 250)

# Load training data
X_train = []
y_train = []
for filename in os.listdir(train_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(train_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, image_size)
            X_train.append(img.flatten())
            y_train.append(1 if filename.startswith('class1') else 0)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Load validation data
X_val = []
y_val = []
for filename in os.listdir(val_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(val_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, image_size)
            X_val.append(img.flatten())
            y_val.append(1 if filename.startswith('class1') else 0)

X_val = np.array(X_val)
y_val = np.array(y_val)

# Load test data
X_test = []
y_test = []
for filename in os.listdir(test_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(test_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, image_size)
            X_test.append(img.flatten())
            y_test.append(1 if filename.startswith('class1') else 0)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Initialize weights and bias
weights = np.zeros(X_train.shape[1])
bias = 0

# Hyperparameters
learning_rate = 0.01
epochs = 50

# Training the Perceptron
for epoch in range(epochs):
    # Training
    for i in range(X_train.shape[0]):
        output = np.dot(X_train[i], weights) + bias
        prediction = 1 if output >= 0 else 0
        weights += learning_rate * (y_train[i] - prediction) * X_train[i]
        bias += learning_rate * (y_train[i] - prediction)
    
    # Compute training accuracy
    correct_train = np.sum(np.where(np.dot(X_train, weights) + bias >= 0, 1, 0) == y_train)
    accuracy_train = (correct_train / float(X_train.shape[0])) * 100
    
    # Compute validation accuracy
    correct_val = np.sum(np.where(np.dot(X_val, weights) + bias >= 0, 1, 0) == y_val)
    accuracy_val = (correct_val / float(X_val.shape[0])) * 100
    
    print(f"Epoch: {epoch+1}")
    print(f"Train Accuracy: {accuracy_train:.2f}%")
    print(f"Validation Accuracy: {accuracy_val:.2f}%")
    print()

# Test the Perceptron on test data
correct_test = np.sum(np.where(np.dot(X_test, weights) + bias >= 0, 1, 0) == y_test)
accuracy_test = (correct_test / float(X_test.shape[0])) * 100
print(f"Test Accuracy: {accuracy_test:.2f}%")


