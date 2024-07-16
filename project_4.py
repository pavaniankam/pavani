
import os
import numpy as np
from PIL import Image

# Parameters
train_folder_path = '/home/pavaniankam/Desktop/paathu/lion_train_data/'
val_folder_path = '/home/pavaniankam/Desktop/paathu/lion_validation_data/'
test_folder_path = '/home/pavaniankam/Desktop/paathu/lion_test_data/'

# Load and prepare training data
files_train = os.listdir(train_folder_path)
X_train = []
y_train = []

for file_name in files_train:
    file_path = os.path.join(train_folder_path, file_name)
    img = Image.open(file_path).convert('L')
    img = img.resize((64, 64))
    arr = np.array(img)
    features = arr.flatten() / 255.0
    label = 1 if 'class_name' in file_name else 0
    X_train.append(features)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Load and prepare validation data
files_val = os.listdir(val_folder_path)
X_val = []
y_val = []

for file_name in files_val:
    file_path = os.path.join(val_folder_path, file_name)
    img = Image.open(file_path).convert('L')
    img = img.resize((64, 64))
    arr = np.array(img)
    features = arr.flatten() / 255.0
    label = 1 if 'class_name' in file_name else 0
    X_val.append(features)
    y_val.append(label)

X_val = np.array(X_val)
y_val = np.array(y_val)

# Load and prepare test data
files_test = os.listdir(test_folder_path)
X_test = []
y_test = []

for file_name in files_test:
    file_path = os.path.join(test_folder_path, file_name)
    img = Image.open(file_path).convert('L')
    img = img.resize((64, 64))
    arr = np.array(img)
    features = arr.flatten() / 255.0
    label = 1 if 'class_name' in file_name else 0
    X_test.append(features)
    y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Shuffle training data
shuffle_index = np.random.permutation(len(X_train))
X_train = X_train[shuffle_index]
y_train = y_train[shuffle_index]

# Parameters
num_features = X_train.shape[1]
learning_rate = 0.1
lambda_val = 0.01
dropout_rate = 0.5  # Dropout rate (fraction of weights to drop out)
epochs = 10

# Initialize Perceptron with Dropout Regularization
weights_dropout = np.zeros(num_features + 1)  # +1 for the bias

# Training with Dropout Regularization
for epoch in range(epochs):
    for features, label in zip(X_train, y_train):
        # Dropout regularization: Create a dropout mask
        dropout_mask = np.random.binomial(1, 1 - dropout_rate, size=weights_dropout[1:].shape)
        masked_weights_dropout = weights_dropout[1:] * dropout_mask
        
        activation = np.dot(features, masked_weights_dropout) + weights_dropout[0]
        prediction = 1 if activation >= 0 else 0
        error = label - prediction
        weights_dropout[1:] += learning_rate * error * features
        weights_dropout[0] += learning_rate * error

# Evaluate the perceptron with Dropout Regularization on training data
correct_train_dropout = 0
for features, label in zip(X_train, y_train):
    dropout_mask = np.random.binomial(1, 1 - dropout_rate, size=weights_dropout[1:].shape)
    masked_weights_dropout = weights_dropout[1:] * dropout_mask

    activation = np.dot(features, masked_weights_dropout) + weights_dropout[0]
    prediction = 1 if activation >= 0 else 0
    if prediction == label:
        correct_train_dropout += 1
train_accuracy_dropout = correct_train_dropout / len(y_train)

# Evaluate the perceptron with Dropout Regularization on validation data
correct_val_dropout = 0
for features, label in zip(X_val, y_val):
    dropout_mask = np.random.binomial(1, 1 - dropout_rate, size=weights_dropout[1:].shape)
    masked_weights_dropout = weights_dropout[1:] * dropout_mask

    activation = np.dot(features, masked_weights_dropout) + weights_dropout[0]
    prediction = 1 if activation >= 0 else 0
    if prediction == label:
        correct_val_dropout += 1
val_accuracy_dropout = correct_val_dropout / len(y_val)

# Evaluate the perceptron with Dropout Regularization on testing data
correct_test_dropout = 0
for features, label in zip(X_test, y_test):
    dropout_mask = np.random.binomial(1, 1 - dropout_rate, size=weights_dropout[1:].shape)
    masked_weights_dropout = weights_dropout[1:] * dropout_mask

    activation = np.dot(features, masked_weights_dropout) + weights_dropout[0]
    prediction = 1 if activation >= 0 else 0
    if prediction == label:
        correct_test_dropout += 1
test_accuracy_dropout = correct_test_dropout / len(y_test)

print("Dropout Regularization:")
print(f"Training Accuracy: {train_accuracy_dropout:.0%}")
print(f"Validation Accuracy: {val_accuracy_dropout:.0%}")
print(f"Testing Accuracy: {test_accuracy_dropout:.0%}")

# Initialize Perceptron with L1 regularization
weights_l1 = np.zeros(num_features + 1)  # +1 for the bias

# Training with L1 regularization
for epoch in range(epochs):
    for features, label in zip(X_train, y_train):
        activation = np.dot(features, weights_l1[1:]) + weights_l1[0]
        prediction = 1 if activation >= 0 else 0
        error = label - prediction
        weights_l1[1:] += learning_rate * error * features
        weights_l1[0] += learning_rate * error
        
        # Apply L1 regularization
        weights_l1[1:] -= lambda_val * np.sign(weights_l1[1:])

# Evaluate the perceptron with L1 regularization on training data
correct_train_l1 = 0
for features, label in zip(X_train, y_train):
    activation = np.dot(features, weights_l1[1:]) + weights_l1[0]
    prediction = 1 if activation >= 0 else 0
    if prediction == label:
        correct_train_l1 += 1
train_accuracy_l1 = correct_train_l1 / len(y_train)

# Evaluate the perceptron with L1 regularization on validation data
correct_val_l1 = 0
for features, label in zip(X_val, y_val):
    activation = np.dot(features, weights_l1[1:]) + weights_l1[0]
    prediction = 1 if activation >= 0 else 0
    if prediction == label:
        correct_val_l1 += 1
val_accuracy_l1 = correct_val_l1 / len(y_val)

# Evaluate the perceptron with L1 regularization on testing data
correct_test_l1 = 0
for features, label in zip(X_test, y_test):
    activation = np.dot(features, weights_l1[1:]) + weights_l1[0]
    prediction = 1 if activation >= 0 else 0
    if prediction == label:
        correct_test_l1 += 1
test_accuracy_l1 = correct_test_l1 / len(y_test)

print("\nL1 Regularization:")
print(f"Training Accuracy: {train_accuracy_l1:.0%}")
print(f"Validation Accuracy: {val_accuracy_l1:.0%}")
print(f"Testing Accuracy: {test_accuracy_l1:.0%}")

# Initialize Perceptron with L2 regularization
weights_l2 = np.zeros(num_features + 1)  # +1 for the bias

# Training with L2 regularization
for epoch in range(epochs):
    for features, label in zip(X_train, y_train):
        activation = np.dot(features, weights_l2[1:]) + weights_l2[0]
        prediction = 1 if activation >= 0 else 0
        error = label - prediction
        weights_l2[1:] += learning_rate * error * features
        weights_l2[0] += learning_rate * error
        
        # Apply L2 regularization
        weights_l2[1:] -= lambda_val * weights_l2[1:]

# Evaluate the perceptron with L2 regularization on training data
correct_train_l2 = 0
for features, label in zip(X_train, y_train):
    activation = np.dot(features, weights_l2[1:]) + weights_l2[0]
    prediction = 1 if activation >= 0 else 0
    if prediction == label:
        correct_train_l2 += 1
train_accuracy_l2 = correct_train_l2 / len(y_train)

# Evaluate the perceptron with L2 regularization on validation data
correct_val_l2 = 0
for features, label in zip(X_val, y_val):
    activation = np.dot(features, weights_l2[1:]) + weights_l2[0]
    prediction = 1 if activation >= 0 else 0
    if prediction == label:
        correct_val_l2 += 1
val_accuracy_l2 = correct_val_l2 / len(y_val)

# Evaluate the perceptron with L2 regularization on testing data
correct_test_l2 = 0
for features, label in zip(X_test, y_test):
    activation = np.dot(features, weights_l2[1:]) + weights_l2[0]
    prediction = 1 if activation >= 0 else 0
    if prediction == label:
        correct_test_l2 += 1
test_accuracy_l2 = correct_test_l2 / len(y_test)

print("\nL2 Regularization:")
print(f"Training Accuracy: {train_accuracy_l2:.0%}")
print(f"Validation Accuracy: {val_accuracy_l2:.0%}")
print(f"Testing Accuracy: {test_accuracy_l2:.0%}")
