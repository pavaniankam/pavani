
import os
import numpy as np
from PIL import Image

# Parameters
train_folder_path = '/home/pavaniankam/Desktop/paathu/lion_train_data/'
val_folder_path = '/home/pavaniankam/Desktop/paathu/lion_validation_data/'
test_folder_path = '/home/pavaniankam/Desktop/paathu/lion_test_data/'

# Load and prepare data function
def load_and_prepare_data(folder_path):
    files = os.listdir(folder_path)
    X = []
    y = []
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        img = Image.open(file_path).convert('L')
        img = img.resize((64, 64))
        arr = np.array(img)
        features = arr.flatten() / 255.0
        label = 1 if 'class_name' in file_name else 0
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

# Load data
X_train, y_train = load_and_prepare_data(train_folder_path)
X_val, y_val = load_and_prepare_data(val_folder_path)
X_test, y_test = load_and_prepare_data(test_folder_path)

# Shuffle training data
shuffle_index = np.random.permutation(len(X_train))
X_train = X_train[shuffle_index]
y_train = y_train[shuffle_index]

# Parameters
num_features = X_train.shape[1]
learning_rate = 0.01
epochs = 10
batch_size = 32
momentum_factor = 0.9
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Helper functions
def evaluate(weights, X, y):
    correct = 0
    for features, label in zip(X, y):
        activation = np.dot(features, weights[1:]) + weights[0]
        prediction = 1 if activation >= 0 else 0
        if prediction == label:
            correct += 1
    return correct / len(y)

# Optimizer: Stochastic Gradient Descent (SGD)
weights_sgd = np.zeros(num_features + 1)
for epoch in range(epochs):
    for features, label in zip(X_train, y_train):
        activation = np.dot(features, weights_sgd[1:]) + weights_sgd[0]
        prediction = 1 if activation >= 0 else 0
        error = label - prediction
        weights_sgd[1:] += learning_rate * error * features
        weights_sgd[0] += learning_rate * error

# Evaluate SGD
train_accuracy_sgd = evaluate(weights_sgd, X_train, y_train)
val_accuracy_sgd = evaluate(weights_sgd, X_val, y_val)
test_accuracy_sgd = evaluate(weights_sgd, X_test, y_test)

print("Stochastic Gradient Descent (SGD):")
print(f"Training Accuracy: {train_accuracy_sgd:.0%}")
print(f"Validation Accuracy: {val_accuracy_sgd:.0%}")
print(f"Testing Accuracy: {test_accuracy_sgd:.0%}")

# Optimizer: Mini-Batch Gradient Descent
weights_mbgd = np.zeros(num_features + 1)
for epoch in range(epochs):
    permutation = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[permutation]
    y_train_shuffled = y_train[permutation]
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train_shuffled[i:i+batch_size]
        batch_y = y_train_shuffled[i:i+batch_size]
        for features, label in zip(batch_X, batch_y):
            activation = np.dot(features, weights_mbgd[1:]) + weights_mbgd[0]
            prediction = 1 if activation >= 0 else 0
            error = label - prediction
            weights_mbgd[1:] += learning_rate * error * features
            weights_mbgd[0] += learning_rate * error

# Evaluate Mini-Batch Gradient Descent
train_accuracy_mbgd = evaluate(weights_mbgd, X_train, y_train)
val_accuracy_mbgd = evaluate(weights_mbgd, X_val, y_val)
test_accuracy_mbgd = evaluate(weights_mbgd, X_test, y_test)

print("\nMini-Batch Gradient Descent:")
print(f"Training Accuracy: {train_accuracy_mbgd:.0%}")
print(f"Validation Accuracy: {val_accuracy_mbgd:.0%}")
print(f"Testing Accuracy: {test_accuracy_mbgd:.0%}")

# Optimizer: Momentum
weights_momentum = np.zeros(num_features + 1)
velocity = np.zeros(num_features + 1)
for epoch in range(epochs):
    for features, label in zip(X_train, y_train):
        activation = np.dot(features, weights_momentum[1:]) + weights_momentum[0]
        prediction = 1 if activation >= 0 else 0
        error = label - prediction
        gradient = np.concatenate(([error], error * features))
        velocity = momentum_factor * velocity + learning_rate * gradient
        weights_momentum += velocity

# Evaluate Momentum
train_accuracy_momentum = evaluate(weights_momentum, X_train, y_train)
val_accuracy_momentum = evaluate(weights_momentum, X_val, y_val)
test_accuracy_momentum = evaluate(weights_momentum, X_test, y_test)

print("\nMomentum:")
print(f"Training Accuracy: {train_accuracy_momentum:.0%}")
print(f"Validation Accuracy: {val_accuracy_momentum:.0%}")
print(f"Testing Accuracy: {test_accuracy_momentum:.0%}")

# Optimizer: Adam
weights_adam = np.zeros(num_features + 1)
m = np.zeros(num_features + 1)
v = np.zeros(num_features + 1)
for epoch in range(epochs):
    for features, label in zip(X_train, y_train):
        activation = np.dot(features, weights_adam[1:]) + weights_adam[0]
        prediction = 1 if activation >= 0 else 0
        error = label - prediction
        gradient = np.concatenate(([error], error * features))
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient ** 2
        m_hat = m / (1 - beta1 ** (epoch + 1))
        v_hat = v / (1 - beta2 ** (epoch + 1))
        weights_adam += learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

# Evaluate Adam
train_accuracy_adam = evaluate(weights_adam, X_train, y_train)
val_accuracy_adam = evaluate(weights_adam, X_val, y_val)
test_accuracy_adam = evaluate(weights_adam, X_test, y_test)

print("\nAdam:")
print(f"Training Accuracy: {train_accuracy_adam:.0%}")
print(f"Validation Accuracy: {val_accuracy_adam:.0%}")
print(f"Testing Accuracy: {test_accuracy_adam:.0%}")
