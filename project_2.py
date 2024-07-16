
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import os

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

training_path = "/home/pavaniankam/Desktop/paathu/lion_train_data/"
validation_path = "/home/pavaniankam/Desktop/paathu/lion_validation_data/"
testing_path = "/home/pavaniankam/Desktop/paathu/lion_test_data/"

root = training_path
transform = transform
samples = []
for root, _, fnames in sorted(os.walk(root)):
    for fname in sorted(fnames):
        path = os.path.join(root, fname)
        try:
            img = Image.open(path).convert('L')
            samples.append((path, img))
        except Exception as e:
            print(f"Skipping {path}: {str(e)}")

train_loader = DataLoader([(transform(sample[1]), 0) for sample in samples], batch_size=32, shuffle=True)

root = validation_path
transform = transform
samples = []
for root, _, fnames in sorted(os.walk(root)):
    for fname in sorted(fnames):
        path = os.path.join(root, fname)
        try:
            img = Image.open(path).convert('L')
            samples.append((path, img))
        except Exception as e:
            print(f"Skipping {path}: {str(e)}")

val_loader = DataLoader([(transform(sample[1]), 0) for sample in samples], batch_size=32, shuffle=False)

root = testing_path
transform = transform
samples = []
for root, _, fnames in sorted(os.walk(root)):
    for fname in sorted(fnames):
        path = os.path.join(root, fname)
        try:
            img = Image.open(path).convert('L')
            samples.append((path, img))
        except Exception as e:
            print(f"Skipping {path}: {str(e)}")

test_loader = DataLoader([(transform(sample[1]), 0) for sample in samples], batch_size=32, shuffle=False)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(4):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)

    # Calculate training accuracy
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in train_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    train_acc = correct / total

    # Calculate validation accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = correct / total

    print(f"Epoch [{epoch+1}/10], Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# Evaluate the model on the test set
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
test_acc = correct / total
print(f"Testing Accuracy: {test_acc:.4f}")
