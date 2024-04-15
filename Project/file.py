import os
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import Adam
import torchvision as tv
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import timm
import tensorflow as tf

# Define the dataset class
class ChestXRayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label
    
# Load your data
data = pd.read_csv('/Users/ananyajain/Desktop/CSC413/CSC413-Final-Project/archive/sample_labels.csv')
data['labels'] = data['Finding Labels'].map(lambda x: x.split('|'))
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(data['labels'])
print("Number of classes:", len(mlb.classes_))
# if len(mlb.classes_) != 15:
#     raise ValueError("The number of target classes does not match num_classes in the model")
labels = np.array(labels, dtype=float)

image_dir = '/Users/ananyajain/Desktop/CSC413/CSC413-Final-Project/archive/sample/images'
image_paths = [os.path.join(image_dir, x) for x in data['Image Index']]

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Split the data
train_paths, val_test_paths, train_labels, val_test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    val_test_paths, val_test_labels, test_size=0.5, random_state=42)

# Create datasets
train_dataset = ChestXRayDataset(train_paths, train_labels, transform)
val_dataset = ChestXRayDataset(val_paths, val_labels, transform)
test_dataset = ChestXRayDataset(test_paths, test_labels, transform)

# Create DataLoaders
batch_size = 16
loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Setting the device, either GPU cluster or cpu.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

model = timm.create_model('vit_base_patch16_224', pretrained=True)
num_classes = 15
model.head = nn.Linear(model.head.in_features, num_classes)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

def train_model(model, criterion, optimizer, loader_train, loader_val, num_epochs=10):
    model.train()  # Set model to training mode
    print("starting to train")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in loader_train:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(loader_train.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Validation phase
        validate_model(model, loader_val)

def validate_model(model, loader_val, threshold=0.5):
    model.eval()
    total_samples = 0
    total_correct = 0
    
    with torch.no_grad():
        for inputs, labels in loader_val:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = outputs.sigmoid() > threshold  # Apply sigmoid and threshold to convert logits to binary predictions
            total_correct += (predicted == labels.byte()).sum().item()  # Correct predictions per label
            total_samples += labels.numel()  # Total number of label predictions

    accuracy = total_correct / total_samples * 100
    print(f'Validation Accuracy: {accuracy:.2f}%')


# Assuming you have your dataloaders ready
train_model(model, criterion, optimizer, loader_train, loader_val)