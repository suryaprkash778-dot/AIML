import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# 1. Hardware Device (Utilizing your RTX 3050)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# 2. Data Loading and Preprocessing
# Mammograms vary in size, so we resize them to a standard 128x128 pixels.
# ImageFolder loads images in RGB format (3 channels) by default.
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load from the local folder
dataset_path = "./Breast Cancer Dataset"
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Print class mapping (e.g., {'Benign': 0, 'Malignant': 1})
class_names = dataset.classes
print(f"Class mapping: {dataset.class_to_idx}")

# Split into 80% Training, 20% Validation/Testing
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, pin_memory=True)


# 3. Define the 4-Layer CNN
class BreastCancerCNN(nn.Module):
    def __init__(self):
        super(BreastCancerCNN, self).__init__()
        # Layer 1: Convolutional
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2: Convolutional
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Layer 3: Fully Connected
        # 128x128 image pooled twice becomes 32x32.
        self.fc1 = nn.Linear(64 * 32 * 32, 128)

        # Layer 4: Output Layer (1 node for Binary Classification)
        self.fc2 = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        # Explicit Sigmoid activation for the output layer (outputs a probability between 0 and 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


model = BreastCancerCNN().to(device)

# 4. Optimization and Loss Function
# Binary Cross-Entropy Loss requires the output and target tensors to be the exact same shape
criterion = nn.BCELoss()
# Using Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop with Validation Tracking
epochs = 10
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        # Reshape labels from [batch_size] to [batch_size, 1] and convert to float for BCE
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation Phase
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# 6. Plotting Training and Validation Loss
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Training Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='o')
plt.title('Binary Cross-Entropy Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 7. Evaluation Metrics: Confusion Matrix and Classification Report
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)

        # Convert Sigmoid probabilities to binary predictions (threshold = 0.5)
        predicted = (outputs >= 0.5).float().cpu().numpy()

        all_preds.extend(predicted)
        all_targets.extend(labels.numpy())

# Clean up arrays for sklearn
all_preds = np.array(all_preds).flatten()
all_targets = np.array(all_targets).flatten()

print("\nClassification Report:")
print(classification_report(all_targets, all_preds, target_names=class_names))

cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix on Validation Data')
plt.xlabel('Predicted Diagnosis')
plt.ylabel('Actual Diagnosis')
plt.show()
