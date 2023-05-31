import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, jaccard_score
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the transforms for data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the dataset and apply transforms
dataset = torchvision.datasets.ImageFolder(root=r'E:\ECE\6th_sem\mini project\dataset', transform=transform)

# Split the dataset into train, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Load the pre-trained ResNet-18 model
model = torchvision.models.resnet18(pretrained=True)

# Replace the last fully connected layer with a new one
num_classes = 2  # Assuming binary classification
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
    # Compute validation accuracy
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    # Print epoch-wise results
    print(f'Epoch [{epoch+1}/{num_epochs}] - '
          f'Train Loss: {train_loss/train_total:.4f}, Train Acc: {100*train_correct/train_total:.2f}%, '
          f'Val Loss: {val_loss/val_total:.4f}, Val Acc: {100*val_correct/val_total:.2f}%')

# Evaluate the model on the test set
model.eval()
test_correct = 0
test_total = 0
test_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        test_predictions.extend(predicted.tolist())
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

# Convert test predictions to numpy array
test_predictions = np.array(test_predictions)

# Get the ground truth labels for the test set
test_labels = np.array([])
for _, labels in test_loader:
    test_labels = np.concatenate((test_labels, labels.cpu().numpy()))

# Compute test metrics
test_cm = confusion_matrix(test_labels, test_predictions)
print("Confusion Matrix:\n", test_cm)
print("F1 Score:", f1_score(test_labels, test_predictions))
print("Accuracy:", accuracy_score(test_labels, test_predictions))
print("Sensitivity (Recall):", recall_score(test_labels, test_predictions))
print("Jaccard Index:", jaccard_score(test_labels, test_predictions))
