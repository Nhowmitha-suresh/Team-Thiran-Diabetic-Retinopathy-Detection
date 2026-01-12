import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

# ===============================================================
# ✅ SETTINGS
# ===============================================================
data_dir = "dataset/train"
num_classes = 5
num_epochs = 2               # quick training for CPU
batch_size = 16
learning_rate = 0.0001

# ===============================================================
# ✅ DEVICE CONFIGURATION
# ===============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# ===============================================================
# ✅ DATA TRANSFORMS
# ===============================================================
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===============================================================
# ✅ LOAD DATASET
# ===============================================================
dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

print(f"📊 Total images: {total_size} | Train: {train_size} | Validation: {val_size}")

# ===============================================================
# ✅ MODEL: RESNET18 (FAST + ACCURATE)
# ===============================================================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes),
    nn.LogSoftmax(dim=1)
)
model = model.to(device)

# ===============================================================
# ✅ LOSS & OPTIMIZER
# ===============================================================
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ===============================================================
# ✅ TRAINING LOOP
# ===============================================================
best_val_acc = 0.0
os.makedirs("models", exist_ok=True)

print("\n🚀 Starting training...\n")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print every few batches
        if batch_idx % 10 == 0:
            print(f"🧩 Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_train_loss = running_loss / len(train_loader)

    # ===============================================================
    # ✅ VALIDATION PHASE
    # ===============================================================
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total

    print(f"\n✅ Epoch [{epoch+1}/{num_epochs}] "
          f"| Train Loss: {avg_train_loss:.4f} "
          f"| Val Loss: {avg_val_loss:.4f} "
          f"| Val Acc: {val_acc:.2f}%\n")

    # ===============================================================
    # ✅ SAVE BEST MODEL
    # ===============================================================
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, "classifier.pt")
        print(f"💾 New best model saved! (Val Acc: {best_val_acc:.2f}%)")

print("\n🎯 Model training complete!")
print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
print("✅ Final model saved as 'classifier.pt'")
