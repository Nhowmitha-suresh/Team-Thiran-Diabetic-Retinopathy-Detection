#!/usr/bin/env python3
"""
Create a dummy classifier.pt file for testing the application
"""
import torch
from torch import nn
from torchvision import models
import os

print("Creating dummy classifier model...")

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model Architecture (same as in create_dummy_classifier.py)
model = models.resnet152(weights=None)
num_ftrs = model.fc.in_features
out_ftrs = 5  # 5 DR severity levels

# Replace the fully connected layer
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Linear(512, out_ftrs),
    nn.LogSoftmax(dim=1)
)

model.to(device)

# Save the model with just initialized weights (for testing)
checkpoint = {
    'model_state_dict': model.state_dict()
}

output_path = os.path.join(os.getcwd(), "classifier.pt")
torch.save(checkpoint, output_path)
print(f"✅ Dummy model saved to: {output_path}")
print("⚠️ Note: This is an untrained model for testing purposes only.")
print("   For real predictions, you need to train the model on actual data.")
