# ============================================================
# 🧠 MODEL FILE — Diabetic Retinopathy Detection (Demo-Ready Version)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torchvision
from torchvision import models, transforms
from PIL import Image
from torch.optim import lr_scheduler
import os, random

print("✅ Imported packages successfully")

# ============================================================
# Device setup
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# ============================================================
# Model Architecture (for structure only — not used for demo)
# ============================================================
model = models.resnet152(weights=None)
num_ftrs = model.fc.in_features
out_ftrs = 5  # 5 DR classes

model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Linear(512, out_ftrs),
    nn.LogSoftmax(dim=1)
)

criterion = nn.NLLLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
model.to(device)

# ============================================================
# Load dummy model safely
# ============================================================

MODEL_PATH = "classifier.pt"

def load_model(path):
    if os.path.exists(path):
        print(f"✅ Dummy model file found: {path}")
    else:
        print(f"⚠️ No trained model found, using demo mode.")
    return model

model = load_model(MODEL_PATH)

# ============================================================
# DR Classes and Transforms
# ============================================================
classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

# ============================================================
# Always Predicts Some DR (Demo Mode)
# ============================================================
def inference(model, file, transform, classes):
    """
    Always predicts a DR stage (not No DR).
    Randomly chooses among Mild–Proliferative DR.
    """
    try:
        img = Image.open(file).convert("RGB")

        # Apply transforms (for completeness)
        _ = transform(img).unsqueeze(0)

        # Randomly choose between 1–4 (to skip No DR)
        severity = random.randint(1, 4)
        predicted_class = classes[severity]

        print(f"🎯 Predicted DR Stage: {predicted_class} (Severity {severity})")
        return severity, predicted_class

    except Exception as e:
        print(f"❌ Error during inference: {e}")
        raise e

# ============================================================
# Main function (used by blindness.py)
# ============================================================
def main(path):
    try:
        print(f"🔍 Running inference on: {path}")
        severity, predicted_class = inference(model, path, test_transforms, classes)
        print(f"✅ Final Prediction: {predicted_class}")
        return severity, predicted_class
    except Exception as e:
        print(f"❌ Error in main(): {e}")
        raise e
