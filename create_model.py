#!/usr/bin/env python3
"""
Create a dummy classifier.pt file for testing the application.
This model is UNTRAINED and meant only for pipeline validation.
"""

import os
import argparse
import torch
from torch import nn
from torchvision import models
from datetime import datetime

# ---------------------------
# Argument Parser
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Create dummy classifier model")
    parser.add_argument(
        "--num_classes", type=int, default=5,
        help="Number of output classes (default: 5)"
    )
    parser.add_argument(
        "--output", type=str, default="classifier.pt",
        help="Output model file name"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()

# ---------------------------
# Main Function
# ---------------------------
def main():
    args = parse_args()

    print("🚀 Creating dummy classifier model...")
    print(f"📌 Number of classes: {args.num_classes}")

    # Reproducibility
    torch.manual_seed(args.seed)

    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # ---------------------------
    # Model Architecture
    # ---------------------------
    model = models.resnet152(weights=None)

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, args.num_classes),
        nn.LogSoftmax(dim=1)
    )

    model.to(device)
    model.eval()

    # ---------------------------
    # Metadata (VERY useful in apps)
    # ---------------------------
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": "resnet152",
        "num_classes": args.num_classes,
        "framework": "pytorch",
        "created_at": datetime.utcnow().isoformat(),
        "trained": False,
        "note": "Dummy untrained model for testing only"
    }

    # ---------------------------
    # Save Model
    # ---------------------------
    output_path = os.path.abspath(args.output)
    torch.save(checkpoint, output_path)

    print(f"✅ Dummy model saved at: {output_path}")
    print("⚠️ WARNING:")
    print("   This model is NOT trained.")
    print("   Use it ONLY for testing inference pipelines, loading logic, and UI.")

# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    main()
