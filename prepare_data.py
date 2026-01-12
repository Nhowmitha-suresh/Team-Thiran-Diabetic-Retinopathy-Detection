import os
import shutil
import pandas as pd
from tqdm import tqdm

# ===============================
# PATH CONFIGURATION
# ===============================
base_dir = "dataset"
csv_path = os.path.join(base_dir, "train.csv")       # CSV file path
img_dir = os.path.join(base_dir, "train_images")     # Folder with original images
output_dir = os.path.join(base_dir, "train")         # Destination for organized images

# ===============================
# CREATE OUTPUT FOLDERS
# ===============================
classes = ['0_No_DR', '1_Mild', '2_Moderate', '3_Severe', '4_Proliferative_DR']

# Create main class folders if not already there
for c in classes:
    os.makedirs(os.path.join(output_dir, c), exist_ok=True)

# ===============================
# READ CSV
# ===============================
df = pd.read_csv(csv_path)
print(f"✅ Found {len(df)} labeled images in CSV.")

# ===============================
# MOVE IMAGES TO CLASS FOLDERS
# ===============================
moved_count = 0
missing_count = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_id = str(row['id_code'])
    label = int(row['diagnosis'])
    label_folder = classes[label]

    # Build source and destination paths
    src = os.path.join(img_dir, img_id + ".png")     # assumes all are .png
    dst = os.path.join(output_dir, label_folder, img_id + ".png")

    # Move if file exists
    if os.path.exists(src):
        shutil.move(src, dst)
        moved_count += 1
    else:
        missing_count += 1

# ===============================
# SUMMARY
# ===============================
print(f"\n✅ Dataset organized successfully!")
print(f"📦 {moved_count} images moved into class folders.")
if missing_count > 0:
    print(f"⚠️  {missing_count} images not found in {img_dir}")
else:
    print("🎉 All images found and organized perfectly!")
