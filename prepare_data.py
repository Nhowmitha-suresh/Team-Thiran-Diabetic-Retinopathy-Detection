import os
import shutil
import argparse
import pandas as pd
from tqdm import tqdm
from collections import Counter

# -------------------------------
# Argument Parser
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Organize DR dataset into class folders")
    parser.add_argument("--base_dir", type=str, default="dataset",
                        help="Base dataset directory")
    parser.add_argument("--csv", type=str, default="train.csv",
                        help="CSV file name")
    parser.add_argument("--img_dir", type=str, default="train_images",
                        help="Directory with original images")
    parser.add_argument("--output_dir", type=str, default="train",
                        help="Output directory")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of moving them")
    parser.add_argument("--dry_run", action="store_true",
                        help="Simulate the process without moving/copying files")
    return parser.parse_args()

# -------------------------------
# Main Function
# -------------------------------
def main():
    args = parse_args()

    base_dir = args.base_dir
    csv_path = os.path.join(base_dir, args.csv)
    img_dir = os.path.join(base_dir, args.img_dir)
    output_dir = os.path.join(base_dir, args.output_dir)

    classes = {
        0: "0_No_DR",
        1: "1_Mild",
        2: "2_Moderate",
        3: "3_Severe",
        4: "4_Proliferative_DR"
    }

    print("🚀 Starting dataset organization")
    print(f"📂 CSV: {csv_path}")
    print(f"🖼️ Images: {img_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"🛠️ Mode: {'COPY' if args.copy else 'MOVE'}")
    if args.dry_run:
        print("🧪 DRY RUN ENABLED (no files will be changed)")

    # -------------------------------
    # Create output folders
    # -------------------------------
    for folder in classes.values():
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    # -------------------------------
    # Read CSV
    # -------------------------------
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} labeled entries")

    moved = 0
    missing = 0
    invalid_labels = 0
    class_counter = Counter()

    # -------------------------------
    # Process Images
    # -------------------------------
    for img_id, label in tqdm(
        zip(df["id_code"], df["diagnosis"]),
        total=len(df),
        desc="Organizing images"
    ):
        if label not in classes:
            invalid_labels += 1
            continue

        src = os.path.join(img_dir, f"{img_id}.png")
        dst = os.path.join(output_dir, classes[label], f"{img_id}.png")

        if not os.path.exists(src):
            missing += 1
            continue

        if not args.dry_run:
            if args.copy:
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)

        moved += 1
        class_counter[classes[label]] += 1

    # -------------------------------
    # Summary
    # -------------------------------
    print("\n📊 ORGANIZATION SUMMARY")
    print("-" * 40)
    print(f"✔ Images processed : {moved}")
    print(f"❌ Missing images  : {missing}")
    print(f"⚠ Invalid labels  : {invalid_labels}")

    print("\n📁 Images per class:")
    for cls, count in class_counter.items():
        print(f"  {cls}: {count}")

    if args.dry_run:
        print("\n🧪 Dry run completed. No files were modified.")
    else:
        print("\n🎉 Dataset organization completed successfully!")

# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    main()
