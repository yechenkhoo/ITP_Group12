import os
import shutil
import csv
import re

# === CONFIG ===
root_dir = "./out"            # initial folder containing P1, P2, ... (to be deleted)
output_dir = "./new_out"  # new grouped folder location (to be renamed to ./out)
csv_path = "_map_to_labels.csv"

os.makedirs(output_dir, exist_ok=True)

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "golfer_name", "label"])

    for folder in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            # Extract golfer name before "_frame"
            match = re.match(r"([a-zA-Z0-9_]+)_frame\d+", file)
            if not match:
                print(f"Skipping unrecognized file: {file}")
                continue
            golfer_name = match.group(1)

            # Make golfer folder
            golfer_dir = os.path.join(output_dir, golfer_name)
            os.makedirs(golfer_dir, exist_ok=True)

            # Copy file to new structure
            src_path = os.path.join(folder_path, file)
            dst_path = os.path.join(golfer_dir, file)
            shutil.copy2(src_path, dst_path)

            # Write row to CSV (only filename, golfer_name, original folder)
            writer.writerow([file, golfer_name, folder])

print("Done! Grouped by golfer and CSV mapping created.")
