import os
import csv

# Paths to your folders
threat_folder = "./threat_dataset/threat"
non_threat_folder = "./threat_dataset/non-threat"
output_csv = "./threat_dataset/dataset.csv"

# Collect data
data = []

# Process threat folder
for filename in os.listdir(threat_folder):
    if filename.endswith(".jpg"):  # Adjust the file extension if needed
        # Use os.path.join to correctly handle paths
        data.append([os.path.join(threat_folder, filename).replace("\\", "/"), 1])  # Label 1 for 'threat'

# Process non-threat folder
for filename in os.listdir(non_threat_folder):
    if filename.endswith(".jpg"):
        # Use os.path.join to correctly handle paths
        data.append([os.path.join(non_threat_folder, filename).replace("\\", "/"), 0])  # Label 0 for 'non-threat'

# Write to CSV
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image_path", "label"])  # Header
    writer.writerows(data)

print(f"CSV file saved to {output_csv}")
