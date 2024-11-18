import os
import csv

# Paths to your folders
threat_folder = "./threat_dataset/threatv2/threat"
non_threat_folder = "./threat_dataset/threatv2/non-threat"
output_csv = "./threat_dataset/threatv2/dataset.csv"

# Collect data
data = []

# Function to process folders
def process_folder(base_folder, label):
    for root, _, files in os.walk(base_folder):
        for filename in files:
            if filename.endswith(".jpg"):  # Adjust the file extension if needed
                # Use os.path.join to correctly handle paths
                filepath = os.path.join(root, filename).replace("\\", "/")
                data.append([filepath, label])

# Process threat and non-threat folders
process_folder(threat_folder, 1)  # Label 1 for 'threat'
process_folder(non_threat_folder, 0)  # Label 0 for 'non-threat'

# Write to CSV
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image_path", "label"])  # Header
    writer.writerows(data)

print(f"CSV file saved to {output_csv}")
