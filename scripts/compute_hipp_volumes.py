import trimesh
import argparse

# arguments

import os
import trimesh
import pandas as pd


def valid_directory(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid directory.")
    return path
# Directory containing the 3D model files
parser = argparse.ArgumentParser(description="Compute volumes of 3D models")
parser.add_argument("datapath", type=valid_directory, help="Directory containing the 3D model files")

args = parser.parse_args()
datapath = args.datapath
models_dir = os.path.join(args.datapath, "models")
assert os.path.exists(models_dir), "Directory does not exist"
assert len([ f for f in os.listdir(models_dir) if f.endswith('.obj') ]) == 0, "Models must contain .obj files"

# List to store the results
data = []

# Iterate through all files in the directory
for file_name in os.listdir(models_dir):
    if file_name.endswith(".obj"):  # Process only .obj files
        # Extract the ID from the file name
        file_id = file_name.split("_", maxsplit=1)[0]
        parts = file_id.split("-")
        site = parts[0]
        subject_id = parts[1]
        session = parts[3]
        side = file_name.split("_")[-1].replace(".obj", "")

        # Load the 3D model
        file_path = os.path.join(models_dir, file_name)
        mesh = trimesh.load(file_path)

        # Compute the volume
        volume = mesh.volume

        # Append the result to the list
        data.append({
            "id": file_id,
            "volume": volume,
            "site": site,
            "subject_id": subject_id,
            "session": session,
            "hemisphere": side
        })

# Create a DataFrame from the results
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv(os.path.join(datapath, "volumes.csv"), index=False)

print("Volumes computed and saved to volumes.csv")