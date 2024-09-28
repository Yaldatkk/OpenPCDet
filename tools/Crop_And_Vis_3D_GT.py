import os
import numpy as np
from pathlib import Path

# Define valid classes and point thresholds
VALID_CLASSES = {
    'Car': 25,        # Minimum 25 points for Car
    'Pedestrian': 10, # Minimum 10 points for Pedestrian
    'Cyclist': 10     # Minimum 10 points for Cyclist
}

def convert_bin_to_txt(bin_file, output_dir, train_file):
    """Reads a .bin file, converts it to a .txt file, and saves it if it meets the minimum point threshold per class. Also adds valid filenames to raw_train.txt."""
    # Read point cloud data from the .bin file
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    
    # Extract filename and class from bin_file
    bin_filename = os.path.basename(bin_file)
    file_parts = bin_filename.split('_')  # E.g., ['000000', 'Pedestrian', '0.bin']
    
    if len(file_parts) != 3:
        print(f"Warning: Invalid file name format for {bin_filename}")
        return
    
    # Extract relevant information
    sample_id = file_parts[0]
    class_name = file_parts[1]
    instance_id = file_parts[2].split('.')[0]  # Remove the .bin extension
    
    # Check if the class is valid
    if class_name not in VALID_CLASSES:
        print(f"Skipping {bin_file} as it is not in the valid classes.")
        return
    
    # Check if the point cloud meets the minimum points required for the class
    min_points_required = VALID_CLASSES[class_name]
    if points.shape[0] < min_points_required:
        print(f"Skipping {bin_file} as it contains less than {min_points_required} points for class {class_name}.")
        return
    
    # If points are less than 1024, duplicate them uniformly
    #if points.shape[0] < 1024:
    #    num_to_add = 1024 - points.shape[0]
    #    # Generate indices to duplicate points
    #    indices = np.random.choice(points.shape[0], num_to_add, replace=True)
    #    points = np.vstack((points, points[indices]))
    
    # Prepare output directory for the class (e.g., ./cropped_points/Pedestrian)
    class_output_dir = Path(output_dir) / class_name
    class_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct output filename (e.g., 000000_0.txt)
    output_filename = f"{sample_id}_{class_name}_{instance_id}.txt"  # Removed class name from output filename
    output_file_path = class_output_dir / output_filename
    
    # Save the points in .txt format
    np.savetxt(output_file_path, points, fmt="%.6f")
    print(f"Saved {output_file_path}")
    
    # Add the filename (without extension) to raw_train.txt
    with open(train_file, 'a') as f:
        f.write(f"{sample_id}_{class_name}_{instance_id}\n")

def process_gt_database(gt_database_dir, output_dir, train_file):
    """Process all .bin files in the gt_database folder and save valid filenames to raw_train.txt."""
    # Remove raw_train.txt if it exists to start fresh
    if os.path.exists(train_file):
        os.remove(train_file)
    
    # Iterate over all .bin files in the gt_database_dir
    for bin_file in Path(gt_database_dir).rglob("*.bin"):
        convert_bin_to_txt(bin_file, output_dir, train_file)
    
    print("All files processed.")

if __name__ == '__main__':
    # Define paths
    gt_database_dir = "/home/saeed/3D_Obj_Det/OpenPCDet/data/kitti/gt_database"  # Input directory
    output_dir = "./cropped_points"  # Output directory
    train_file = "./raw_train.txt"   # File to save valid filenames

    # Process the ground truth database and save filenames to raw_train.txt
    process_gt_database(gt_database_dir, output_dir, train_file)

