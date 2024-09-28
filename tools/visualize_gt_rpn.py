import os
import numpy as np
from pathlib import Path
from visual_utils import open3d_vis_utils as V  # Assuming Open3D is used for visualization

def visualize_cropped_points(cropped_points_dir):
    """Visualize cropped points from the directory and display class and number of points."""
    # Iterate through all the saved cropped points in .txt files
    for class_dir in Path(cropped_points_dir).iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name  # Get the class from the folder name
            
            # Only visualize if the class is "Cyclist"
            if class_name.lower() == "cyclist":  # Check if the class is Cyclist (case-insensitive)
                for txt_file in class_dir.glob("*.txt"):
                    points = np.loadtxt(txt_file)  # Load points from each .txt file
                    
                    num_points = points.shape[0]  # Number of points in the file
                    
                    # Display class and number of points
                    print(f"Class: {class_name}, Number of points: {num_points} from {txt_file}")
                    
                    # Visualize the cropped points (x, y, z)
                    V.draw_scenes(points=points[:, :3])

if __name__ == '__main__':
    # Define path
    cropped_points_dir = "./cropped_points"  # Input directory

    # Visualize the cropped points
    visualize_cropped_points(cropped_points_dir)

