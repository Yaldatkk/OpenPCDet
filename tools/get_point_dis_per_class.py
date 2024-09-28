import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def get_points_distribution(cropped_points_dir):
    """Get the distribution of number of points for each class."""
    points_distribution = {}  # Dictionary to store list of points count per file for each class
    
    # Iterate through all the saved cropped points in .txt files
    for class_dir in Path(cropped_points_dir).iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name  # Get the class from the folder name
            points_distribution[class_name] = []  # Initialize list for storing points count
            
            for txt_file in class_dir.glob("*.txt"):
                points = np.loadtxt(txt_file)  # Load points from each .txt file
                num_points = points.shape[0]  # Number of points in the file
                points_distribution[class_name].append(num_points)  # Append to class list
    
    return points_distribution

def plot_histogram(points_distribution):
    """Plot histogram of points count per class."""
    for class_name, points_list in points_distribution.items():
        plt.figure()  # Create a new figure for each class
        plt.hist(points_list, bins=10, edgecolor='black')  # Plot histogram with 10 bins
        plt.title(f'Histogram of Points Count for {class_name}')
        plt.xlabel('Number of Points')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()  # Display the plot

if __name__ == '__main__':
    # Define path
    cropped_points_dir = "./cropped_points"  # Input directory

    # Get the distribution of number of points for each class
    points_distribution = get_points_distribution(cropped_points_dir)

    # Plot the histogram of points count for each class
    plot_histogram(points_distribution)

