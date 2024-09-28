import os

def create_raw_train_file(output_dir, output_file, valid_classes):
    """Collects all filenames (without .txt extension) from specified class folders and saves them in raw_train.txt."""
    with open(output_file, 'w') as f:
        for class_name in valid_classes:
            class_dir = os.path.join(output_dir, class_name)
            if os.path.exists(class_dir):
                for file in os.listdir(class_dir):
                    if file.endswith(".txt"):
                        # Remove the '.txt' extension and write the filename (without the folder) to raw_train.txt
                        file_name = os.path.splitext(file)[0]
                        f.write(file_name + '\n')
    print(f"Filenames from {', '.join(valid_classes)} saved to {output_file}")

if __name__ == '__main__':
    # Define paths and valid classes
    output_dir = "./cropped_points"  # Directory containing the class folders
    output_file = "raw_train.txt"    # Output file to store the list of filenames
    valid_classes = ['Car', 'Pedestrian', 'Cyclist']  # Only include these classes
    
    # Create raw_train.txt with filenames from specified classes
    create_raw_train_file(output_dir, output_file, valid_classes)

