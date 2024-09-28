"""
python save_paired_yolo_projected.py --image_dir ../data/kitti/testing/image_2/ --output_dir ~/3D_Obj_Det/Hungarian_Matching/Cropped_Images/ --data_path /home/saeed/3D_Obj_Det/Hungarian_Matching/Hungarian_Results/

"""

import os
import cv2
import numpy as np
import json
import argparse
from pathlib import Path

def extract_boxes_from_match(match):
    """Extract bounding boxes and YOLO label from the 'match' entry in the JSON."""
    boxes = {}
    try:
        rpn_box = match.get('rpn_box')
        projected_box = match.get('projected_box')
        yolo_label = match.get('yolo_label')  # Extract YOLO label

        if rpn_box and len(rpn_box) == 4:
            boxes['rpn'] = rpn_box
        if projected_box and len(projected_box) == 4:
            boxes['projected'] = projected_box
        boxes['yolo_label'] = yolo_label if yolo_label is not None else 'unknown'  # Default label if missing
    except ValueError as e:
        print(f"Error parsing boxes: {e}")
        return None
    return boxes

def crop_image(image_path, box):
    """Crop the image based on the bounding box."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error loading image {image_path}")
        return None

    # Ensure that the box has exactly 4 values
    if len(box) != 4:
        print(f"Invalid box format: {box}")
        return None

    x_min, y_min, x_max, y_max = map(int, box)
    # Ensure that the cropping coordinates are within image bounds
    x_min, y_min = max(x_min, 0), max(y_min, 0)
    x_max, y_max = min(x_max, img.shape[1]), min(y_max, img.shape[0])
    cropped_img = img[y_min:y_max, x_min:x_max]
    
    # Log dimensions of the cropped image
    if cropped_img is not None:
        print(f"Cropped image dimensions: {cropped_img.shape}")
    return cropped_img

def main():
    parser = argparse.ArgumentParser(description='Extract and process images based on bounding boxes.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save cropped images.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to Hungarian results files.')
    args = parser.parse_args()

    # Check if output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    for result_file in Path(args.data_path).glob('*_results.json'):
        print(f"Processing result file: {result_file}")
        
        # Extract image file name from the result file name (removing '_results.json')
        image_file_name = result_file.stem.replace('_results', '')
        image_path = Path(args.image_dir) / f'{image_file_name}.png'

        if not image_path.exists():
            print(f"Image file {image_path} does not exist.")
            continue

        with open(result_file, 'r') as file:
            data = json.load(file)

        # Extract frame_id (image file name should correspond to the frame_id)
        frame_id = data.get("frame_id")
        if frame_id != image_file_name:
            print(f"Frame ID {frame_id} does not match image file name {image_file_name}")
            continue

        # Loop through the matches
        for i, match in enumerate(data.get('matches', [])):
            print(f"Processing match {i} from result file")
            boxes = extract_boxes_from_match(match)
            if boxes is None:
                print(f"No boxes found in match {i}.")
                continue
            
            # Crop images for both RPN Box and Projected Box
            rpn_box = boxes.get('rpn')
            projected_box = boxes.get('projected')
            yolo_label = boxes.get('yolo_label')  # Get YOLO label

            # Process RPN Box
            if rpn_box:
                print(f"Processing RPN Box: {rpn_box}")
                cropped_rpn_img = crop_image(image_path, rpn_box)
                if cropped_rpn_img is None:
                    print(f"Failed to crop RPN image for {image_file_name}_{i}")
                    continue

            # Process Projected Box
            if projected_box:
                print(f"Processing Projected Box: {projected_box}")
                cropped_projected_img = crop_image(image_path, projected_box)
                if cropped_projected_img is None:
                    print(f"Failed to crop Projected image for {image_file_name}_{i}")
                    continue

            # Save the cropped images to .npz format
            output_file = Path(args.output_dir) / f'{image_file_name}_label_{yolo_label}_{i}.npz'
            np.savez_compressed(output_file, rpn_box=cropped_rpn_img, projected_box=cropped_projected_img)
            print(f"Saved {output_file}")

if __name__ == '__main__':
    main()

