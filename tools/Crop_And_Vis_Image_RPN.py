"""
python Crop_And_Vis_Image_RPN.py --image_dir ../data/kitti/training/image_2/ --output_dir ../../Hungarian_Matching/Cropped_Images/2D_RPN/ --data_path ../../Hungarian_Matching/Hungarian_Results/

"""


import os
import cv2
import numpy as np
import json
import clip
import torch
import argparse
from pathlib import Path
from PIL import Image

def extract_boxes_from_json(data):
    """Extract bounding boxes and YOLO labels from JSON data."""
    boxes = []
    try:
        # Extract the matches from the JSON data
        matches = data.get('matches', [])
        
        for match in matches:
            rpn_box = match.get('rpn_2d')
            projected_box = match.get('projected_2d')
            yolo_label = match.get('yolo_label')
            match_id = match.get('id')  # Extract the id
            
            if rpn_box and len(rpn_box) == 4 and yolo_label is not None:
                boxes.append({
                    'rpn': rpn_box,
                    'projected': projected_box,
                    'yolo_label': yolo_label,
                    'id': match_id  # Include the id
                })
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
    return cropped_img

def main():
    parser = argparse.ArgumentParser(description='Extract and process images based on bounding boxes.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save cropped images.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the folder containing results files.')
    args = parser.parse_args()

    # Create output directories
    rpn_output_dir = Path(args.output_dir) / '2D_RPN'
    projected_output_dir = Path(args.output_dir) / '2D_Projected'
    rpn_output_dir.mkdir(parents=True, exist_ok=True)
    projected_output_dir.mkdir(parents=True, exist_ok=True)

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Iterate through all result files in the folder
    for result_file in Path(args.data_path).glob('*_results.json'):

        with open(result_file, 'r') as file:
            results = json.load(file)
        
        # Extract the numeric part of the file name (before "_results.json")
        image_file_name = result_file.stem.split('_')[0]
        image_path = Path(args.image_dir) / f'{image_file_name}.png'

        if not image_path.exists():
            print(f"Image file {image_path} does not exist.")
            continue

        for entry in extract_boxes_from_json(results):
            rpn_box = entry.get('rpn')
            projected_box = entry.get('projected')
            yolo_label = entry.get('yolo_label')
            match_id = entry.get('id')  # Get the match id for file naming

            # Process RPN Box
            if rpn_box:
                print(f"Processing RPN Box: {rpn_box}")
                cropped_rpn_img = crop_image(image_path, rpn_box)
                if cropped_rpn_img is not None:
                    rpn_img_pil = Image.fromarray(cropped_rpn_img)
                    rpn_img_tensor = preprocess(rpn_img_pil).unsqueeze(0).to(device)
                    rpn_features = model.encode_image(rpn_img_tensor).cpu().detach().numpy()
                    print(f"RPN Box features shape: {rpn_features.shape}")
                else:
                    rpn_features = None

            # Process Projected Box
            if projected_box:
                print(f"Processing Projected Box: {projected_box}")
                cropped_projected_img = crop_image(image_path, projected_box)
                if cropped_projected_img is not None:
                    projected_img_pil = Image.fromarray(cropped_projected_img)
                    projected_img_tensor = preprocess(projected_img_pil).unsqueeze(0).to(device)
                    projected_features = model.encode_image(projected_img_tensor).cpu().detach().numpy()
                    print(f"Projected Box features shape: {projected_features.shape}")
                else:
                    projected_features = None

            # Save both RPN and Projected features if available
            if rpn_features is not None:
                # Use the match id in the file name
                rpn_file_name = f'{image_file_name}_{yolo_label}_{match_id}_rpn.npy'
                np.save(rpn_output_dir / rpn_file_name, rpn_features)
                print(f"Saved RPN features for {rpn_file_name}")

            if projected_features is not None:
                # Use the match id in the file name
                projected_file_name = f'{image_file_name}_{yolo_label}_{match_id}_projected.npy'
                np.save(projected_output_dir / projected_file_name, projected_features)
                print(f"Saved Projected features for {projected_file_name}")

if __name__ == '__main__':
    main()

