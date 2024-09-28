"""
python Crop_And_Vis_3D_RPN.py  --cfg_file cfgs/kitti_models/pointrcnn.yaml --data_path ../data/kitti/training/velodyne/

"""


import json
import numpy as np
from pathlib import Path

def crop_points_inside_boxes(points, boxes):
    cropped_points_list = []
    for box in boxes:
        x_center, y_center, z_center, dx, dy, dz, heading = box
        x_min = x_center - dx / 2
        x_max = x_center + dx / 2
        y_min = y_center - dy / 2
        y_max = y_center + dy / 2
        z_min = z_center - dz / 2
        z_max = z_center + dz / 2

        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        cropped_points = points[mask]
        if cropped_points.size > 0:  # Only append if there are points
            cropped_points_list.append(cropped_points)
    return cropped_points_list

def save_cropped_points_to_file(cropped_points_list, output_dir, file_name, match_id):
    # Combine all cropped points into one file if multiple sets of cropped points are present
    output_file = output_dir / f'{file_name}_{match_id}.txt'
    with open(output_file, 'w') as f:
        for cropped_points in cropped_points_list:
            for point in cropped_points:
                f.write(' '.join(map(str, point)) + '\n')

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', required=True, help='Path to the config file')
    parser.add_argument('--data_path', required=True, help='Path to the Velodyne data directory')
    args = parser.parse_args()

    results_dir = Path("/home/saeed/3D_Obj_Det/Hungarian_Matching/Hungarian_Results")
    output_dir = Path("/home/saeed/3D_Obj_Det/Hungarian_Matching/Cropped_Points")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path)

    for result_file in results_dir.glob('*_results.json'):
        with open(result_file, 'r') as file:
            results = json.load(file)
            # Check if the JSON structure contains 'matches' key
            if isinstance(results, dict) and 'matches' in results:
                matches = results['matches']
            else:
                print(f"Unexpected JSON structure in file: {result_file}")
                continue

            file_name = results.get('frame_id', 'unknown_frame')
            bin_file_path = data_path / f'{file_name}.bin'
            if not bin_file_path.exists():
                print(f"Warning: {bin_file_path} does not exist.")
                continue

            # Load the point cloud
            points = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)

            print(f"Processing file: {bin_file_path}")
            boxes = []
            match_ids = set()  # To track match IDs for current file

            for entry in matches:
                rpn_3d_box = entry.get('rpn_3d')
                match_id = entry.get('id', 0)  # Extract the match_id
                
                if rpn_3d_box:
                    # Convert RPN 3D box format to the required format
                    x_center = rpn_3d_box[0]
                    y_center = rpn_3d_box[1]
                    z_center = rpn_3d_box[2]
                    dx = rpn_3d_box[3]
                    dy = rpn_3d_box[4]
                    dz = rpn_3d_box[5]
                    heading = rpn_3d_box[6]
                    boxes.append([x_center, y_center, z_center, dx, dy, dz, heading])
                    match_ids.add(match_id)
            
            if not boxes:
                print(f"No bounding boxes found for file: {bin_file_path}")
                continue

            # Crop points inside each bounding box
            cropped_points_list = crop_points_inside_boxes(points, boxes)

            # Save cropped points to text files if points are found
            if cropped_points_list:
                for match_id in match_ids:
                    save_cropped_points_to_file(cropped_points_list, output_dir, file_name, match_id)
            else:
                print(f"No points found inside bounding boxes for file: {bin_file_path}")

    print('Demo done.')

if __name__ == '__main__':
    main()

