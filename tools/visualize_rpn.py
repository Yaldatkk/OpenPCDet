import json
import numpy as np
from pathlib import Path
import argparse
from visual_utils import open3d_vis_utils as V  # Assuming Open3D is used for visualization

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

def visualize_cropped_points_and_boxes(points, cropped_points_list, boxes):
    # Visualize the original point cloud and bounding boxes
    print("Visualizing the scene with bounding boxes and cropped points...")
    V.draw_scenes(
        points=points[:, :3],         # Use XYZ coordinates from the original point cloud
        ref_boxes=np.array(boxes),    # Bounding boxes
        ref_scores=None,              # Optionally, scores can be visualized if available
        ref_labels=None               # Optionally, class labels can be visualized if available
    )

    # Optionally visualize cropped points (this can clutter the visualization)
    for cropped_points in cropped_points_list:
        V.draw_scenes(points=cropped_points[:, :3])  # Visualize cropped points inside the boxes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', required=True, help='Path to the config file')
    parser.add_argument('--data_path', required=True, help='Path to the Velodyne data directory')
    args = parser.parse_args()

    results_dir = Path("/home/saeed/3D_Obj_Det/Hungarian_Matching/Hungarian_Results")
    data_path = Path(args.data_path)

    for result_file in results_dir.glob('*_results.json'):
        print(result_file)
        with open(result_file, 'r') as file:
            results = json.load(file)
            if isinstance(results, dict) and 'matches' in results:
                matches = results['matches']
            else:
                print(f"Unexpected JSON structure in file: {result_file}")
                continue

            file_name = results.get('frame_id', 'unknown_frame')
            bin_file_path = data_path / f'{file_name}.bin'
            print(f"Processing file: {bin_file_path}")
            if not bin_file_path.exists():
                print(f"Warning: {bin_file_path} does not exist.")
                continue

            # Load the point cloud
            points = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)


            boxes = []
            match_ids = set()  # Track match IDs

            for entry in matches:
                rpn_3d_box = entry.get('rpn_3d')
                if rpn_3d_box:
                    x_center = rpn_3d_box[0]
                    y_center = rpn_3d_box[1]
                    z_center = rpn_3d_box[2]
                    dx = rpn_3d_box[3]
                    dy = rpn_3d_box[4]
                    dz = rpn_3d_box[5]
                    heading = rpn_3d_box[6]
                    boxes.append([x_center, y_center, z_center, dx, dy, dz, heading])
                    match_ids.add(entry.get('id', 0))

            if not boxes:
                print(f"No bounding boxes found for file: {bin_file_path}")
                continue

            # Crop points inside each bounding box
            cropped_points_list = crop_points_inside_boxes(points, boxes)

            # Visualize cropped points and bounding boxes
            visualize_cropped_points_and_boxes(points, cropped_points_list, boxes)

    print('Visualization complete.')

if __name__ == '__main__':
    main()

