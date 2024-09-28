import argparse
import glob
import os
import numpy as np
import open3d as o3d
import torch
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError
        input_dict = {'points': points, 'frame_id': index}
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def load_ground_truth_labels(label_file_path):
    with open(label_file_path, 'r') as file:
        lines = file.readlines()

    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 15:
            continue
        label = int(parts[0])
        height, width, length = float(parts[8]), float(parts[9]), float(parts[10])
        x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
        rotation_y = float(parts[14])
        boxes.append((label, height, width, length, x, y, z, rotation_y))
    return boxes

def create_bounding_box(center, dimensions, rotation_y):
    h, w, l = dimensions
    x, y, z = center
    corners = np.array([
        [l/2, -w/2, h/2], [l/2, w/2, h/2], [-l/2, w/2, h/2], [-l/2, -w/2, h/2],
        [l/2, -w/2, -h/2], [l/2, w/2, -h/2], [-l/2, w/2, -h/2], [-l/2, -w/2, -h/2]
    ])
    R = np.array([
        [np.cos(rotation_y), 0, np.sin(rotation_y)],
        [0, 1, 0],
        [-np.sin(rotation_y), 0, np.cos(rotation_y)]
    ])
    corners = np.dot(corners, R.T) + np.array(center)
    lines = [
        [corners[0], corners[1]], [corners[1], corners[2]], [corners[2], corners[3]], [corners[3], corners[0]],
        [corners[4], corners[5]], [corners[5], corners[6]], [corners[6], corners[7]], [corners[7], corners[4]],
        [corners[0], corners[4]], [corners[1], corners[5]], [corners[2], corners[6]], [corners[3], corners[7]]
    ]
    return lines

def visualize_point_clouds(data_path, label_path, pred_dicts):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for filename in sorted(os.listdir(data_path)):
        if filename.endswith('.bin'):
            file_path = os.path.join(data_path, filename)
            points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points[:, 1:])  # x, y, z

            vis.clear_geometries()
            vis.add_geometry(point_cloud)

            label_file = filename.replace('.bin', '.txt')
            label_file_path = os.path.join(label_path, label_file)

            if os.path.exists(label_file_path):
                boxes = load_ground_truth_labels(label_file_path)
                for label, height, width, length, x, y, z, rotation_y in boxes:
                    box_lines = create_bounding_box((x, y, z), (height, width, length), rotation_y)
                    lines = o3d.geometry.LineSet()
                    lines.points = o3d.utility.Vector3dVector(np.array([line[0] for line in box_lines] + [line[1] for line in box_lines]))
                    lines.lines = o3d.utility.Vector2iVector([
                        [i, i + 1] for i in range(len(box_lines))
                    ] + [[7, 0], [8, 4], [9, 5], [10, 6], [11, 7]])
                    lines.paint_uniform_color([1, 0, 0])  # Red color for ground truth
                    vis.add_geometry(lines)

            if pred_dicts:
                pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
                pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
                pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()

                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    x, y, z, h, w, l, rot = box
                    box_lines = create_bounding_box((x, y, z), (h, w, l), rot)
                    lines = o3d.geometry.LineSet()
                    lines.points = o3d.utility.Vector3dVector(np.array([line[0] for line in box_lines] + [line[1] for line in box_lines]))
                    lines.lines = o3d.utility.Vector2iVector([
                        [i, i + 1] for i in range(len(box_lines))
                    ] + [[7, 0], [8, 4], [9, 5], [10, 6], [11, 7]])
                    lines.paint_uniform_color([0, 1, 0])  # Green color for predictions
                    vis.add_geometry(lines)

            vis.poll_events()
            vis.update_renderer()
            print(f"Showing: {filename}. Close the window to view the next file.")
            vis.run()  # This will wait for the window to be closed before continuing

    vis.destroy_window()

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            print(pred_dicts)

            visualize_point_clouds(args.data_path, 'label_2', pred_dicts)

    logger.info('Demo done.')

if __name__ == '__main__':
    main()

