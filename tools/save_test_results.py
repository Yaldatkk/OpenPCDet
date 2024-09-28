"""
python save_test_results.py --cfg_file cfgs/kitti_models/pointrcnn.yaml  --data_path /home/saeed/3D_Obj_Det/OpenPCDet/data/kitti/testing --ckpt ../output/kitti_models/pointrcnn/default/ckpt/checkpoint_epoch_100.pth 
"""


import argparse
import glob
from pathlib import Path
import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import open3d_vis_utils as V
import os
import cv2

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'velodyne/*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        # Use the original filename from the sample file list
        file_path = self.sample_file_list[index]  # Ensure file_path is correctly defined
    
        if self.ext == '.bin':
            points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(file_path)
        else:
            raise NotImplementedError
    
        input_dict = {
        'points': points,
        'frame_id': Path(file_path).stem,  # This extracts the filename without the extension
        }
    
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/home/saeed/3D_Obj_Det/OpenPCDet/data/kitti/testing',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--output_dir', type=str, default='/home/saeed/3D_Obj_Det/OpenPCDet/tools/test_results',
                        help='specify the output directory to save results')

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

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Processing sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)

            # Get predictions and global features
            pred_dicts, recall_dicts = model.forward(data_dict)
            #print(global_feat.shape)

            #if global_feat is not None:  # Ensure global_feat is not None
            #    global_feat_tensor = global_feat.cpu().numpy()  # Convert to numpy array
            #    global_feat_path = os.path.join(output_dir, f'global_feat_{idx}.npy')
            #    np.save(global_feat_path, global_feat_tensor)
            #    logger.info(f'Global features saved to: {global_feat_path}')
            #else:
            #    logger.warning(f'No global features for sample index: \t{idx + 1}')

    logger.info('Demo done.')


if __name__ == '__main__':
    main()

