import cv2
import numpy as np
from path import Path
import logging
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from activestereonet.utils.preprocess import mask_depth_image, norm_image_255, crop_input
import activestereonet.utils.io as io
from activestereonet.utils.eval_file_logger import get_pixel_grids_np
from activestereonet.models.self_supervise import Fetch_Module


class SuLabIndoorActiveSet(Dataset):
    img_height, img_width = 512, 640
    left_img_idx, right_img_idx = 0, 1

    def __init__(self, root_dir, mode, view_list_file, use_mask=False):
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.view_list_file = view_list_file
        self.use_mask = use_mask
        self.fetch_module = Fetch_Module()

        self.path_list = self._load_dataset()
        logger = logging.getLogger("activestereonet.dataset")
        logger.info(
            "Dataset: [SULab Indoor Active]; mode: [{}]; length: [{}]; view_list_file: [{}]; use_mask: [{}]".format(
                self.mode, len(self.path_list), self.view_list_file, use_mask
            ))

    def _load_dataset(self):
        path_list = []
        with open(self.view_list_file, "r") as f:
            lines = f.readlines()
        for l in tqdm(lines):
            scene_dir = self.root_dir / l.strip()
            for frame_view_dir in sorted(scene_dir.listdir()):
                paths = {}
                view_image_paths = []
                view_cam_paths = []
                view_depth_paths = []
                view_mask_paths = []
                for view_idx in (self.left_img_idx, self.right_img_idx):
                    image_path = frame_view_dir / "coded_light" / "{}.png".format(view_idx)
                    cam_path = frame_view_dir / "cams" / "{}.txt".format(view_idx)
                    depth_path = frame_view_dir / "depth" / "{}.npy".format(view_idx)
                    if self.use_mask:
                        mask_path = frame_view_dir / "masks" / "{}.png".format(view_idx)
                        if not mask_path.exists():
                            continue
                        view_mask_paths.append(mask_path)
                    if not (image_path.exists() and cam_path.exists() and depth_path.exists()):
                        continue
                    view_image_paths.append(image_path)
                    view_cam_paths.append(cam_path)
                    view_depth_paths.append(depth_path)
                if not len(view_image_paths) == 2:
                    continue
                paths["view_image_paths"] = view_image_paths
                paths["view_cam_paths"] = view_cam_paths
                paths["view_depth_paths"] = view_depth_paths
                paths["view_mask_paths"] = view_mask_paths

                path_list.append(paths)

        return path_list

    def __getitem__(self, index):
        paths = self.path_list[index]
        images = []
        cams = []
        depths = []
        for view in range(2):
            while True:
                try:
                    image = cv2.imread(paths["view_image_paths"][view], 0)
                    image = norm_image_255(image)
                except Exception:
                    print(paths["view_image_paths"][view])
                    continue
                break
            cam = io.load_cam_indoor(paths["view_cam_paths"][view])
            depth = np.load(paths["view_depth_paths"][view]) * 1000.0
            images.append(image)
            cams.append(cam)
            depths.append(depth)

        RT_left, K_left = cams[0][0], cams[0][1, :3, :3]
        RT_right, K_right = cams[1][0], cams[1][1, :3, :3]
        assert (np.allclose(K_left, K_right)), paths["view_cam_paths"][0]

        # compute baseline_length
        RT_ij = RT_left @ np.linalg.inv(RT_right)
        assert (np.allclose(RT_ij[:3, :3], np.eye(3)))
        assert (np.sum(RT_ij[1:3, 3]**2) < 1e-6)
        baseline_length = RT_ij[0, 3]
        focal_length = K_left[0, 0]

        # convert to z-axis depth
        feature_grid = get_pixel_grids_np(depths[0].shape[0], depths[0].shape[1])
        uv = np.matmul(np.linalg.inv(K_left), feature_grid)
        uv /= np.linalg.norm(uv, axis=0, keepdims=True)
        uv = uv.reshape(3, depths[0].shape[0], depths[0].shape[1])
        for i in range(2):
            depths[i] = torch.tensor(depths[i] * uv[-1]).float()

        # compute visibility mask
        data_batch = {}
        left_disp_map = (baseline_length * focal_length / depths[0]).unsqueeze(0).unsqueeze(0)
        right_disp_map = (baseline_length * focal_length / depths[1]).unsqueeze(0).unsqueeze(0)
        reproj_disp_map = self.fetch_module(right_disp_map, left_disp_map)
        data_batch["visibility_mask"] = torch.abs(left_disp_map[0] - reproj_disp_map[0]) < 1e-3

        data_batch["left_ir"] = torch.tensor(images[0]).float().unsqueeze(0)
        data_batch["right_ir"] = torch.tensor(images[1]).float().unsqueeze(0)
        data_batch["disp_map"] = left_disp_map[0]

        return data_batch

    def __len__(self):
        return len(self.path_list)


if __name__ == '__main__':
    dataset = SuLabIndoorActiveSet(
        root_dir="/home/xulab/Nautilus/",
        mode="train",
        view_list_file="/home/xulab/Nautilus/example.txt"
    )

    print("length", dataset.__len__())

    for k, v in dataset[0].items():
        print(k, v.shape)
