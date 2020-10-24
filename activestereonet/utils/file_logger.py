import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
from path import Path

import json
import matplotlib.pyplot as plt
import open3d as o3d


def file_logger(data_batch, preds, output_dir, prefix):
    # only save the first object in the batch
    ref_image_path = data_batch["image_path"][0]
    output_dir = Path(output_dir)
    l = ref_image_path.split("/")
    category, scene, frame_view = l[-5:-2]
    step_dir = output_dir / prefix / category / scene / frame_view
    Path(step_dir).makedirs_p()
    print("start saving files in ", step_dir)

    left_ir, right_ir = data_batch["left_ir"][0][0], data_batch["right_ir"][0][0]
    left_ir = (left_ir * 127.5 + 127.5).astype(np.uint8)
    cv2.imwrite(step_dir / "img_0.png", left_ir)
    right_ir = (right_ir * 127.5 + 127.5).astype(np.uint8)
    cv2.imwrite(step_dir / "img_1.png", right_ir)

    img_height, img_width = left_ir.shape
    cam_left, cam_right = data_batch["cam_left"][0].numpy(), data_batch["cam_right"][0].numpy()
    baseline_length, focal_length = data_batch["baseline_length"][0].numpy(), data_batch["focal_length"][0].numpy()
    RT_left, K_left = cam_left[0], cam_left[1, :3, :3]
    RT_right, K_right = cam_right[0], cam_right[1, :3, :3]

    R = RT_left[:3, :3]
    t = RT_left[:3, 3:4]
    R_inv = np.linalg.inv(R)

    invalid_mask = data_batch["invalid_mask"][0][0]
    invalid_mask_bool = invalid_mask.astype(np.bool)
    valid_mask = 1 - invalid_mask
    valid_mask_bool = valid_mask.astype(np.bool)

    # process groundtruth
    gt_disp = data_batch["disp_map"][0][0].numpy()
    gt_depth = baseline_length * focal_length / gt_disp
    np.save(step_dir / "gt_depth.npy", gt_depth)
    fig = plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)
    im = plt.imshow(gt_depth, cmap="jet")
    pos = fig.add_axes([0.88, 0.2, 0.04, 0.45])
    fig.colorbar(im, cax=pos)
    plt.savefig(step_dir / "gt_depth.png")

    gt_world_points = depth2pts_np(gt_depth, K_left, RT_left, left_ir)
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(gt_world_points)
    o3d.geometry.estimate_normals(p)
    o3d.geometry.orient_normals_towards_camera_location(p, np.matmul(R_inv, -t))
    o3d.io.write_point_cloud(step_dir / "gt_points.pcd", p)

    gt_normals = np.array(p.normals).copy()
    gt_normals = gt_normals.reshape(left_ir.shape[0], left_ir.shape[1], 3)
    np.save(step_dir / "gt_normals.npy", gt_normals)
    visualize_normal_map(gt_normals, step_dir / "gt_normals.png")

    # process prediction
    for k in ["upsampled_disp", "refined_disp"]:
        kshort = k[:-5]
        metric = {}
        disp = preds[k][0][0].detach().cpu().numpy()
        depth = baseline_length * focal_length / disp
        np.save(step_dir / f"{kshort}_depth.npy", depth)

        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)
        im = plt.imshow(depth, cmap="jet")
        pos = fig.add_axes([0.88, 0.2, 0.04, 0.45])
        fig.colorbar(im, cax=pos)
        plt.savefig(step_dir / f"{kshort}_depth.png")

        err = depth - gt_depth
        err[invalid_mask_bool] = 0.0
        metric["mean L1"] = np.abs(err).sum() / valid_mask_bool.sum()
        metric["max"] = err.max()
        metric["min"] = err.min()
        metric["<10 mm"] = (np.abs(err) < 10).sum() * 100.0 / valid_mask_bool.sum()
        metric["<30 mm"] = (np.abs(err) < 30).sum() * 100.0 / valid_mask_bool.sum()
        metric["<50 mm"] = (np.abs(err) < 50).sum() * 100.0 / valid_mask_bool.sum()

        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)
        im = plt.imshow(err, cmap="jet")
        pos = fig.add_axes([0.88, 0.2, 0.04, 0.45])
        fig.colorbar(im, cax=pos)
        plt.savefig(step_dir / f"{kshort}_depth_err.png")

        pred_world_points = depth2pts_np(depth, K_left, RT_left)
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(pred_world_points)
        o3d.geometry.estimate_normals(p)
        o3d.geometry.orient_normals_towards_camera_location(p, np.matmul(R_inv, -t))
        o3d.io.write_point_cloud(step_dir / f"{kshort}_points.pcd", p)

        normals = np.array(p.normals).copy()
        normals = normals.reshape(left_ir.shape[0], left_ir.shape[1], 3)
        np.save(step_dir / f"{kshort}_normals.npy", normals)

        normal_err = np.arccos(np.clip((gt_normals * normals).sum(-1), -1, 1)) * 180 / np.pi

        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)
        im = plt.imshow(normal_err, cmap="jet")
        pos = fig.add_axes([0.88, 0.2, 0.04, 0.45])
        fig.colorbar(im, cax=pos)
        plt.savefig(step_dir / f"{kshort}_norm_err.png")

        visualize_normal_map(normals, step_dir / f"{kshort}_normals.png")

        metric["mean norm err"] = np.abs(normal_err).mean()
        metric["<5 deg"] = (np.abs(normal_err) < 5).mean()
        metric["<10 deg"] = (np.abs(normal_err) < 10).mean()

        for k, v in metric.items():
            metric[k] = str(v)
        json.dump(metric, open(step_dir / f"{kshort}_metric.json", "w"))

    print("saving finished.")


def depth2pts_np(depth_map, cam_intrinsic, cam_extrinsic, color_map=None):
    feature_grid = get_pixel_grids_np(depth_map.shape[0], depth_map.shape[1])

    uv = np.matmul(np.linalg.inv(cam_intrinsic), feature_grid)
    cam_points = uv * np.reshape(depth_map, (1, -1))

    R = cam_extrinsic[:3, :3]
    t = cam_extrinsic[:3, 3:4]
    R_inv = np.linalg.inv(R)

    world_points = np.matmul(R_inv, cam_points - t).transpose()
    if color_map is None:
        return world_points
    else:
        if len(color_map.shape) == 2:
            color_map = color_map.reshape(-1, 1)
            color_map = color_map.repeat(3, 1)
            world_points = np.concatenate([world_points, color_map], axis=-1)
            return world_points
        elif len(color_map.shape) == 3:
            color_map = color_map.reshape(-1, color_map.shape[-1])
            if color_map.shape[-1] == 1:
                color_map = color_map.repeat(3, 1)
            elif color_map.shape[-1] == 3:
                color_map = color_map
            else:
                raise NotImplementedError
            world_points = np.concatenate([world_points, color_map], axis=-1)
            return world_points
        else:
            raise NotImplementedError


def get_pixel_grids_np(height, width):
    x_linspace = np.linspace(0.5, width - 0.5, width)
    y_linspace = np.linspace(0.5, height - 0.5, height)
    x_coordinates, y_coordinates = np.meshgrid(x_linspace, y_linspace)
    x_coordinates = np.reshape(x_coordinates, (1, -1))
    y_coordinates = np.reshape(y_coordinates, (1, -1))
    ones = np.ones_like(x_coordinates).astype(np.float)
    grid = np.concatenate([x_coordinates, y_coordinates, ones], axis=0)

    return grid


def visualize_normal_map(normal_map, save_path):
    normal_map = ((normal_map + 1.0) / 2.0 * 255.0).astype(np.uint8)
    cv2.imwrite(save_path, normal_map)


