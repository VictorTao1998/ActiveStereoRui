import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
from path import Path

import json
import matplotlib.pyplot as plt
import open3d as o3d


def file_logger(data_batch, preds, output_dir, prefix, skip_exist=False):
    # only save the first object in the batch
    ref_image_path = data_batch["image_path"][0]
    output_dir = Path(output_dir)
    l = ref_image_path.split("/")
    category, scene, frame_view = l[-5:-2]
    step_dir = output_dir / prefix / category / scene / frame_view
    if step_dir.exists():
        if skip_exist:
            print("Skip ", step_dir)
            return
        else:
            print("Overwrite ", step_dir)
    Path(step_dir).makedirs_p()
    print("start saving files in ", step_dir)

    left_ir, right_ir = data_batch["left_ir"][0][0].cpu().numpy(), data_batch["right_ir"][0][0].cpu().numpy()
    left_ir = (left_ir * 127.5 + 127.5).astype(np.uint8)
    cv2.imwrite(step_dir / "img_0.png", left_ir)
    right_ir = (right_ir * 127.5 + 127.5).astype(np.uint8)
    cv2.imwrite(step_dir / "img_1.png", right_ir)

    img_height, img_width = left_ir.shape
    cam_left, cam_right = data_batch["cam_left"][0].cpu().numpy(), data_batch["cam_right"][0].cpu().numpy()
    baseline_length, focal_length = data_batch["baseline_length"][0].cpu().numpy(), data_batch["focal_length"][
        0].cpu().numpy()
    RT_left, K_left = cam_left[0], cam_left[1, :3, :3]
    RT_right, K_right = cam_right[0], cam_right[1, :3, :3]

    R = RT_left[:3, :3]
    t = RT_left[:3, 3:4]
    R_inv = np.linalg.inv(R)

    invalid_mask = data_batch["invalid_mask"][0][0].cpu().numpy()
    cv2.imwrite(step_dir / "gt_invalid_mask.png", (invalid_mask * 255.0).astype(np.uint8))
    invalid_mask_bool = invalid_mask.astype(np.bool)
    valid_mask = 1 - invalid_mask
    valid_mask_bool = valid_mask.astype(np.bool)

    object_mask = data_batch["object_mask"][0][0].cpu().numpy()
    cv2.imwrite(step_dir / "valid_object_mask.png", (object_mask * 255.0).astype(np.uint8))
    object_mask_bool = object_mask.astype(np.bool)

    # process groundtruth
    gt_disp = data_batch["disp_map"][0][0].cpu().numpy()
    gt_depth = baseline_length * focal_length / gt_disp
    gt_depth_min = gt_depth.min()
    gt_depth_max = gt_depth.max()
    depth_threshold = 100.0
    np.save(step_dir / "gt_depth.npy", gt_depth)
    fig = plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)
    im = plt.imshow(gt_depth, cmap="jet")
    pos = fig.add_axes([0.88, 0.2, 0.04, 0.45])
    fig.colorbar(im, cax=pos)
    plt.savefig(step_dir / "gt_depth.png")

    gt_world_points = depth2pts_np(gt_depth, K_left, RT_left, left_ir)
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(gt_world_points[:, :3])
    if gt_world_points.shape[1] == 6:
        p.colors = o3d.utility.Vector3dVector(gt_world_points[:, 3:] / 255.0)
    o3d.geometry.estimate_normals(p)
    o3d.geometry.orient_normals_towards_camera_location(p, np.matmul(R_inv, -t))

    gt_normals = np.array(p.normals).copy()
    gt_normals = gt_normals.reshape(left_ir.shape[0], left_ir.shape[1], 3)
    np.save(step_dir / "gt_normals.npy", gt_normals)
    visualize_normal_map(gt_normals, step_dir / "gt_normals.png")

    p = p.scale(1e-3, False)
    o3d.io.write_point_cloud(step_dir / "gt_points.pcd", p)

    # process prediction
    if "invalid_mask" in preds.keys():
        invalid_mask_pred = preds["invalid_mask"][0, 0].detach().cpu().numpy()
        plt.imsave(step_dir / "pred_invalid_mask.png", invalid_mask_pred, cmap="jet")
    for k in ["upsampled_disp", "refined_disp"]:
        kshort = k[:-5]
        metric = {}
        disp = preds[k][0][0].detach().cpu().numpy()
        depth = baseline_length * focal_length / disp
        depth = np.clip(depth, gt_depth_min - depth_threshold, gt_depth_max + depth_threshold)
        np.save(step_dir / f"{kshort}_depth.npy", depth)

        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)
        im = plt.imshow(depth, cmap="jet")
        pos = fig.add_axes([0.88, 0.2, 0.04, 0.45])
        fig.colorbar(im, cax=pos)
        plt.savefig(step_dir / f"{kshort}_depth.png")

        err = depth - gt_depth
        valid_err = err[valid_mask_bool]
        metric["val mean L1"] = np.abs(valid_err).mean()
        metric["val max L1"] = valid_err.max()
        metric["val min L1"] = valid_err.min()
        metric["val <10 mm"] = (np.abs(valid_err) < 10).mean() * 100.0
        metric["val <30 mm"] = (np.abs(valid_err) < 30).mean() * 100.0
        metric["val <50 mm"] = (np.abs(valid_err) < 50).mean() * 100.0
        obj_err = err[object_mask_bool]
        metric["obj mean L1"] = np.abs(obj_err).mean()
        metric["obj max L1"] = obj_err.max()
        metric["obj min L1"] = obj_err.min()
        metric["obj <10 mm"] = (np.abs(obj_err) < 10).mean() * 100.0
        metric["obj <30 mm"] = (np.abs(obj_err) < 30).mean() * 100.0
        metric["obj <50 mm"] = (np.abs(obj_err) < 50).mean() * 100.0

        inv_err = err[invalid_mask_bool]
        metric["inv mean L1"] = np.abs(inv_err).mean()
        metric["inv max L1"] = inv_err.max()
        metric["inv min L1"] = inv_err.min()
        metric["inv <10 mm"] = (np.abs(inv_err) < 10).mean() * 100.0
        metric["inv <30 mm"] = (np.abs(inv_err) < 30).mean() * 100.0
        metric["inv <50 mm"] = (np.abs(inv_err) < 50).mean() * 100.0

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

        normals = np.array(p.normals).copy()
        normals = normals.reshape(left_ir.shape[0], left_ir.shape[1], 3)
        np.save(step_dir / f"{kshort}_normals.npy", normals)

        p = p.scale(1e-3, False)
        o3d.io.write_point_cloud(step_dir / f"{kshort}_points.pcd", p)

        normal_err = np.arccos(np.clip(np.abs((gt_normals * normals).sum(-1)), -1, 1)) * 180 / np.pi

        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)
        im = plt.imshow(normal_err, cmap="jet")
        pos = fig.add_axes([0.88, 0.2, 0.04, 0.45])
        fig.colorbar(im, cax=pos)
        plt.savefig(step_dir / f"{kshort}_norm_err.png")

        visualize_normal_map(normals, step_dir / f"{kshort}_normals.png")
        valid_normal_err = normal_err[valid_mask_bool]
        metric["val mean norm err"] = np.abs(valid_normal_err).mean()
        metric["val <5 deg"] = (np.abs(valid_normal_err) < 5).mean() * 100.0
        metric["val <10 deg"] = (np.abs(valid_normal_err) < 10).mean() * 100.0

        obj_normal_err = normal_err[object_mask_bool]
        metric["obj mean norm err"] = np.abs(obj_normal_err).mean()
        metric["obj <5 deg"] = (np.abs(obj_normal_err) < 5).mean() * 100.0
        metric["obj <10 deg"] = (np.abs(obj_normal_err) < 10).mean() * 100.0

        inv_normal_err = normal_err[invalid_mask_bool]
        metric["inv mean norm err"] = np.abs(inv_normal_err).mean()
        metric["inv <5 deg"] = (np.abs(inv_normal_err) < 5).mean() * 100.0
        metric["inv <10 deg"] = (np.abs(inv_normal_err) < 10).mean() * 100.0

        for mk, mv in metric.items():
            metric[mk] = "{:.2f}".format(mv)
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
