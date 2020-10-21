import numpy as np
import os.path as osp
import cv2
from path import Path

import torch
import torch.nn.functional as F
import json
from activestereonet.functions.functions import get_pixel_grids
import matplotlib.pyplot as plt
import open3d as o3d
from activestereonet.utils.eval_file_logger import get_pixel_grids_np, depth2pts_np


def organized_file_logger(data_batch, preds, ref_img_path_list, output_dir):
    output_dir = Path(output_dir)

    img_list = data_batch["img_list"]
    batch_size, num_view, img_channel, img_height, img_width = list(img_list.size())
    cam_params_list = data_batch["cam_params_list"].cpu().numpy()

    for batch_idx, ref_img_path in enumerate(ref_img_path_list):
        l = ref_img_path.split("/")
        obj, texture, pos = l[-5:-2]
        save_dir = output_dir / obj / texture / pos
        save_dir.makedirs_p()
        print("start saving files in ", save_dir)

        for i in range(num_view):
            img_view = img_list[batch_idx, i].detach().permute(1, 2, 0).cpu().numpy()
            img_view = (img_view * 127.5 + 127.5).astype(np.uint8)
            cv2.imwrite(save_dir / "img_{}.png".format(i), img_view)

        cam_extrinsic = cam_params_list[batch_idx, 0, 0, :3, :4].copy()  # (3, 4)
        cam_intrinsic = cam_params_list[batch_idx, 0, 1, :3, :3].copy()

        gt_depth_map = data_batch["gt_depth_img"][batch_idx, 0].detach().cpu().numpy()
        np.save(save_dir / "gt_depth.npy", gt_depth_map)
        coarse_depth_map = preds["coarse_depth_map"][batch_idx, 0].detach().cpu().numpy()
        np.save(save_dir / "coarse_depth.npy", coarse_depth_map)
        process_depth(coarse_depth_map, gt_depth_map, cam_intrinsic, cam_extrinsic, save_dir, "coarse",
                      (img_height, img_width))

        for n in ["flow1", "flow2"]:
            if n in preds.keys():
                pred_depth_map = preds[n][batch_idx, 0].detach().cpu().numpy()
                np.save(save_dir / f"{n}_depth.npy", pred_depth_map)
                process_depth(pred_depth_map, gt_depth_map, cam_intrinsic, cam_extrinsic, save_dir, n,
                              (img_height, img_width))
        print("saving finished")


def process_depth(pred_depth_map, gt_depth_map, cam_intrinsic, cam_extrinsic, out_dir, prefix, img_size):
    metric = {}
    R = cam_extrinsic[:3, :3]
    t = cam_extrinsic[:3, 3:4]
    R_inv = np.linalg.inv(R)

    curr_cam_intrinsic = cam_intrinsic.copy()
    scale = (gt_depth_map.shape[0] + 0.0) / (img_size[0] + 0.0) * 4.0
    curr_cam_intrinsic[:2, :3] *= scale
    feature_grid = get_pixel_grids_np(gt_depth_map.shape[0], gt_depth_map.shape[1])
    uv = np.matmul(np.linalg.inv(curr_cam_intrinsic), feature_grid)
    gt_cam_points = uv * np.reshape(gt_depth_map, (1, -1))
    gt_world_points = np.matmul(R_inv, gt_cam_points - t).transpose()
    if prefix == "coarse":
        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)
        im = plt.imshow(gt_depth_map, cmap="jet")
        pos = fig.add_axes([0.88, 0.2, 0.04, 0.45])
        fig.colorbar(im, cax=pos)
        plt.savefig(out_dir / "gt_depth.png")

        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(gt_world_points)
        o3d.geometry.estimate_normals(p)
        o3d.geometry.orient_normals_towards_camera_location(p, np.matmul(R_inv, -t))
        # p.estimate_normals()
        # p.orient_normals_towards_camera_location(np.matmul(R_inv, -t))
        o3d.io.write_point_cloud(out_dir / "gt_points.pcd", p)

        gt_normals = np.array(p.normals).copy()
        gt_normals = gt_normals.reshape(gt_depth_map.shape[0], gt_depth_map.shape[1], 3)
        np.save(out_dir / "gt_normals.npy", gt_normals)
        visualize_normal_map(gt_normals, out_dir / "gt_normals.png")

    fig = plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)
    im = plt.imshow(pred_depth_map, cmap="jet")
    pos = fig.add_axes([0.88, 0.2, 0.04, 0.45])
    fig.colorbar(im, cax=pos)
    plt.savefig(out_dir / "{}_depth.png".format(prefix))
    gt_depth_map = cv2.resize(gt_depth_map, (pred_depth_map.shape[1], pred_depth_map.shape[0]))
    gt_valid_mask = (gt_depth_map > 0)
    gt_invalid_mask = (gt_depth_map == 0)
    err = pred_depth_map - gt_depth_map
    err[gt_invalid_mask] = 0
    metric["mean L1"] = np.abs(err).sum() / gt_valid_mask.sum()
    metric["max"] = err.max()
    metric["min"] = err.min()
    metric["<10 mm"] = (np.abs(err) < 10).sum() * 100.0 / gt_valid_mask.sum()
    metric["<30 mm"] = (np.abs(err) < 30).sum() * 100.0 / gt_valid_mask.sum()
    metric["<50 mm"] = (np.abs(err) < 50).sum() * 100.0 / gt_valid_mask.sum()

    fig = plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)
    im = plt.imshow(err, cmap="jet")
    pos = fig.add_axes([0.88, 0.2, 0.04, 0.45])
    fig.colorbar(im, cax=pos)
    plt.savefig(out_dir / "{}_depth_err.png".format(prefix))

    curr_cam_intrinsic = cam_intrinsic.copy()
    scale = (pred_depth_map.shape[0] + 0.0) / (img_size[0] + 0.0) * 4.0
    curr_cam_intrinsic[:2, :3] *= scale

    feature_grid = get_pixel_grids_np(pred_depth_map.shape[0], pred_depth_map.shape[1])
    uv = np.matmul(np.linalg.inv(curr_cam_intrinsic), feature_grid)

    pred_cam_points = uv * np.reshape(pred_depth_map, (1, -1))
    pred_world_points = np.matmul(R_inv, pred_cam_points - t).transpose()
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(pred_world_points)
    o3d.geometry.estimate_normals(p)
    o3d.geometry.orient_normals_towards_camera_location(p, np.matmul(R_inv, -t))
    # p.estimate_normals()
    # p.orient_normals_towards_camera_location(np.matmul(R_inv, -t))
    o3d.io.write_point_cloud(out_dir / f"{prefix}_points.pcd", p)

    pred_normals = np.array(p.normals).copy()
    pred_normals = pred_normals.reshape(pred_depth_map.shape[0], pred_depth_map.shape[1], 3)
    np.save(out_dir / f"{prefix}_normals.npy", pred_normals)

    if not prefix == "coarse":
        gt_normals = np.load(out_dir / "gt_normals.npy")
    gt_normals = cv2.resize(gt_normals, (pred_depth_map.shape[1], pred_depth_map.shape[0]))

    normal_err = np.arccos(np.clip((gt_normals * pred_normals).sum(-1), -1, 1)) * 180 / np.pi

    fig = plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)
    im = plt.imshow(normal_err, cmap="jet")
    pos = fig.add_axes([0.88, 0.2, 0.04, 0.45])
    fig.colorbar(im, cax=pos)
    plt.savefig(out_dir / "{}_norm_err.png".format(prefix))

    visualize_normal_map(pred_normals, out_dir / f"{prefix}_normals.png")

    metric["mean norm err"] = np.abs(normal_err).mean()
    metric["<5 deg"] = (np.abs(normal_err) < 5).mean()
    metric["<10 deg"] = (np.abs(normal_err) < 10).mean()

    for k, v in metric.items():
        metric[k] = str(v)
    json.dump(metric, open(out_dir / f"{prefix}_metric.json", "w"))


def visualize_normal_map(normal_map, save_path):
    normal_map = ((normal_map + 1.0) / 2.0 * 255.0).astype(np.uint8)
    cv2.imwrite(save_path, normal_map)


def file_logger(data_batch, preds, step, output_dir, prefix):
    step_dir = osp.join(output_dir, "{}_step{:05d}".format(prefix, step))
    Path(step_dir).makedirs_p()
    print("start saving files in ", step_dir)

    img_list = data_batch["img_list"]
    batch_size, num_view, img_channel, img_height, img_width = list(img_list.size())

    cam_params_list = data_batch["cam_params_list"]

    for i in range(num_view):
        img_view = img_list[0, i].detach().permute(1, 2, 0).cpu().numpy()
        img_view = (img_view * 127.5 + 127.5).astype(np.uint8)
        cv2.imwrite(osp.join(step_dir, "img_{}.png".format(i)), img_view)

    cam_extrinsic = cam_params_list[0, 0, 0, :3, :4].clone()  # (3, 4)
    cam_intrinsic = cam_params_list[0, 0, 1, :3, :3].clone()

    world_points = preds["world_points"]
    world_points = world_points[0].cpu().numpy().transpose()
    save_points(osp.join(step_dir, "world_points.xyz"), world_points)

    prob_map = preds["coarse_prob_map"][0][0].cpu().numpy()

    coarse_points = depth2pts(preds["coarse_depth_map"], prob_map,
                              cam_intrinsic, cam_extrinsic, (img_height, img_width))
    save_points(osp.join(step_dir, "coarse_point.xyz"), coarse_points)

    gt_points = depth2pts(data_batch["gt_depth_img"], prob_map,
                          cam_intrinsic, cam_extrinsic, (img_height, img_width))
    save_points(osp.join(step_dir, "gt_points.xyz"), gt_points)

    compute_depth_err_map(preds["coarse_depth_map"], data_batch["gt_depth_img"],
                          osp.join(step_dir, "coarse_depth_err.png"))
    compute_normal_err_map(preds["coarse_depth_map"], data_batch["gt_depth_img"], cam_intrinsic, step_dir,
                           "coarse", (img_height, img_width))

    if "flow1" in preds.keys():
        flow1_points = depth2pts(preds["flow1"], prob_map,
                                 cam_intrinsic, cam_extrinsic, (img_height, img_width))
        compute_depth_err_map(preds["flow1"], data_batch["gt_depth_img"],
                              osp.join(step_dir, "flow1_depth_err.png"))
        compute_normal_err_map(preds["flow1"], data_batch["gt_depth_img"], cam_intrinsic, step_dir,
                               "flow1", (img_height, img_width))
        save_points(osp.join(step_dir, "flow1_points.xyz"), flow1_points)

    if "flow2" in preds.keys():
        flow2_points = depth2pts(preds["flow2"], prob_map,
                                 cam_intrinsic, cam_extrinsic, (img_height, img_width))
        compute_depth_err_map(preds["flow2"], data_batch["gt_depth_img"],
                              osp.join(step_dir, "flow2_depth_err.png"))
        compute_normal_err_map(preds["flow2"], data_batch["gt_depth_img"], cam_intrinsic, step_dir,
                               "flow2", (img_height, img_width))
        save_points(osp.join(step_dir, "flow2_points.xyz"), flow2_points)

    print("saving finished.")


def compute_depth_err_map(pred_depth_map, gt_depth_map, out_file_path):
    pred_depth_map = pred_depth_map[0, 0].detach().cpu().numpy()
    gt_depth_map = gt_depth_map[0, 0].cpu().numpy()
    gt_depth_map = cv2.resize(gt_depth_map, (pred_depth_map.shape[1], pred_depth_map.shape[0]))
    err = pred_depth_map - gt_depth_map
    fig = plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)
    im = plt.imshow(err, cmap="jet")
    pos = fig.add_axes([0.88, 0.2, 0.04, 0.45])
    fig.colorbar(im, cax=pos)
    plt.savefig(out_file_path)


def compute_normal_err_map(pred_depth_map, gt_depth_map, cam_intrinsic, out_dir, prefix, img_size):
    pred_depth_map = pred_depth_map[0, 0].detach().cpu().numpy()
    gt_depth_map = gt_depth_map[0, 0].cpu().numpy()
    gt_depth_map = cv2.resize(gt_depth_map, (pred_depth_map.shape[1], pred_depth_map.shape[0]))
    curr_cam_intrinsic = cam_intrinsic.cpu().numpy()
    scale = (pred_depth_map.shape[0] + 0.0) / (img_size[0] + 0.0) * 4.0
    curr_cam_intrinsic[:2, :3] *= scale

    feature_grid = get_pixel_grids_np(pred_depth_map.shape[0], pred_depth_map.shape[1])
    uv = np.matmul(np.linalg.inv(curr_cam_intrinsic), feature_grid)

    pred_cam_points = uv * np.reshape(pred_depth_map, (1, -1))
    gt_cam_points = uv * np.reshape(gt_depth_map, (1, -1))

    search_param = o3d.geometry.KDTreeSearchParamKNN(30)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pred_cam_points.transpose())
    o3d.geometry.estimate_normals(pcd, search_param)
    o3d.geometry.orient_normals_towards_camera_location(pcd)
    # pcd.estimate_normals(search_param)
    # pcd.orient_normals_towards_camera_location()
    pred_normals = np.array(pcd.normals).copy()
    pred_normals = pred_normals.reshape((pred_depth_map.shape[0], pred_depth_map.shape[1], 3))
    cv2.imwrite(osp.join(out_dir, "{}_pred_normals.png".format(prefix)), (pred_normals * 255.0).astype(np.uint8))

    pcd.points = o3d.utility.Vector3dVector(gt_cam_points.transpose())
    o3d.geometry.estimate_normals(pcd, search_param)
    o3d.geometry.orient_normals_towards_camera_location(pcd)
    # pcd.estimate_normals(search_param)
    # pcd.orient_normals_towards_camera_location()
    gt_normals = np.array(pcd.normals).copy()
    gt_normals = gt_normals.reshape((pred_depth_map.shape[0], pred_depth_map.shape[1], 3))
    cv2.imwrite(osp.join(out_dir, "gt_normals.png"), (gt_normals * 255.0).astype(np.uint8))

    normal_err = np.arccos(np.clip((gt_normals * pred_normals).sum(-1), -1, 1)) * 180 / np.pi
    fig = plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)
    im = plt.imshow(normal_err, cmap="jet")
    pos = fig.add_axes([0.88, 0.2, 0.04, 0.45])
    fig.colorbar(im, cax=pos)
    plt.savefig(osp.join(out_dir, "{}_normal_err.png".format(prefix)))


def depth2pts(depth_map, prob_map, cam_intrinsic, cam_extrinsic, img_size):
    feature_map_indices_grid = get_pixel_grids(depth_map.size(2), depth_map.size(3)).to(depth_map.device)  # (3, H*W)

    curr_cam_intrinsic = cam_intrinsic.clone()
    scale = (depth_map.size(2) + 0.0) / (img_size[0] + 0.0) * 4.0
    curr_cam_intrinsic[:2, :3] *= scale

    uv = torch.matmul(torch.inverse(curr_cam_intrinsic), feature_map_indices_grid)
    cam_points = uv * depth_map[0].view(1, -1)

    R = cam_extrinsic[:3, :3]
    t = cam_extrinsic[:3, 3].unsqueeze(-1)
    R_inv = torch.inverse(R)

    world_points = torch.matmul(R_inv, cam_points - t).detach().cpu().numpy().transpose()

    curr_prob_map = prob_map.copy()
    if curr_prob_map.shape[0] != depth_map.size(2):
        curr_prob_map = cv2.resize(curr_prob_map, (depth_map.size(3), depth_map.size(2)),
                                   interpolation=cv2.INTER_LANCZOS4)
    curr_prob_map = np.reshape(curr_prob_map, (-1, 1))

    world_points = np.concatenate([world_points, curr_prob_map], axis=1)

    return world_points


def save_points(path, points):
    np.savetxt(path, points, delimiter=' ', fmt='%.4f')
