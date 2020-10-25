import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_less_percentage(disp_pred, disp_gt, threshold_list=(1.0, 3.0)):
    abs_diff_image = torch.abs(disp_pred - disp_gt)
    percentage_list = []
    for threshold in threshold_list:
        percentage = (abs_diff_image <= threshold).float().mean()
        percentage_list.append(percentage)

    return percentage_list


class Metric(nn.Module):
    def __init__(self, invalid_threshold_list=(0.5, 0.8)):
        super(Metric, self).__init__()
        self.invalid_threshold_list = invalid_threshold_list

    def forward(self, preds, data_batch):
        coarse_disp_pred = preds["coarse_disp"]
        refined_disp_pred = preds["refined_disp"]
        disp_gt = data_batch["disp_map"]

        disp_gt_resized = F.interpolate(disp_gt, (coarse_disp_pred.shape[2], coarse_disp_pred.shape[3]))
        coarse_1pixel_percent, coarse_3pixel_percent = compute_less_percentage(coarse_disp_pred, disp_gt_resized)
        refined_1pixel_percent, refined_3pixel_percent = compute_less_percentage(refined_disp_pred, disp_gt)

        metrics = {}
        metrics["cor_1pix"], metrics["cor_3pix"] = coarse_1pixel_percent, coarse_3pixel_percent
        metrics["ref_1pix"], metrics["ref_3pix"] = refined_1pixel_percent, refined_3pixel_percent

        if "invalid_mask" in preds.keys():
            invalid_mask_pred = preds["invalid_mask"]
            invalid_mask_gt = data_batch["invalid_mask"]
            for threshold in self.invalid_threshold_list:
                true_pos = (invalid_mask_pred > threshold).float() * invalid_mask_gt
                rec = true_pos.float().sum() / (invalid_mask_gt.float().sum() + 1e-6)
                pre = true_pos.float().sum() / (invalid_mask_pred.float().sum() + 1e-6)
                metrics[f"{threshold}_rec"] = rec
                metrics[f"{threshold}_pre"] = pre

        return metrics

