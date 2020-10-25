import torch
import torch.nn as nn
import torch.nn.functional as F

from activestereonet.modules.networks import SiameseTower, CostVolumeFilter, DisparityRefinement, InvalidationNetwork
from activestereonet.modules.functions import build_cost_volume


class ActiveStereoNet(nn.Module):
    def __init__(self, base_channel, max_disp):
        super(ActiveStereoNet, self).__init__()
        self.num_disp = (max_disp // 8) + 1
        self.siamese_tower = SiameseTower(1, base_channel)
        self.cost_volume_filter = CostVolumeFilter(base_channel)
        self.disp_refine_net = DisparityRefinement(base_channel)
        self.invalid_net = InvalidationNetwork(base_channel)

    def forward(self, data_batch, pred_invalid=False, consistency_check=False):
        preds = {}
        left_ir, right_ir = data_batch["left_ir"], data_batch["right_ir"]
        batch_size, _, height, width = left_ir.shape
        left_tower_feature = self.siamese_tower(left_ir)
        right_tower_feature = self.siamese_tower(right_ir)
        cost_volume = build_cost_volume(left_tower_feature, right_tower_feature, self.num_disp)
        filtered_cost_volume = self.cost_volume_filter(cost_volume)
        filtered_cost_volume = filtered_cost_volume.squeeze(1)
        filtered_cost_volume = F.softmax(filtered_cost_volume, dim=-1)

        disp_array = torch.linspace(0, self.num_disp - 1, self.num_disp, dtype=left_ir.dtype, device=left_ir.device) \
            .view((1, 1, 1, self.num_disp))

        coarse_disp_pred = (disp_array * filtered_cost_volume).sum(-1).unsqueeze(1)
        preds["coarse_disp"] = coarse_disp_pred

        upsampled_disp_pred = F.upsample_bilinear(coarse_disp_pred, (height, width))
        preds["upsampled_disp"] = upsampled_disp_pred
        normed_upsampled_disp_pred = upsampled_disp_pred / self.num_disp
        refined_disp_pred = self.disp_refine_net(normed_upsampled_disp_pred, left_ir)

        if pred_invalid:
            invalid_mask_pred = self.invalid_net(left_tower_feature, right_tower_feature, refined_disp_pred, left_ir)
            preds["invalid_mask"] = invalid_mask_pred
            if consistency_check:
                # right image disparity prediction for consistency check
                with torch.no_grad():
                    flip_left_ir = torch.flip(left_ir, [3])
                    flip_right_ir = torch.flip(right_ir, [3])
                    flip_left_tower_feature = self.siamese_tower(flip_left_ir)
                    flip_right_tower_feature = self.siamese_tower(flip_right_ir)
                    flip_cost_volume = build_cost_volume(flip_right_tower_feature, flip_left_tower_feature,
                                                         self.num_disp)
                    flip_filtered_cost_volume = self.cost_volume_filter(flip_cost_volume)
                    flip_filtered_cost_volume = flip_filtered_cost_volume.squeeze(1)
                    flip_filtered_cost_volume = F.softmax(flip_filtered_cost_volume, dim=-1)
                    disp_array = torch.linspace(0, self.num_disp - 1, self.num_disp, dtype=left_ir.dtype,
                                                device=left_ir.device).view((1, 1, 1, self.num_disp))
                    flip_coarse_disp_pred = (disp_array * flip_filtered_cost_volume).sum(-1).unsqueeze(1)
                    flip_upsampled_disp_pred = F.upsample_bilinear(flip_coarse_disp_pred, (height, width))
                    flip_upsampled_disp_pred = flip_upsampled_disp_pred / self.num_disp
                    flip_upsampled_disp_pred = self.disp_refine_net(flip_upsampled_disp_pred, flip_right_ir)
                    flip_right_disp_pred = flip_upsampled_disp_pred * self.num_disp
                    preds["right_disp"] = torch.flip(flip_right_disp_pred, [3])

        refined_disp_pred = refined_disp_pred * self.num_disp
        preds["refined_disp"] = refined_disp_pred

        return preds


if __name__ == '__main__':
    batch_size = 1
    heigt = 128
    width = 192

    active_stereo_net = ActiveStereoNet(base_channel=32, num_disp=18)
    data_batch = {"left_ir": torch.rand((batch_size, 1, heigt, width)).float(),
                  "right_ir": torch.rand((batch_size, 1, heigt, width)).float()}

    preds = active_stereo_net(data_batch)
    for k, v in preds.items():
        print(k, v.shape)
