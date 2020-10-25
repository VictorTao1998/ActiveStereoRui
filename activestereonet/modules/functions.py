import torch
import torch.nn as nn
import torch.nn.functional as F


def build_cost_volume(left_tower, right_tower, num_disp):
    assert left_tower.shape == right_tower.shape
    batch_size, feature_channel, height, width = left_tower.shape
    # cost_volume = left_tower.unsqueeze(-1).expand((batch_size, feature_channel, height, width, num_disp))
    # for i in range(num_disp):
    #     if i == 0:
    #         cost_volume[:, :, :, :, i] -= right_tower
    #     else:
    #         cost_volume[:, :, :, i:, i] -= right_tower[:, :, :, :-i]
    # cost_volume = cost_volume.contiguous()
    cost_list = []
    for i in range(num_disp):
        if i == 0:
            cost = left_tower - right_tower
        else:
            cost = left_tower.clone()
            cost[:, :, :, i:] = left_tower[:, :, :, i:] - right_tower[:, :, :, :-i]
        cost_list.append(cost)

    cost_volume = torch.stack(cost_list, dim=-1).contiguous()
    return cost_volume
