import torch
import torch.nn as nn
import torch.nn.functional as F


def build_cost_volume(left_tower, right_tower, num_disp):
    assert left_tower.shape == right_tower.shape
    batch_size, feature_channel, height, width = left_tower.shape
    cost_volume = torch.zeros((batch_size, feature_channel, height, width, num_disp),
                              dtype=left_tower.dtype, device=left_tower.device, requires_grad=True)
    for i in range(num_disp):
        if i == 0:
            cost_volume[:, :, :, :, i] = left_tower - right_tower
        else:
            cost_volume[:, :, :, i:, i] = left_tower[:, :, :, i:] - right_tower[:, :, :, :-i]

    cost_volume = cost_volume.contiguous()
    return cost_volume
