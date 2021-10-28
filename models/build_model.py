import torch
import torch.nn as nn
import torch.nn.functional as F

from models.forward_model import ActiveStereoNet
from models.loss_functions import Windowed_Matching_Loss, Supervision_Loss
from models.metric_functions import Metric


def build_model(cfg):
    model = ActiveStereoNet(
        base_channel=cfg.MODEL.BASE_CHANNEL,
        max_disp=cfg.MODEL.MAX_DISP,
    )

    loss_func = Windowed_Matching_Loss(
        lcn_kernel_size=cfg.MODEL.SELF_SUPERVISE.LCN_KERNEL_SIZE,
        window_size=cfg.MODEL.SELF_SUPERVISE.WINDOW_SIZE,
        sigma_weight=cfg.MODEL.SELF_SUPERVISE.SIGMA_WEIGHT,
        invalid_reg_weight=cfg.MODEL.INVALID_REG_WEIGHT,
        invalid_weight=cfg.MODEL.INVALID_WEIGHT,
    )

    metric_func = Metric(cfg.MODEL.INVALID_THRESHOLD)

    return model, loss_func, metric_func
