import torch
import torch.nn as nn
import torch.nn.functional as F

from activestereonet.models.forward_model import ActiveStereoNet
from activestereonet.models.loss_functions import Windowed_Matching_Loss, Supervision_Loss
from activestereonet.models.metric_functions import Metric


def build_model(cfg):
    model = ActiveStereoNet(
        base_channel=cfg.MODEL.BASE_CHANNEL,
        max_disp=cfg.MODEL.MAX_DISP,
    )
    if cfg.MODEL.LOSS_TYPE == "SELF_SUPERVISE":
        loss_func = Windowed_Matching_Loss(
            lcn_kernel_size=cfg.MODEL.SELF_SUPERVISE.LCN_KERNEL_SIZE,
            window_size=cfg.MODEL.SELF_SUPERVISE.WINDOW_SIZE,
            sigma_weight=cfg.MODEL.SELF_SUPERVISE.SIGMA_WEIGHT,
            invalid_reg_weight=cfg.MODEL.INVALID_REG_WEIGHT,
            invalid_weight=cfg.MODEL.INVALID_WEIGHT,
        )
    elif cfg.MODEL.LOSS_TYPE == "SUPERVISE":
        loss_func = Supervision_Loss(
            invalid_weight=cfg.MODEL.INVALID_WEIGHT,
        )
    else:
        raise NotImplementedError

    metric_func = Metric(cfg.MODEL.INVALID_THRESHOLD)

    return model, loss_func, metric_func
