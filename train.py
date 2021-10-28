#!/usr/bin/env python
import argparse
import os
import logging
import time
import sys

from utils.config import cfg


import torch
import torch.nn as nn


from utils.torch_utils import set_random_seed
from models.build_model import build_model
from solver import build_optimizer, build_scheduler
from data_loader.build_data_loader import build_data_loader
from utils.metric_logger import MetricLogger
from tensorboardX import SummaryWriter
from utils.util import *
from utils.warp_ops import apply_disparity_cu
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch ActiveStereoNet Training")
    parser.add_argument('--config-file', type=str, default='./configs/local_train_steps.yaml',
                        metavar='FILE', help='Config files')
    parser.add_argument('--summary_freq', type=int, default=500, help='Frequency of saving temporary results')
    parser.add_argument('--save-freq', type=int, default=1000, help='Frequency of saving checkpoint')
    parser.add_argument('--logdir', required=True, help='Directory to save logs and checkpoints')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)')
    parser.add_argument('--debug', action='store_true', help='Whether run in debug mode (will load less data)')
    parser.add_argument('--warp_op', action='store_true',default=True, help='whether use warp_op function to get disparity')
    parser.add_argument('--loadmodel', type=str, help='load pretrained model')

    args = parser.parse_args()
    return args


def train_model(model,
                loss_fn,
                metric_fn,
                pred_invalid,
                consistency_check,
                data_loader,
                optimizer,
                curr_epoch,
                writer,
                log_period=1,
                file_log_period=100,
                output_dir="",
                ):
    logger = setup_logger("ActiveStereoRui_TRAIN", distributed_rank=0, save_dir=output_dir)
    meters = MetricLogger(delimiter="  ")
    model.train()
    end = time.time()
    total_iteration = data_loader.__len__()
    path_list = []

    iteration = 0
    for data_batch in data_loader:
        iteration += 1
        data_time = time.time() - end
        data_batch_input = {}
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor):
                data_batch_input[k] = v.cuda(non_blocking=True)
            else:
                data_batch_input[k] = v
        data_batch = data_batch_input

        img_disp_r = data_batch['img_disp_r']
        img_disp_r = F.interpolate(img_disp_r, scale_factor=0.5, mode='nearest',
                                   recompute_scale_factor=False)
        warp_gt = apply_disparity_cu(img_disp_r, img_disp_r.type(torch.int))  # [bs, 1, H, W]
        invalid_mask = (warp_gt == 0).float()
        data_batch['invalid_mask'] = (1 - invalid_mask).float().cuda()
        
        del img_disp_r

        preds = model(data_batch, pred_invalid, consistency_check)
        optimizer.zero_grad()

        loss_dict = loss_fn(preds=preds, data_batch=data_batch)
        metric_dict = metric_fn(preds=preds, data_batch=data_batch)
        losses = sum(loss_dict.values())
        meters.update(loss=losses, **loss_dict, **metric_dict)

        losses.backward()

        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        if iteration % log_period == 0:
            logger.info(
                meters.delimiter.join(
                    [
                        "EPOCH: {epoch:2d}",
                        "iter: {iter:4d}",
                        "{meters}",
                        "lr: {lr:.2e}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    epoch=curr_epoch,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )
            save_scalars(writer, 'train', loss_dict, curr_epoch * total_iteration + iteration)
            save_scalars(writer, 'train', metric_dict, curr_epoch * total_iteration + iteration)


    return meters


def validate_model(model,
                   loss_fn,
                   metric_fn,
                   pred_invalid,
                   consistency_check,
                   data_loader,
                   curr_epoch,
                   writer,
                   log_period=1,
                   file_log_period=100,
                   output_dir="",
                   ):
    logger = setup_logger("ActiveStereoRui_VAL", distributed_rank=0, save_dir=output_dir)
    meters = MetricLogger(delimiter="  ")
    model.train()
    end = time.time()
    total_iteration = data_loader.__len__()
    with torch.no_grad():
        iteration = 0
        for data_batch in data_loader:
            iteration += 1
            data_time = time.time() - end
            data_batch_input = {}
            for k, v in data_batch.items():
                if isinstance(v, torch.Tensor):
                    data_batch_input[k] = v.cuda(non_blocking=True)
                else:
                    data_batch_input[k] = v
            data_batch = data_batch_input

            img_disp_r = data_batch['img_disp_r']
            img_disp_r = F.interpolate(img_disp_r, scale_factor=0.5, mode='nearest',
                                    recompute_scale_factor=False)
            warp_gt = apply_disparity_cu(img_disp_r, img_disp_r.type(torch.int))  # [bs, 1, H, W]
            invalid_mask = (warp_gt == 0).float()
            data_batch['invalid_mask'] = (1 - invalid_mask).float().cuda()
            
            del img_disp_r

            preds = model(data_batch, pred_invalid, consistency_check)
            loss_dict = loss_fn(preds=preds, data_batch=data_batch)
            metric_dict = metric_fn(preds=preds, data_batch=data_batch)
            losses = sum(loss_dict.values())
            meters.update(loss=losses, **loss_dict, **metric_dict)
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            if iteration % log_period == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "EPOCH: {epoch:2d}",
                            "iter: {iter:4d}",
                            "{meters}",
                        ]
                    ).format(
                        epoch=curr_epoch,
                        iter=iteration,
                        meters=str(meters),
                    )
                )
                save_scalars(writer, 'val', meters.meters, curr_epoch * total_iteration + iteration)


    return meters

def save_checkpoint(self, best=False):

        ckpt_root = os.path.join(self.args.logdir, 'checkpoints')

        if not os.path.exists(ckpt_root):
            os.makedirs(ckpt_root) 
        
        ckpt_name = 'ep_{:d}.pth'.format(self.epoch)

        if best:
            ckpt_name = 'best_epe_{:f}.pth'.format(self.best_epe)

        states = {
            'epoch': self.epoch,
            'best_epe': self.best_epe,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
        }
        ckpt_full = os.path.join(ckpt_root, ckpt_name)
        
        torch.save(states, ckpt_full)


def train(args, cfg, output_dir, writer):
    logger = setup_logger("ActiveStereoRui_TRAINER", distributed_rank=0, save_dir=output_dir)

    # build model
    #set_random_seed(cfg.RNG_SEED)
    model, loss_fn, metric_fn = build_model(cfg)
    logger.info("Build model:\n{}".format(str(model)))
    model = nn.DataParallel(model).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,3,4], gamma=0.5)

    start_epoch = 0
    best_metric = None
    if args.loadmodel:
        ckpt_root = os.path.join(args.loadmodel)

        states = torch.load(ckpt_root, map_location=lambda storage, loc: storage)

        start_epoch = states['epoch']
        best_metric = states['best_metric']
        model.load_state_dict(states['model_state'])
        optimizer.load_state_dict(states['optimizer_state'])
        scheduler.load_state_dict(states['scheduler_state'])


    # build data loader
    train_data_loader = build_data_loader(cfg, mode="train")
    val_data_loader = build_data_loader(cfg, mode="val") 


    # train
    max_epoch = cfg.SOLVER.EPOCHS
    
    best_metric_name = "best_{}".format(cfg.TRAIN.VAL_METRIC)

    logger.info("Start training from epoch {}".format(start_epoch))
    for epoch in range(start_epoch, max_epoch):
        cur_epoch = epoch + 1
        scheduler.step()
        start_time = time.time()
        train_meters = train_model(model,
                                   loss_fn,
                                   metric_fn,
                                   pred_invalid=(cfg.MODEL.LOSS_TYPE == "SUPERVISE") or
                                                ((cur_epoch > cfg.SCHEDULER.INIT_EPOCH) and
                                                 (cfg.MODEL.LOSS_TYPE == "SELF_SUPERVISE")),
                                   consistency_check=
                                   (cur_epoch > cfg.SCHEDULER.INIT_EPOCH) and (cfg.MODEL.LOSS_TYPE == "SELF_SUPERVISE"),
                                   data_loader=train_data_loader,
                                   optimizer=optimizer,
                                   curr_epoch=epoch,
                                   writer=writer,
                                   log_period=args.summary_freq,
                                   file_log_period=100000,
                                   output_dir=output_dir,
                                   )
        epoch_time = time.time() - start_time
        logger.info("Epoch[{}]-Train {}  total_time: {:.2f}s".format(
            epoch, train_meters.summary_str, epoch_time))

        # checkpoint
        ckpt_root = os.path.join(output_dir, 'checkpoints')

        if not os.path.exists(ckpt_root):
            os.makedirs(ckpt_root) 
        
        ckpt_name = 'ep_{:d}.pth'.format(cur_epoch)
        states = {
            'epoch': cur_epoch,
            'best_metric': None,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict()
        }
        ckpt_full = os.path.join(ckpt_root, ckpt_name)
        
        torch.save(states, ckpt_full)

        # validate

        val_meters = validate_model(model,
                                    loss_fn,
                                    metric_fn,
                                    pred_invalid=(cfg.MODEL.LOSS_TYPE == "SUPERVISE") or
                                                    ((cur_epoch > cfg.SCHEDULER.INIT_EPOCH) and
                                                    (cfg.MODEL.LOSS_TYPE == "SELF_SUPERVISE")),
                                    consistency_check=(cur_epoch > cfg.SCHEDULER.INIT_EPOCH) and (
                                            cfg.MODEL.LOSS_TYPE == "SELF_SUPERVISE"),
                                    data_loader=val_data_loader,
                                    curr_epoch=epoch,
                                    writer=writer,
                                    log_period=100,
                                    file_log_period=1000000,
                                    output_dir=output_dir,
                                    )
        logger.info("Epoch[{}]-Val {}".format(epoch, val_meters.summary_str))

        # best validation
        cur_metric = val_meters.meters[cfg.TRAIN.VAL_METRIC].global_avg
        if best_metric is None or cur_metric > best_metric:
            best_metric = cur_metric

            ckpt_root = os.path.join(output_dir, 'checkpoints')

            if not os.path.exists(ckpt_root):
                os.makedirs(ckpt_root) 
            
            ckpt_name = 'best_ep_{:d}_.pth'.format(cur_epoch)
            states = {
                'epoch': cur_epoch,
                'best_metric': best_metric,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict()
            }
            ckpt_full = os.path.join(ckpt_root, ckpt_name)
            
            torch.save(states, ckpt_full)

    logger.info("Best val-{} = {}".format(cfg.TRAIN.VAL_METRIC, best_metric))

    return model


def main():
    args = parse_args()
    num_gpus = torch.cuda.device_count()

    cfg.merge_from_file(args.config_file)

    output_dir = args.logdir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    logger = setup_logger("activestereonet", 0, output_dir)

    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    writer = SummaryWriter(args.logdir)

    train(args, cfg, output_dir, writer)


if __name__ == "__main__":
    main()
