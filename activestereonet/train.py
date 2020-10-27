#!/usr/bin/env python
import argparse
import os.path as osp
import logging
import time
import git
import open3d
import sys
from path import Path

sys.path.insert(0, osp.dirname(__file__) + '/..')

import torch
import torch.nn as nn

from activestereonet.config import load_cfg_from_file
from activestereonet.utils.logger import setup_logger
from activestereonet.utils.torch_utils import set_random_seed
from activestereonet.models.build_model import build_model
from activestereonet.solver import build_optimizer, build_scheduler
from activestereonet.utils.checkpoint import Checkpointer
from activestereonet.data_loader.build_data_loader import build_data_loader
from activestereonet.utils.tensorboard_logger import TensorboardLogger
from activestereonet.utils.metric_logger import MetricLogger
from activestereonet.utils.file_logger import file_logger


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch ActiveStereoNet Training")
    parser.add_argument(
        "--cfg",
        dest="config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

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
                tensorboard_logger,
                log_period=1,
                file_log_period=100,
                output_dir="",
                ):
    logger = logging.getLogger("activestereonet.train")
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
            tensorboard_logger.add_scalars(loss_dict, curr_epoch * total_iteration + iteration, prefix="train")
            tensorboard_logger.add_scalars(metric_dict, curr_epoch * total_iteration + iteration, prefix="train")

        if file_log_period == 0:
            continue
        if iteration % file_log_period == 0:
            file_logger(data_batch, preds, output_dir, prefix="train")

    return meters


def validate_model(model,
                   loss_fn,
                   metric_fn,
                   pred_invalid,
                   consistency_check,
                   data_loader,
                   curr_epoch,
                   tensorboard_logger,
                   log_period=1,
                   file_log_period=100,
                   output_dir="",
                   ):
    logger = logging.getLogger("activestereonet.validate")
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
                tensorboard_logger.add_scalars(meters.meters, curr_epoch * total_iteration + iteration, prefix="valid")

            if file_log_period == 0:
                continue
            if iteration % file_log_period == 0:
                file_logger(data_batch, preds, output_dir, prefix="valid")

    return meters


def train(cfg, output_dir=""):
    logger = logging.getLogger("activestereonet.trainer")

    # build model
    set_random_seed(cfg.RNG_SEED)
    model, loss_fn, metric_fn = build_model(cfg)
    logger.info("Build model:\n{}".format(str(model)))
    model = nn.DataParallel(model).cuda()

    # build optimizer
    optimizer = build_optimizer(cfg, model)

    # build lr scheduler
    scheduler = build_scheduler(cfg, optimizer)

    # build checkpointer
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir,
                                logger=logger)

    checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, resume=cfg.AUTO_RESUME)
    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD

    # build data loader
    train_data_loader = build_data_loader(cfg, mode="train")
    val_period = cfg.TRAIN.VAL_PERIOD
    val_data_loader = build_data_loader(cfg, mode="val") if val_period > 0 else None

    # build tensorboard logger (optionally by comment)
    tensorboard_logger = TensorboardLogger(output_dir)

    # train
    max_epoch = cfg.SCHEDULER.MAX_EPOCH
    start_epoch = checkpoint_data.get("epoch", 0)
    best_metric_name = "best_{}".format(cfg.TRAIN.VAL_METRIC)
    best_metric = checkpoint_data.get(best_metric_name, None)
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
                                   tensorboard_logger=tensorboard_logger,
                                   log_period=cfg.TRAIN.LOG_PERIOD,
                                   file_log_period=cfg.TRAIN.FILE_LOG_PERIOD,
                                   output_dir=output_dir,
                                   )
        epoch_time = time.time() - start_time
        logger.info("Epoch[{}]-Train {}  total_time: {:.2f}s".format(
            epoch, train_meters.summary_str, epoch_time))

        # checkpoint
        if cur_epoch % ckpt_period == 0 or cur_epoch == max_epoch:
            checkpoint_data["epoch"] = cur_epoch
            checkpoint_data[best_metric_name] = best_metric
            checkpointer.save("model_{:03d}".format(cur_epoch), **checkpoint_data)

        # validate
        if val_period < 1:
            continue
        if cur_epoch % val_period == 0 or cur_epoch == max_epoch:
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
                                        tensorboard_logger=tensorboard_logger,
                                        log_period=cfg.TEST.LOG_PERIOD,
                                        file_log_period=cfg.TEST.FILE_LOG_PERIOD,
                                        output_dir=output_dir,
                                        )
            logger.info("Epoch[{}]-Val {}".format(epoch, val_meters.summary_str))

            # best validation
            cur_metric = val_meters.meters[cfg.TRAIN.VAL_METRIC].global_avg
            if best_metric is None or cur_metric > best_metric:
                best_metric = cur_metric
                checkpoint_data["epoch"] = cur_epoch
                checkpoint_data[best_metric_name] = best_metric
                checkpointer.save("model_best", **checkpoint_data)

    logger.info("Best val-{} = {}".format(cfg.TRAIN.VAL_METRIC, best_metric))

    return model


def main():
    args = parse_args()
    num_gpus = torch.cuda.device_count()

    if len(args.opts) == 1:
        args.opts = args.opts[0].strip().split(" ")

    cfg = load_cfg_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace("configs", "outputs")
        output_dir = output_dir.replace('@', config_path)
        output_dir = Path(output_dir)
        output_dir.makedirs_p()

    logger = setup_logger("activestereonet", output_dir, prefix="train")
    try:
        repo = git.Repo(path=output_dir, search_parent_directories=True)
        sha = repo.head.object.hexsha
        logger.info("Git SHA: {}".format(sha))
    except:
        logger.info("No Git info")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg, output_dir)


if __name__ == "__main__":
    main()
