#!/usr/bin/env python
import argparse
import os.path as osp
import logging
import time
import git
import sys
import open3d
from path import Path

sys.path.insert(0, osp.dirname(__file__) + '/..')

import torch
import torch.nn as nn

from activestereonet.config import load_cfg_from_file
from activestereonet.utils.logger import setup_logger
from activestereonet.models.build_model import build_model
from activestereonet.utils.checkpoint import Checkpointer
from activestereonet.data_loader.build_data_loader import build_data_loader
from activestereonet.utils.metric_logger import MetricLogger
from activestereonet.utils.file_logger import file_logger


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch ActiveStereoNet Evaluation")
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


def test_model(model,
               loss_fn,
               metric_fn,
               pred_invalid,
               consistency_check,
               data_loader,
               log_period=1,
               output_dir="",
               ):
    logger = logging.getLogger("activestereonet.test")
    meters = MetricLogger(delimiter="  ")
    # model.train()
    model.eval()
    end = time.time()

    iteration = 0
    with torch.no_grad():
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
                            "iter: {iter:4d}",
                            "{meters}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        iter=iteration,
                        meters=str(meters),
                        memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                    )
                )

            file_logger(data_batch, preds, output_dir, prefix="test")


def test(cfg, output_dir):
    logger = logging.getLogger("activestereonet.tester")
    # build model
    model, loss_func, metric_func = build_model(cfg)
    model = nn.DataParallel(model).cuda()

    # build checkpointer
    checkpointer = Checkpointer(model, save_dir=output_dir)

    if cfg.TEST.WEIGHT:
        weight_path = cfg.TEST.WEIGHT.replace("@", output_dir)
        checkpointer.load(weight_path, resume=False)
    else:
        checkpointer.load(None, resume=True)

    # build data loader
    test_data_loader = build_data_loader(cfg, mode="test")
    start_time = time.time()
    test_model(model,
               loss_func,
               metric_func,
               pred_invalid=True,
               consistency_check=False,
               data_loader=test_data_loader,
               log_period=cfg.TEST.LOG_PERIOD,
               output_dir=output_dir,
               )
    test_time = time.time() - start_time
    logger.info("Test forward time: {:.2f}s".format(test_time))


def main():
    args = parse_args()
    num_gpus = torch.cuda.device_count()

    if len(args.opts) == 1:
        args.opts = args.opts[0].strip().split(" ")

    cfg = load_cfg_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    assert cfg.TEST.BATCH_SIZE == 1

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace("configs", "outputs")
        output_dir = output_dir.replace('@', config_path)
        Path(output_dir).makedirs_p()

    logger = setup_logger("activestereonet", output_dir, prefix="test")
    try:
        repo = git.Repo(path=output_dir, search_parent_directories=True)
        sha = repo.head.object.hexsha
        logger.info("Git SHA: {}".format(sha))
    except:
        logger.info("No git info")

    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    test(cfg, output_dir)


if __name__ == "__main__":
    main()
