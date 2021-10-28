#!/bin/bash
export PYTHONWARNINGS="ignore"

now=$(date +"%Y%m%d_%H%M%S")
model_name="StereoGAN"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u /jianyu-fast-vol/ActiveStereoRui/train.py \
--logdir='/jianyu-fast-vol/eval/ActiveStereoRui_train' \
--config-file '/jianyu-fast-vol/ActiveStereoRui/configs/remote_train_gan.yaml' \
--summary_freq 100