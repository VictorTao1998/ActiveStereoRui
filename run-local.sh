#!/bin/bash
export PYTHONWARNINGS="ignore"

now=$(date +"%Y%m%d_%H%M%S")
model_name="StereoGAN"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py \
--logdir='/data/eval/ActiveStereoRui_train' \
--config-file '/code/configs/local_train_gan.yaml' \
--summary_freq 1