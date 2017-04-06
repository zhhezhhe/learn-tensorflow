#!/bin/sh

outpath=~/zhenghe/im2txt/im2txt_finetune_log_gpu_201704043.csv

cd ~/models/im2txt/
MSCOCO_DIR="${HOME}/zhenghe/im2txt/data/mscoco"
INCEPTION_CHECKPOINT="${HOME}/zhenghe/im2txt/data/inception_v3.ckpt"
MODEL_DIR="${HOME}/zhenghe/im2txt/model"
bazel build -c opt im2txt/...
export CUDA_VISIBLE_DEVICES="3"
# Restart the training script with --train_inception=true.
bazel-bin/im2txt/train \
  --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=true \
  --number_of_steps=3000000 > ${outpath} 2>&1 &
