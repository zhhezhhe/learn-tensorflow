#!/bin/sh

outpath=/home/store-1-img/zhenghe/im2txt/im2txt_log0.csv

cd ~/models/im2txt/
MSCOCO_DIR="/home/store-1-img/zhenghe/im2txt/data/mscoco"
INCEPTION_CHECKPOINT="/home/store-1-img/zhenghe/im2txt/data/inception_v3.ckpt"
MODEL_DIR="/home/store-1-img/zhenghe/im2txt/model"
bazel build -c opt im2txt/...
export CUDA_VISIBLE_DEVICES="0"
bazel-bin/im2txt/train \
  --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/traintest" \
  --train_inception=false \
  --number_of_steps=1000000
