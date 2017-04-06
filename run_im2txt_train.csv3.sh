#!/bin/sh

outpath=/home/store-1-img/zhenghe/im2txt/im2txt_log3.csv

cd ~/models/im2txt3/
MSCOCO_DIR="/home/store-1-img/zhenghe/im2txt/data3/mscoco"
INCEPTION_CHECKPOINT="/home/store-1-img/zhenghe/im2txt/data3/inception_v3.ckpt"
MODEL_DIR="/home/store-1-img/zhenghe/im2txt/model3"
bazel build -c opt im2txt/...
export CUDA_VISIBLE_DEVICES="3"
bazel-bin/im2txt/train \
  --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=false \
  --number_of_steps=1000000000 > ${outpath} 2>&1 &
