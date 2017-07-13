#!/bin/sh

outpath=/home/store-1-img/zhenghe/chuangxin_data/im2txt_for_chuangxin_log.csv

cd /home/store-1-img/zhenghe/models/im2txt_for_chuangxin
CHUANGXIN_DIR="/home/store-1-img/zhenghe/caption_faceplusplus/chuangxinTFRECORD_data"
INCEPTION_CHECKPOINT="/home/store-1-img/zhenghe/im2txt/data/inception_v3.ckpt"
MODEL_DIR="/home/store-1-img/zhenghe/chuangxin_data/model"
bazel build -c opt im2txt/...
export CUDA_VISIBLE_DEVICES="2"
bazel-bin/im2txt/train \
  --input_file_pattern="${CHUANGXIN_DIR}/train-?????-of-00224" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=false \
  --number_of_steps=100000 > ${outpath} 2>&1 &
