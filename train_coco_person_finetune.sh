#!/bin/sh

outpath=/home/store-1-img/zhenghe/coco_person/im2txt_coco_person_finetune.csv

cd /home/store-1-img/zhenghe/models/im2txt
COCO_PERSON_DIR="/home/store-1-img/zhenghe/coco_person/TFRECORD_data"
INCEPTION_CHECKPOINT="/home/store-1-img/zhenghe/im2txt/data/inception_v3.ckpt"
MODEL_DIR="/home/store-1-img/zhenghe/coco_person/model"
bazel build -c opt im2txt/...
export CUDA_VISIBLE_DEVICES="1"
bazel-bin/im2txt/train \
  --input_file_pattern="${COCO_PERSON_DIR}/train-?????-of-00032" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=true \
  --number_of_steps=300000 > ${outpath} 2>&1 &
