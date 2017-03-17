#!/bin/sh
cd ~/models/im2txt/
CHECKPOINT_PATH="${HOME}/zhenghe/im2txt/model/train"
VOCAB_FILE="${HOME}/zhenghe/im2txt/data/mscoco/word_counts.txt"
IMAGE_FILE="${HOME}/zhenghe/im2txt/data/mscoco/raw-data/val2014/COCO_val2014_000000224477.jpg"
bazel build -c opt im2txt/run_inference
export CUDA_VISIBLE_DEVICES="3"
bazel-bin/im2txt/run_inference \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --vocab_file=${VOCAB_FILE} \
  --input_files=${IMAGE_FILE}

