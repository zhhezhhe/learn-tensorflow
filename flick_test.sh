#!/bin/sh
cd /home/store-1-img/zhenghe/models/im2txt_for_flick
#CHECKPOINT_PATH="/home/store-1-img/zhenghe/flickr8kcn/model/train"
CHECKPOINT_PATH="/home/store-1-img/zhenghe/chuangxin_data/model/train"
#VOCAB_FILE="/home/store-1-img/zhenghe/flickr8kcn/TFRECORD_data/word_counts.txt"
VOCAB_FILE="/home/store-1-img/zhenghe/caption_faceplusplus/chuangxinTFRECORD_data/word_counts.txt"
IMAGE_FILE="/home/store-1-img/zhenghe/flickr8kcn/Flicker8k_Dataset/33108590_d685bfe51c.jpg"

#IMAGE_FILE="/media/zh/E/2.jpg"
bazel build -c opt im2txt/run_inference
export CUDA_VISIBLE_DEVICES=""
bazel-bin/im2txt/run_inference \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --vocab_file=${VOCAB_FILE} \
  --input_files=${IMAGE_FILE}

