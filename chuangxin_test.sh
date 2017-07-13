#!/bin/sh
cd /home/store-1-img/zhenghe/models/im2txt_for_chuangxin/
CHECKPOINT_PATH="/home/store-1-img/zhenghe/chuangxin_data/model/train"
VOCAB_FILE="/home/store-1-img/zhenghe/caption_faceplusplus/chuangxinTFRECORD_data/word_counts.txt"
#IMAGE_FILE="${HOME}/zhenghe/im2txt/data/mscoco/raw-data/val2014/COCO_val2014_000000224477.jpg"
#IMAGE_FILE="${HOME}/zhenghe/imageForTest/images_coco/COCO_train2014_000000029799.jpg"
#bazel build -c opt im2txt/run_inference
export CUDA_VISIBLE_DEVICES="0"
file=`ls /home/store-1-img/zhenghe/caption_faceplusplus/caption_20170707_1_10000/9*.jpg`
IMAGE_FILE=''
for name in $file
do
	IMAGE_FILE=$IMAGE_FILE,$name
#	echo $name
done
#echo $IMAGE_FILE
#IMAGE_FILE="${HOME}/zhenghe/imageForTest/images_coco/1.jpg,${HOME}/zhenghe/imageForTest/images_coco/2.jpg"	
bazel-bin/im2txt/run_inference \
 	--checkpoint_path=${CHECKPOINT_PATH} \
  	--vocab_file=${VOCAB_FILE} \
  	--input_files=${IMAGE_FILE}

