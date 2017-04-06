#!/bin/sh
cd ~/models/im2txt/
CHECKPOINT_PATH="${HOME}/zhenghe/im2txt/model/train"
VOCAB_FILE="${HOME}/zhenghe/im2txt/data/mscoco/word_counts.txt"
#IMAGE_FILE="${HOME}/zhenghe/im2txt/data/mscoco/raw-data/val2014/COCO_val2014_000000224477.jpg"
#IMAGE_FILE="${HOME}/zhenghe/imageForTest/images_coco/COCO_train2014_000000029799.jpg"
#bazel build -c opt im2txt/run_inference
export CUDA_VISIBLE_DEVICES=""
file=`ls ${HOME}/zhenghe/imageForTest/images_coco/*.jpg`
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

