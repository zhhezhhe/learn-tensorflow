#!/bin/sh
nvidia-docker run -ti -v /home/store-1-img/MSCOCO/mscoco/raw-data:/workspace/mscoco -v /home/store-1-img/zhenghe/DCC:/workspace/DCC bvlc/caffe:DCC /bin/bash
#nvidia-docker run -it --volumes-from caffe gpu/caffe /bin/bash
