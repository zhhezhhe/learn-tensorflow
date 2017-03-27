# learn-tensorflow

export CUDA_VISIBLE_DEVICES="" using cpu</br>
export CUDA_VISIBLE_DEVICES="0,1,2,3"  using 4 gpus : 0,1,2,3</br> 
export CUDA_VISIBLE_DEVICES="3" using one gpu: 3 </br>


# Pretrained model for im2txt
https://github.com/tensorflow/models/issues/466</br>
## initial training (1m steps without finetuning inception). perplexity ~8.7
https://drive.google.com/open?id=0Bw6m_66JSYLlNlRDUTRqcm9Jcjg</br>
## finetuned
https://drive.google.com/file/d/0Bw6m_66JSYLlRFVKQ2tGcUJaWjA/view?usp=sharing</br>
# nvidia-smi 记录显卡信息到csv文件
nvidia-smi --query-gpu=index,uuid,utilization.gpu,timestamp,pstate,utilization.memory,memory.total,memory.free,memory.used,temperature.gpu --format=csv -l 10 -f ./GPU-0316-stats.csv</br>
# 显示csv中log信息
tail -f *.csv
# 显示程序使用情况
ps aux | grep python
# kill 进程
kill -9 22222
# 修改服务器别名
sudo vim /etc/hosts
# tensorboard
MODEL_DIR="${HOME}/im2txt/model"</br>
{# Run a TensorBoard server.}</br>
tensorboard --logdir="${MODEL_DIR}"</br>
