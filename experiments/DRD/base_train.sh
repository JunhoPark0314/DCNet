#!/bin/bash
ROOT=../..
export PYTHONPATH=$ROOT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export NGPUS=1
SPLIT=(1)
echo "python path : $PYTHONPATH"
echo "CUDA DEVICES : $CUDA_VISIBLE_DEVICES"
echo "num gpus : $NGPUS"
echo "SPLITS : $SPLIT"
for split in ${SPLIT[*]}
do
  configfile=configs/base/e2e_voc_split${split}_base.yaml
  echo $configfile
  #python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 29512 $ROOT/tools/train_net.py --config-file ${configfile} 2>&1 | tee logs/log_${split}_basetrain.txt 
  #mv model_final.pth model_voc_split${split}_base.pth
  #mv inference/voc_2007_test_split${split}_base/result.txt fs_exp/result_split${split}_base.txt
  #rm last_checkpoint
  #python $ROOT/tools/fewshot_exp/trans_voc_pretrained.py ${split}
done