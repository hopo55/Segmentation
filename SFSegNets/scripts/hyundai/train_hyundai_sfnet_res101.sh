#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=./sfets/sfnet_res18_300e_128_inplanes
mkdir -p ${EXP_DIR}
python -m torch.distributed.launch --nproc_per_node=8 train.py \
  --dataset hyundai \
  --cv 0 \
  --arch network.sfnet_resnet.DeepR101_SF_deeply \
  --class_uniform_pct 0.5 \
  --class_uniform_tile 1024 \
  --lr 0.01 \
  --lr_schedule poly \
  --poly_exp 1.0 \
  --repoly 1.5  \
  --rescale 1.0 \
  --syncbn \
  --sgd \
  --ohem \
  --crop_size 128 \
  --scale_min 0.5 \
  --scale_max 2.0 \
  --color_aug 0.25 \
  --gblur \
  --max_epoch 300 \
  --wt_bound 1.0 \
  --bs_mult 1 \
  --apex \
  --exp cityscapes_SFsegnet_res101 \
  --ckpt ${EXP_DIR}/ \
  --tb_path ${EXP_DIR}/ \
  2>&1 | tee  ${EXP_DIR}/log_${now}.txt &
