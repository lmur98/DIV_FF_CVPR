
VID=$1; CUDA_VISIBLE_DEVICES=1 python train_clip.py \
  --vid $VID \
  --exp_name rel/$VID \
  --train_ratio 1 --num_epochs 3 --lr 5e-4 --use_clip --video_active --ckpt_path_pretrained ckpts/rel/$VID\/epoch=9_VADIM_GEOMETRY.ckpt