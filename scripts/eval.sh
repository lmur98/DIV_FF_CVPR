
CKP=ckpts/$1
VID=$2
EXP=$3
OUT=$4
MASKS_N_SAMPLES=$5
SUMMARY_N_SAMPLES=$6

EPOCH=2

CUDA_VISIBLE_DEVICES=1 python evaluate.py \
  --path $CKP\/$VID\/epoch\=$EPOCH\_FINAL_MODELv2.ckpt \
  --vid $VID --exp $EXP \
  --is_eval_script \
  --outputs $OUT \
  --masks_n_samples $MASKS_N_SAMPLES \
  --summary_n_samples $SUMMARY_N_SAMPLES \
  --positive_queries 'plate','pasta','blue colander'\
  --use_clip  --video_active 
  #--suppress_person
  
#"sink","cutting board","knife","frying pan" \
#"#C C washes the ingredient in the sink","#C C cuts the vegetables in the cutting board","#C C takes the knife with the right hand","#C C adds the ingredients to the frying pan" \
#"The location where the person is cutting the vegetables","The location where the person washes the dishes","The location where I can cook food","The location where I can get some spices"\
#-#C C takes the soap bottle","#C C turns on the faucet","#C C grasps a spice jar","#C C opens the blue container"
##C C grasps a spice jar","#C C takes the blue container","#C C removes the food from the cutting board","#C C places a container on the countertop" \
#  --suppress_person
#-v6_SAM_64.ckpt \ 'sink','plate','pasta','blue colander'\  'green cutting_board','food','pot'\