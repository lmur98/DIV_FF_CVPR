import os
os.chdir('../')
import torch
import extra_utils
from dataset import MaskLoader
import evaluation

vid = 'P01_01'
epoch = 9

from dataset import EPICDiff
dataset = EPICDiff(vid, split='test')
ckpt_path = f'ckpts/rel/{vid}/epoch={epoch}.ckpt'
models = utils.init_model(ckpt_path, dataset)
maskloader = MaskLoader(
    dataset=dataset,
    is_debug=True
)
results = evaluation.evaluate(
    dataset,
    models,
    maskloader,
    vis_i=1,
    save=True,
    save_dir='results/test',
    vid=vid,
    image_ids=dataset.img_ids_test[:5]
)