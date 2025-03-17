import argparse
import os

import numpy as np

import evaluation
import extra_utils
from dataset import SAMPLE_IDS, VIDEO_IDS, EPICDiff, MaskLoader, Object_MaskLoader, Affordances_MaskLoader

import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont



def parse_args(path=None, vid=None, exp=None):

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default=path, help="Path to model.")

    parser.add_argument("--vid", type=str, default=vid, help="Video ID of dataset.")

    parser.add_argument("--exp", type=str, default=exp, help="Experiment name.")

    parser.add_argument(
        "--outputs",
        default=["masks"],
        type=str,
        nargs="+",
        help="Evaluation output. Select `masks` or `summary` or both.",
    )

    parser.add_argument(
        "--masks_n_samples",
        type=int,
        default=0,
        help="Select number of samples for evaluation. If kept at 0, then all test samples are evaluated.",
    )

    parser.add_argument(
        "--summary_n_samples",
        type=int,
        default=0,
        help="Number of samples to evaluate for summary video. If 0 is selected, then the video is rendered with all frames from the dataset.",
    )

    parser.add_argument(
        "--root_data", type=str, default="data/EPIC-Diff", help="Root of the dataset."
    )

    parser.add_argument(
        "--suppress_person",
        default=False,
        action="store_true",
        help="Disables person, e.g. for visualising complete foreground without parts missing where person occludes the foreground.",
    )

    # for opt.py
    parser.add_argument("--is_eval_script", default=True, action="store_true")

    parser.add_argument(
        "--use_clip",
        default=False,
        action="store_true",
        help="Active features head",
    )
    
    parser.add_argument(
        "--video_active", 
        default=False,
        action="store_true",
        help="Use egovideo head model",
    )

    parser.add_argument(
        "--fixed_pointview",
        default=False,
        action="store_true",
        help="Whether to render a video with a fixed pointview in CLIP render",
    )

    # parser.add_argument('--positive_queries', action='append', type=str)
    parser.add_argument('--positive_queries', help='delimited list input', 
        type=lambda s: [str(item) for item in s.split(',')])
    args = parser.parse_args()

    return args


def init(args):
    dataset = EPICDiff(args.vid, root=args.root_data)

    model = extra_utils.init_model(args.path, dataset)

    # update parameters of loaded models
    model.hparams["suppress_person"] = args.suppress_person
    model.hparams["inference"] = True
    
    return model, dataset


def eval_masks(args, model, dataset, root):
    """Evaluate masks to produce mAP (and PSNR) scores."""
    root = os.path.join(root, "masks")
    if not os.path.exists(root):
        os.makedirs(root)

    maskloader = MaskLoader(dataset=dataset)

    image_ids = evaluation.utils.sample_linear(
        dataset.img_ids_test, args.masks_n_samples
    )[0]

    results = evaluation.evaluate(
        dataset,
        model,
        maskloader,
        vis_i=1,
        save_dir=root,
        save=True,
        vid=args.vid,
        image_ids=image_ids,
    )
    
def eval_object_masks(args, model, dataset, root):
    """Evaluate masks to produce mAP (and PSNR) scores."""
    root = os.path.join(root, "object_masks")
    if not os.path.exists(root):
        os.makedirs(root)

    maskloader = Object_MaskLoader(dataset=dataset)
    
    image_ids = maskloader.fnames

    mean_scene_metrics = evaluation.evaluate_object_segmentation(
        dataset,
        model,
        maskloader,
        save_dir=root,
        save=True,
        vid=args.vid,
        image_ids=image_ids,
    )
    
def eval_action_locations(args, model, dataset, root):
    "Evaluate action locations"
    root = os.path.join(root, "action_locations")
    if not os.path.exists(root):
        os.makedirs(root)
        
    target_loader = Affordances_MaskLoader(dataset=dataset)
    
    image_ids = target_loader.fnames
    
    mean_scene_metrics = evaluation.evaluate_affordance_segmentation(
        dataset,
        model,
        target_loader,
        save_dir=root,
        save=True,
        vid=args.vid,
        image_ids=image_ids,
    )

def eval_OOL_AFF(args, model, dataset, root):
    "Evaluate action locations"
    root = os.path.join(root, "OOL_AFF")
    if not os.path.exists(root):
        os.makedirs(root)
        
    target_loader = Affordances_MaskLoader(dataset=dataset)
    
    image_ids = target_loader.fnames
    
    mean_scene_metrics = evaluation.evaluate_ooal_AFF_segmentation(
        dataset,
        model,
        target_loader,
        save_dir=root,
        save=True,
        vid=args.vid,
        image_ids=image_ids,
    )


def eval_masks_average(args):
    """Calculate average of `eval_masks` results for all 10 scenes."""
    scores = []
    for vid in VIDEO_IDS:
        path_metrics = os.path.join("results", args.exp, vid, 'masks', 'metrics.txt')
        with open(f'results/rel/{vid}/masks/metrics.txt') as f:
            lines = f.readlines()
            score_map, score_psnr = [float(s) for s in lines[2].split('\t')[:2]]
            scores.append([score_map, score_psnr])
    scores = np.array(scores).mean(axis=0)
    print('Average for all 10 scenes:')
    print(f'mAP: {(scores[0]*100).round(2)}, PSNR: {scores[1].round(2)}')


def render_video(args, model, dataset, root, save_cache=False):
    """Render a summary video like shown on the project page."""
    root = os.path.join(root, "summary")
    if not os.path.exists(root):
        os.makedirs(root)

    sid = SAMPLE_IDS[args.vid]

    top = evaluation.video.render(
        dataset, model, n_images=args.summary_n_samples
    )
    bot = evaluation.video.render(
        dataset, model, sid, n_images=args.summary_n_samples
    )
    
    if save_cache:
        evaluation.video.save_to_cache(
            args.vid, sid, root=root, top=top, bot=bot
        )

    ims_cat = [
        evaluation.video.convert_rgb(
            evaluation.video.cat_sample(top[k], bot[k])
        )
        for k in bot.keys()
    ]

    extra_utils.write_mp4(f"{root}/cat-{sid}-N{len(ims_cat)}", ims_cat, fps=7)

def render_3D_map(args, model, dataset, root):
    """Render a 3D map with the predicted masks."""
    root = os.path.join(root, "3D_map")
    if not os.path.exists(root):
        os.makedirs(root)

    maskloader = None
    image_ids = evaluation.utils.sample_linear(
        dataset.img_ids_test, args.masks_n_samples
    )[0]

    evaluation.create_map(
        dataset,
        model,
        maskloader,
        save_dir=root,
        save=True,
        vid=args.vid,
        image_ids=image_ids,
    )


def render_clip_video(args, model, dataset, root):
    """Render a summary video with different clip maps."""
    root = os.path.join(root, "clip_video")
    print('EEEEE', dataset.image_encoder.positives)
    if not os.path.exists(root):
        os.makedirs(root)

    if (args.positive_queries is None):
        print("Warning: At least one query is needed to render a CLIP video (check --positive_queries arg). Quiting...")
        return

    sid = SAMPLE_IDS[args.vid]

    top = evaluation.video.render(
        dataset, model, n_images=args.summary_n_samples
    )
    keys_to_visualize = ["im_targ", "im_pred"]
    keys_to_text = ["GT", "Render img"]
    for i in range(len(dataset.image_encoder.positives)):
        keys_to_visualize.append(f"composited_{i}")
        keys_to_text.append(dataset.image_encoder.positives[i])
    print('COOOON PUNTO FIJOOO')
    
    
    bot = evaluation.video.render(
            dataset, model, sid, n_images=args.summary_n_samples
        )
    ims_cat = [
            evaluation.video.convert_rgb(
                evaluation.video.cat_sample(top[k], bot[k], keys=keys_to_visualize, keys_text = keys_to_text)
            )
            for k in bot.keys()
        ]

    #ims_cat = [
    #        evaluation.video.convert_rgb(
    #            evaluation.video.cat_sample(top[k], bot=None, keys=keys_to_visualize)
    #        )
    #        for k in top.keys()
    #]

    extra_utils.write_mp4(f"{root}/cat-{sid}-N{len(ims_cat)}-queries-{dataset.image_encoder.positives}", ims_cat, fps=8)


    
def load_owlvit(checkpoint_path="owlvit-large-patch14", device='cpu'):
    """
    Return: model, processor (for text inputs)
    """
    processor = OwlViTProcessor.from_pretrained(f"google/{checkpoint_path}")
    model = OwlViTForObjectDetection.from_pretrained(f"google/{checkpoint_path}")
    model.to(device)
    model.eval()
    
    return model, processor

def eval_open_voc(args, model, dataset, root):
    """Evaluate masks to produce mAP (and PSNR) scores."""
    root = os.path.join(root, "open_voc_segmentation")
    if not os.path.exists(root):
        os.makedirs(root)

    #maskloader = Object_MaskLoader(dataset=dataset)
    maskloader = Affordances_MaskLoader(dataset=dataset)
    
    image_ids = maskloader.fnames
    

    mean_scene_metrics = evaluation.evaluate_open_voc_segmentation(
        dataset,
        model,
        maskloader,
        save_dir=root,
        save=True,
        vid=args.vid,
        image_ids=image_ids,
    )

def run(args, model, dataset, root):

    if "masks" in args.outputs:
        if (args.use_clip):
            dataset.image_encoder.set_positives(args.positive_queries)
        if (args.video_active):
            dataset.image_encoder.set_video_positives(args.positive_queries)
        # segmentations and renderings with mAP and PSNR
        eval_masks(args, model, dataset, root)
    
    if "3D_map" in args.outputs:
        # 3D maps
        if (args.use_clip):
            dataset.image_encoder.set_positives(args.positive_queries)
        if (args.video_active):
            dataset.image_encoder.set_video_positives(args.positive_queries)
        render_3D_map(args, model, dataset, root)
        
    if "open_voc" in args.outputs:
        # open vocabulary
        if (args.use_clip):
            dataset.image_encoder.set_positives(args.positive_queries)
        if (args.video_active):
            dataset.image_encoder.set_video_positives(args.positive_queries)
        eval_open_voc(args, model, dataset, root)

    if "summary" in args.outputs:
        # summary video
        render_video(args, model, dataset, root)
        
    if "object_segmentation" in args.outputs:
        eval_object_masks(args, model, dataset, root)

    if "action_segmentation" in args.outputs:
        eval_action_locations(args, model, dataset, root)

    if "predict_OOL_AFF" in args.outputs:
        if (args.use_clip):
            dataset.image_encoder.set_positives(args.positive_queries)
        if (args.video_active):
            dataset.image_encoder.set_video_positives(args.positive_queries)
        eval_OOL_AFF(args, model, dataset, root)

    if "clip_video" in args.outputs:
        print('Rendering clip video...')
        dataset.image_encoder.set_positives(args.positive_queries)
        if (args.video_active):
            dataset.image_encoder.set_video_positives(args.positive_queries)
        render_clip_video(args, model, dataset, root)

if __name__ == "__main__":
    args = parse_args()
    if 'average' in args.outputs:
        # calculate average over all 10 scenes for specific experiment
        eval_masks_average(args)
    else:
        model, dataset = init(args)
        root = os.path.join("results", args.exp, args.vid)
        run(args, model, dataset, root)
