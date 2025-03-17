import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

# OwlViT Detection
#from transformers import OwlViTProcessor, OwlViTForObjectDetection
import sys
sys.path.append('/home/lmur/FUSION_FIELDS/OOAL')
from models.ooal import Net

from matplotlib import cm

import cv2
from . import metrics, utils
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = "jet", alpha: float = 0.7) -> Image.Image:
    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("img and mask arguments need to be PIL.Image")

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError("alpha argument is expected to be of type float between 0 and 1")

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img

def viz_pred_test(save_path, image, ego_pred, GT_mask, aff_list, aff_label, img_name, epoch=None):
    print(ego_pred.shape, image.size, '---------')
    """mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    img = image.squeeze(0) * std + mean
    img = img.detach().cpu().numpy() * 255
    img = Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8))"""
    img = image

    #gt = Image.fromarray(GT_mask)
    #gt_result = overlay_mask(img, gt, alpha=0.5)
    aff_str = aff_list[aff_label]

    #os.makedirs(os.path.join(save_path, 'viz_gray'), exist_ok=True)
    #gray_name = os.path.join(save_path, 'viz_gray', img_name + '.jpg')
    #cv2.imwrite(gray_name, ego_pred * 255)

    ego_pred = Image.fromarray(ego_pred)
    ego_pred = overlay_mask(img, ego_pred, alpha=0.5)

    fig, ax = plt.subplots(figsize=(10, 6))
    #for axi in ax.ravel():
    #    axi.set_axis_off()

    ax.imshow(ego_pred)
    ax.set_title(aff_str)
    #ax[1].imshow(gt_result)
    #ax[1].set_title('GT')

    os.makedirs(os.path.join(save_path, 'viz_test'), exist_ok=True)
    if epoch:
        fig_name = os.path.join(save_path, 'viz_test', "iter" + str(epoch) + '_' + img_name + '.jpg')
    else:
        fig_name = os.path.join(save_path, 'viz_test', img_name + '.jpg')
    print('saaaavng here', fig_name)
    plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.close()

def normalize_map(atten_map, img_size):
    print(atten_map.shape, img_size, 'maaaapa de atencionn')
    atten_map = cv2.resize(atten_map, dsize=(img_size[1], img_size[0]))
    min_val = np.min(atten_map)
    max_val = np.max(atten_map)
    atten_norm = (atten_map - min_val) / (max_val - min_val + 1e-10)
    #atten_norm[atten_norm > 0.5] = 1
    
    return atten_norm

def load_ool_AFF(text_queries):
    """
    Return: model, processor (for text inputs)
    """
    args = None
    ooal_model = Net(args, 768, 512, text_queries).cuda()

    model_file = '/home/lmur/FUSION_FIELDS/OOAL/seen_best'
    ooal_model.eval()
    assert os.path.exists(model_file), "Please provide the correct model file for testing"
    state_dict = torch.load(model_file)['model_state_dict']
    ooal_model.load_state_dict(state_dict, strict=False)
    print('mooooodel loaded')
    return ooal_model

def evaluate_ooal_AFF_sample(
    ooal_model,
    ds,
    mask_targets,
    sample_id,
    text_queries_dict,
    scene_metrics,
    save_dir,
    t=None,
    visualise=True,
    model=None,
    save=False,
    pose=None,
):
    """
    Evaluate one sample of a dataset (ds). Calculate PSNR and mAP in the object segmentation,
    and visualise different model components for this sample. Additionally,
    1) a different timestep (`t`) can be chosen, which can be different from the
    timestep of the sample (useful for rendering the same view over different
    timesteps).
    """

    if pose is None:
        sample = ds[sample_id]
    else:
        sample = ds.__getitem__(sample_id, pose)
    img_h = ds.img_h
    intrinsics_fy = ds.K[1][1]

    print(sample.keys())
     
    scene_id = save_dir.split("/")[-2]
    img_gt_path = os.path.join('/home/lmur/FUSION_FIELDS/Lorenzo_Feature_Fields_v2/data/EPIC-Diff', scene_id, 'images', sample["im_path"].replace("IMG", "frame").replace("bmp", "jpg"))
    results = model.render(sample, img_h, intrinsics_fy, ds.image_encoder, t=t)
    rendered_img_tensor = ds.x2im(results["rgb_fine"][:, :3], type_="pt")
    print(rendered_img_tensor.shape)
    #Save the rendered image
    #rendered_img = Image.fromarray((rendered_img_tensor * 255).cpu().numpy().astype(np.uint8))
    img_size = (128, 228)
    print(img_size)
    #Save
    #rendered_img.save(os.path.join(f"./{save_dir}/Rendered_{sample_id}.png"))
    
    rendered_img = Image.open(img_gt_path).convert('RGB')
    transform_img = transforms.Compose([
            transforms.Resize((224, 224), antialias=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))])
    rendered_img_torch = transform_img(rendered_img)

    with torch.no_grad():
        all_ego_pred = ooal_model(rendered_img_torch.unsqueeze(0).cuda(), gt_aff=None)

    for cls_ in range(all_ego_pred.shape[1]): 
        ego_pred = all_ego_pred[:, cls_]
        ego_pred = np.array(ego_pred.squeeze().data.cpu())
        ego_pred = normalize_map(ego_pred, img_size)
        
        text_query = text_queries_dict[cls_]
        print(text_query, 'teeeeeeeee')
        img_name = 'image_' + str(sample_id) + '_' + text_query
        save_path = '/home/lmur/FUSION_FIELDS/Lorenzo_Feature_Fields_v2/results/ooal_vis'
        #viz_pred_test(save_path, rendered_img, ego_pred, None, text_queries_dict, cls_, img_name)
        
        #Compute the metrics for the scene
        for i, t in enumerate(text_queries_dict):
            if t == text_query:
                text_query_ID = i
                break
        #text_query_ID = int(next((t for t in text_queries_dict if t == text_query), None)) #int(next((k for k, v in text_queries_dict.items() if v == text_query), None))
        mask_target_i = mask_targets[:, :, i]#mask_targets == text_query_ID
        pred_mask = ego_pred > 0.5
        print(pred_mask.shape, mask_target_i.shape, 'queeeee comparamos')
        mask_pred_i = pred_mask.astype(bool)
        print(mask_pred_i.dtype, mask_target_i.dtype)
        
        TP_i = (mask_target_i & mask_pred_i).sum().item()
        FP_i = (~mask_target_i & mask_pred_i).sum().item()
        FN_i = (mask_target_i & ~mask_pred_i).sum().item()
        
        scene_metrics[text_query]["TP"] += TP_i
        scene_metrics[text_query]["FP"] += FP_i
        scene_metrics[text_query]["FN"] += FN_i

    return scene_metrics

def evaluate_ooal_AFF_segmentation(
    dataset,
    model,
    mask_loader,
    save_dir,
    save=False,
    vid=None,
    epoch=None,
    timestep_const=None,
    image_ids=None,
):
    """
    Like `evaluate_sample`, but evaluates over all selected image_ids.
    Saves also visualisations and average scores of the selected samples.
    """
    # Load OOAL-AFF: https://openaccess.thecvf.com/content/CVPR2024/papers/Li_One-Shot_Open_Affordance_Learning_with_Foundation_Models_CVPR_2024_paper.pdf
    ooal_model = load_ool_AFF(mask_loader.labels_id)
    print("OOAL AFF model loaded successfully!!!")
    #Define the scene metrics, per object
    class_metrics = {"TP": 0, "FP": 0, "FN": 0, "IoU": 0}
    scene_objects = mask_loader.labels_id #.values()
    scene_metrics = {k: class_metrics.copy() for k in scene_objects if k != 'undefined'}

    for i, sample_id in utils.tqdm(enumerate(image_ids), total=len(image_ids)):
        tqdm.tqdm.write(f"Test sample {i}. Frame {sample_id}.")
        
        mask_targ, text_queries, sample_id = mask_loader[sample_id]
        # ignore evaluation if no mask available
        if mask_targ.sum() == 0:
            print(f"No annotations for frame {sample_id}, skipping.")
            continue
        
        if timestep_const is not None:
            timestep = sample_id
            sample_id = timestep_const
        else:
            timestep = sample_id

        scene_metrics = evaluate_ooal_AFF_sample(
            ooal_model,
            dataset,
            mask_targ,
            sample_id,
            text_queries,
            scene_metrics,
            save_dir,
            model=model,
            t=timestep,
            visualise=True,
            save=save,
        )
        print('SCENE METRICS')
        print(scene_metrics)
        print()


    mean_scene_metrics = {}
    for k, v in scene_metrics.items():
        if v["TP"] + v["FP"] + v["FN"] == 0:
            mean_scene_metrics[k] = 0
        else:
            mean_scene_metrics[k] = 100 * (v["TP"] / (v["TP"] + v["FP"] + v["FN"]))
    mean_scene_metrics["mIoU"] = np.mean(list(mean_scene_metrics.values()))
    print(f"Mean IoU per object: {mean_scene_metrics}")


    if save:
        with open(f"{save_dir}/metrics.txt", "a") as f:
            f.write(f"mIoU: {mean_scene_metrics['mIoU']:.2f}\n")
            for k, v in mean_scene_metrics.items():
                f.write(f"{k}: {v:.2f}\n")

    return mean_scene_metrics
