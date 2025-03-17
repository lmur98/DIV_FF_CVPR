"""
    Evaluate segmentation capacity of model via mAP,
    also includes renderings of segmentations and PSNR evaluation.
"""
import os
from collections import defaultdict

import git
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from sklearn.metrics import average_precision_score
from einops import rearrange

from . import metrics, utils
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def apply_colormap(image):
    image = image - torch.min(image)
    image = image / (torch.max(image) + 1e-6)
    image = image * 2.0 - 1.0
    image_long = (image * 255).long()
    image_long = torch.clip(image_long, 0, 255)
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return torch.tensor(matplotlib.colormaps["turbo"].colors, device=image.device)[image_long[..., 0]]    

def evaluate_object_segmentation_sample(
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
    
    list_text_queries = list(text_queries_dict.values())
    if list_text_queries is not None:
        ds.image_encoder.set_positives(list_text_queries)
    if list_text_queries is not None:
        ds.image_encoder.set_video_positives(list_text_queries)

    results = model.render(sample, img_h, intrinsics_fy, ds.image_encoder, t=t)

    object_segmentation_masks = {}
    all_pred_masks = torch.zeros((128, 228))
    max_relevancies = torch.zeros((128, 228))
    sample_IoU = []
    for i in range(len(ds.image_encoder.positives)):
        print('queeeeeeee pasa ')
        object_mask = (results[f"video_relevancy_{i}"] > 0.7).cpu().view(128, 228)
        object_relevancy = results[f"video_relevancy_{i}"].cpu().view(128, 228)
        text_query = ds.image_encoder.positives[i]
        
        object_segmentation_masks[text_query] = object_mask
        
        #Compute the metrics for the scene
        text_query_ID = int(next((k for k, v in text_queries_dict.items() if v == text_query), None))
        mask_target_i = mask_targets == text_query_ID
        mask_pred_i = object_mask

        update_mask = (object_relevancy > max_relevancies) & mask_pred_i
        all_pred_masks[update_mask] = text_query_ID
        max_relevancies[update_mask] = object_relevancy[update_mask]
        #all_pred_masks = np.where(mask_pred_i, text_query_ID, all_pred_masks)
        
        TP_i = (mask_target_i & mask_pred_i).sum().item()
        FP_i = (~mask_target_i & mask_pred_i).sum().item()
        FN_i = (mask_target_i & ~mask_pred_i).sum().item()
        IoU_i = TP_i / (TP_i + FP_i + FN_i)
        
        scene_metrics[text_query]["TP"] += TP_i
        scene_metrics[text_query]["FP"] += FP_i
        scene_metrics[text_query]["FN"] += FN_i
        sample_IoU.append(IoU_i)
    sample_IoU = np.mean(sample_IoU)

    max_n_classes = int(max(text_queries_dict.keys())) + 1
    visualise = True
    if visualise:
        figure, ax = plt.subplots(1, 3, figsize=(15, 5))
        figure.suptitle(f"Object segmentation of the img {sample_id} with an IoU of {sample_IoU:.2f}")
        
        
        ax[0].set_title("GT")
        ax[0].imshow(mask_targets)
        #ax[0].axis("off")
        ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax[0].spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

        ax[1].set_title("Object Segmentation")
        ax[1].imshow(all_pred_masks)
        ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax[1].spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

        ax[2].set_title("Image")
        ax[2].imshow(ds.x2im(sample["rgbs"], type_="pt"))
        ax[2].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax[2].spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(2)  

        cmap = plt.cm.get_cmap("viridis", max_n_classes)
        legend_elements = [
            Patch(facecolor=cmap(int(i)), edgecolor=cmap(i), label=f"{text}")
            for i, text in enumerate(list_text_queries)
        ]
        figure.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize='small', bbox_to_anchor=(0.5, 0.05))
        plt.subplots_adjust(bottom=0.2)
        
        
        #cmap = plt.cm.get_cmap("viridis", max_n_classes)
        #legend_elements = [Patch(facecolor=cmap(int(k)), edgecolor=cmap(int(k)), label=f"{v}") for k, v in text_queries_dict.items()]
        #plt.legend(handles=legend_elements, loc = 'lower center')

        
        plt.savefig(f"{save_dir}/Obj_Segm_{sample_id}.png")
        print(f"Saved segmentation visualisation for {text_query} at {save_dir}/Obj_Segm_{sample_id}.png")
        plt.close(figure)

    return scene_metrics

def evaluate_object_segmentation(
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

    #Define the scene metrics, per object
    class_metrics = {"TP": 0, "FP": 0, "FN": 0, "IoU": 0}
    scene_objects = mask_loader.labels_id.values()
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

        scene_metrics = evaluate_object_segmentation_sample(
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
