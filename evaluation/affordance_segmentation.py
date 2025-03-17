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


def evaluate_affordance_segmentation_sample(
    ds,
    mask_targets,
    sample_id,
    list_text_queries,
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
    
    if list_text_queries is not None:
        ds.image_encoder.set_positives(list_text_queries)
        ds.image_encoder.set_video_positives(list_text_queries)

    results = model.render(sample, img_h, intrinsics_fy, ds.image_encoder, t=t)

    action_segmentation_masks = {}
    all_pred_masks = torch.zeros((128, 228))
    max_relevancies = torch.zeros((128, 228))
    sample_IoU = []
    for i in range(len(ds.image_encoder.positives)):
        print('queeeeeee')
        action_mask = (results[f"video_relevancy_{i}"] > 0.54).cpu().view(128, 228).numpy()
        action_relevancy = results[f"video_relevancy_{i}"].cpu().view(128, 228)
        text_query = ds.image_encoder.positives[i]
        action_segmentation_masks[text_query] = action_mask
        
        #Compute the metrics for the scene
        mask_target_i = mask_targets[:, :, i]
        if np.sum(mask_target_i) == 0:
            #We skip the evaluation of the affordance if there are no annotations
            continue

        update_mask = (action_relevancy > max_relevancies) & action_mask
        all_pred_masks[update_mask] = i + 1 #We keep 0 for the background
        max_relevancies[update_mask] = action_relevancy[update_mask]
        #all_pred_masks = np.where(mask_pred_i, text_query_ID, all_pred_masks)
        
        TP_i = (mask_target_i & action_mask).sum().item()
        FP_i = (~mask_target_i & action_mask).sum().item()
        FN_i = (mask_target_i & ~action_mask).sum().item()
        IoU_i = TP_i / (TP_i + FP_i + FN_i)
        
        scene_metrics[text_query]["TP"] += TP_i
        scene_metrics[text_query]["FP"] += FP_i
        scene_metrics[text_query]["FN"] += FN_i
        sample_IoU.append(IoU_i)
    sample_IoU = np.mean(sample_IoU)

    
    visualise = True
    
    custom_colors_rgb = {
    0: (255, 255, 255),  # Blanco para el fondo
    1: (0, 63, 92),      # Azul oscuro
    2: (88, 80, 141),    # Púrpura oscuro
    3: (188, 80, 144),   # Rosa fuerte
    4: (255, 110, 84),   # Naranja intenso
    5: (255, 166, 0),    # Amarillo brillante
    6: (55, 76, 128),    # Azul grisáceo
    7: (106, 76, 147),   # Púrpura real
    8: (212, 80, 135),   # Rosa oscuro
    9: (249, 93, 106),   # Rojo coral
    10: (255, 166, 121)  # Melocotón claro
    }

    base_img = ds.x2im(sample["rgbs"], type_="pt").cpu().numpy()
    print(base_img.shape)
    mask_targets_show = base_img #np.ones((128, 228, 3), dtype=np.uint8) * 255 #White background
    mask_pred_show = np.ones((128, 228, 3), dtype=np.uint8) * 255 #White background
    for i in range(len(list_text_queries)):
        mask_targets_show[mask_targets[:, :, i] > 0] = custom_colors_rgb[i + 1]
        mask_pred_show[all_pred_masks == i + 1] = custom_colors_rgb[i + 1]

    if visualise:
        figure, ax = plt.subplots(1, 3, figsize=(15, 5))
        figure.suptitle(f"Affordances segmentation of the img {sample_id} with an IoU of {sample_IoU:.2f}")
        
        ax[0].set_title("GT")
        ax[0].imshow(mask_targets_show)
        #ax[0].axis("off")
        ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax[0].spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

        ax[1].set_title("Pred")
        ax[1].imshow(mask_pred_show)
        #ax[1].axis("off")
        ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax[1].spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

        ax[2].set_title("Image")
        ax[2].imshow(ds.x2im(sample["rgbs"], type_="pt"))
        #ax[2].axis("off")
        ax[2].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax[2].spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(2)                

        def normalize_rgb(rgb_tuple):
            return tuple(c / 255.0 for c in rgb_tuple)

        # Crear los elementos de la leyenda con colores normalizados
        legend_elements = [
            Patch(facecolor=normalize_rgb(custom_colors_rgb[i + 1]), edgecolor=normalize_rgb(custom_colors_rgb[i + 1]), label=f"{text}")
            for i, text in enumerate(list_text_queries)
        ]
        figure.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize='small', bbox_to_anchor=(0.5, 0.05))
        plt.subplots_adjust(bottom=0.2)

        plt.savefig(f"{save_dir}/Obj_Segm_{sample_id}.png")
        print(f"Saved segmentation visualisation for {text_query} at {save_dir}/Obj_Segm_{sample_id}.png")
        plt.close(figure)

    return scene_metrics

def evaluate_affordance_segmentation(
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
    scene_actions = mask_loader.labels_id
    scene_metrics = {k: class_metrics.copy() for k in scene_actions if k != 'undefined'}

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

        scene_metrics = evaluate_affordance_segmentation_sample(
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
    print(f"Mean IoU per affordance: {mean_scene_metrics}")


    if save:
        with open(f"{save_dir}/metrics.txt", "a") as f:
            f.write(f"mIoU: {mean_scene_metrics['mIoU']:.2f}\n")
            for k, v in mean_scene_metrics.items():
                f.write(f"{k}: {v:.2f}\n")

    return mean_scene_metrics