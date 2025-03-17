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

import cv2
from PIL import Image
import torchvision
from . import metrics, utils

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = torchvision.transforms.ToTensor()(x_)  # (3, H, W)
    return x_


def visualize_sample(
    ds,
    sample_id,
    t=None,
    visualise=True,
    gt_masked=None,
    model=None,
    mask_targ=None,
    save=False,
    pose=None,
):
    """
    Evaluate one sample of a dataset (ds). Calculate PSNR and mAP,
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
    img_wh = tuple(sample["img_wh"].numpy())
    img_gt = ds.x2im(sample["rgbs"], type_="pt")

    print('Extrinsics', sample["c2w"])

    results = model.render(sample, img_h, intrinsics_fy, ds.image_encoder, t=t)
    print(results.keys())
    print(results["rgb_fine"].shape)
    print(results["depth_fine"].shape)
    img_pred = ds.x2im(results["rgb_fine"][:, :3], type_="pt")
    depth_pred = ds.x2im(results["depth_fine"], type_="pt")
    depth_map = visualize_depth(depth_pred) #.permute(1, 2, 0)
    print(img_pred.shape)
    print(depth_pred.shape)
    print(ds.K, 'intrinsic matrix')
    print(depth_map.shape)

    W, H = img_wh
    fx, fy, cx, cy = ds.K[0, 0], ds.K[1, 1], ds.K[0, 2], ds.K[1, 2]

    # Crear una grilla de coordenadas 2D
    i, j = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    i, j = i.float(), j.float()

    # Asumiendo que depth_pred es el mapa de profundidad en el tamaño (128, 228)
    z = depth_pred.view(-1)
    x = (j.view(-1) - cx) * z / fx
    y = (i.view(-1) - cy) * z / fy

    # Crear puntos 3D
    points_3D = torch.stack([x, y, z], dim=1)
    points_color = img_pred.view(-1, 3)

    mask = z > 0
    points_3D = points_3D[mask]
    points_color = points_color[mask]


    """mask_stat = ds.x2im(results["_rgb_fine_static"][:, 3])
    mask_transient = ds.x2im(results["_rgb_fine_transient"][:, 4])
    mask_person = ds.x2im(results["_rgb_fine_person"][:, 5])
    mask_pred = mask_transient + mask_person

        
    beta = ds.x2im(results["beta"])
    img_pred_static = ds.x2im(results["rgb_fine_static"][:, :3], type_="pt")
    img_pred_transient = ds.x2im(results["_rgb_fine_transient"][:, :3])
    img_pred_person = ds.x2im(results["_rgb_fine_person"][:, :3])"""

    return points_3D, points_color, depth_map


def create_map(
    dataset,
    model,
    mask_loader,
    vis_i=5,
    save_dir="results/test",
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

    if image_ids is None:
        image_ids = dataset.img_ids_test

    for i, sample_id in utils.tqdm(enumerate(image_ids), total=len(image_ids)):

        do_visualise = i % vis_i == 0

        tqdm.tqdm.write(f"Test sample {i}. Frame {sample_id}.")


        if timestep_const is not None:
            timestep = sample_id
            sample_id = timestep_const
        else:
            timestep = sample_id
        points_3D, points_color, depth_map = visualize_sample(
                                                                dataset,
                                                                sample_id,
                                                                model=model,
                                                                t=timestep,
                                                                visualise=do_visualise,
                                                                gt_masked=None,
                                                                mask_targ=None,
                                                                save=save,
                                                            )
        if save:
            save_path = os.path.join(save_dir, f"{sample_id}_3d.npy")
            np.save(save_path, points_3D.cpu().numpy())
            save_path = os.path.join(save_dir, f"{sample_id}_color.npy")
            np.save(save_path, points_color.cpu().numpy())
            #Save the depth map
            save_path = os.path.join(save_dir, f"{sample_id}_depth.png")
            depth_image = torchvision.transforms.ToPILImage()(depth_map)
            depth_image.save(save_path)
            print(f"Saved 3D points to {save_path}")
        break


    print(f"avgpre: {results['metrics']['avgpre']}, PSNR: {results['metrics']['psnr']}")

    return results
