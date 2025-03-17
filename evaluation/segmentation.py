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
from sklearn.decomposition import PCA
from einops import rearrange

from . import metrics, utils
from sklearn.preprocessing import MinMaxScaler
import time


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

def evaluate_sample(
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
    time_1 = time.time()
    results = model.render(sample, img_h, intrinsics_fy, ds.image_encoder, t=t)
    time_2 = time.time()
    print('Time to render:', time_2 - time_1)
    figure = None
    print(results.keys())
    output_clip = "raw_relevancy" in results
    output_PCA = "video" in results
    if output_PCA:
        clip_map = results["video"].detach().cpu().numpy()
        pca = PCA(n_components=3)
        scaler = MinMaxScaler()
        data_reduced = pca.fit_transform(clip_map)
        print(data_reduced.shape, np.max(data_reduced), np.min(data_reduced))
        data_reduced = scaler.fit_transform(data_reduced)
        feat_map = ds.x2im(torch.tensor(data_reduced), type_="pt") 
        feat_map = feat_map
        print(feat_map.shape)
        print(data_reduced.shape)
        print(clip_map.shape)
    
    if output_clip:
        relevancy_maps = {}
        for i in range(len(ds.image_encoder.positives)):
            img_relevancy_i = ds.x2im(results[f"relevancy_{i}"], type_="pt")
            #p_i = img_relevancy_i
            p_i = torch.clip(img_relevancy_i - 0.50, 0, 1)
            p_i = rearrange(p_i, " n1 n2 -> n1 n2 1") 
            show_color = True
            if show_color:
                results[f"composited_{i}"] = apply_colormap(p_i / (p_i.max() + 1e-6)).cpu()
                mask = (results[f"relevancy_{i}"] < 0.53).cpu().view(128, 228) #TRICK PUT 0.55 to filter
                color = ds.x2im(results["rgb_fine"][:, :3], type_="pt")

                # Aclarar el color donde la máscara es verdadera
                alpha = 0.5  # Factor de transparencia, ajusta según sea necesario
                lightened_color = color * (1 - alpha) + torch.ones_like(color) * alpha  # Aclara el color
                print('thisss')
                results[f"composited_{i}"][mask] = lightened_color[mask]

                #results[f"composited_{i}"][mask] = color[mask]
            else:
                results[f"composited_{i}"] = apply_colormap(p_i / (p_i.max() + 1e-6)).cpu()
            # mask = (results["relevancy_0"] < 0.5).squeeze()
            # results[f"composited_{i}"][mask, :] = results["rgb"][mask, :]
            # name = f"relevancy_{ds.image_encoder.positives[i]}_{sample_id}.png"
            # plt.imshow(results[f"composited_{i}"])
            # plt.savefig(name) 
            relevancy_maps[f"composited_{i}"] = results[f"composited_{i}"]
            
    output_video = "raw_video_relevancy" in results
    if output_video:
        video_relevancy_maps = {}
        for i in range(len(ds.image_encoder.positives)):
            img_relevancy_i = ds.x2im(results[f"video_relevancy_{i}"], type_="pt")
            #p_i = img_relevancy_i
            p_i = torch.clip(img_relevancy_i - 0.51, 0, 1)
            p_i = rearrange(p_i, " n1 n2 -> n1 n2 1")
            show_color = False
            if show_color:
                results[f"video_composited_{i}"] = apply_colormap(p_i / (p_i.max() + 1e-6)).cpu()
                mask = (results[f"video_relevancy_{i}"] < 0.57).cpu().view(128, 228) #TRICK PUT 0.55 to filter
                color = ds.x2im(results["rgb_fine"][:, :3], type_="pt")
                
                alpha = 0.5  # Factor de transparencia, ajusta según sea necesario
                lightened_color = color * (1 - alpha) + torch.ones_like(color) * alpha  # Aclara el color
                results[f"video_composited_{i}"][mask] = lightened_color[mask]
            else:
                results[f"video_composited_{i}"] = apply_colormap(p_i / (p_i.max() + 1e-6)).cpu()
            video_relevancy_maps[f"video_composited_{i}"] = results[f"video_composited_{i}"]

    output_person = "person_weights_sum" in results
    output_transient = "_rgb_fine_transient" in results

    img_wh = tuple(sample["img_wh"].numpy())
    img_gt = ds.x2im(sample["rgbs"], type_="pt")

    img_pred = ds.x2im(results["rgb_fine"][:, :3], type_="pt")

    mask_stat = ds.x2im(results["_rgb_fine_static"][:, 3])
    if output_transient:
        mask_transient = ds.x2im(results["_rgb_fine_transient"][:, 4])
        mask_pred = mask_transient
        
        if output_person:
            mask_person = ds.x2im(results["_rgb_fine_person"][:, 5])
            mask_pred = mask_pred + mask_person
        else:
            mask_person = np.zeros_like(mask_transient)
        
    beta = ds.x2im(results["beta"])
    img_pred_static = ds.x2im(results["rgb_fine_static"][:, :3], type_="pt")
    img_pred_transient = ds.x2im(results["_rgb_fine_transient"][:, :3])
    if output_person:
        img_pred_person = ds.x2im(results["_rgb_fine_person"][:, :3])

    if mask_targ is not None:
        average_precision = average_precision_score(
            mask_targ.reshape(-1), mask_pred.reshape(-1)
        )

    psnr = metrics.psnr(img_pred, img_gt).item()
    psnr_static = metrics.psnr(img_pred_static, img_gt).item()

    if visualise:

        figure, ax = plt.subplots(figsize=(8, 5))
        figure.suptitle(f"Sample: {sample_id}.\n")
        plt.tight_layout()
        plt.subplot(331)
        plt.title("GT")
        if gt_masked is not None:
            plt.imshow(torch.from_numpy(gt_masked))
        else:
            plt.imshow(img_gt)
        plt.axis("off")
        plt.subplot(332)
        plt.title(f"Pred. PSNR: {psnr:.2f}")
        plt.imshow(img_pred.clamp(0, 1))
        plt.axis("off")
        plt.subplot(333)
        plt.axis("off")

        plt.subplot(334)
        plt.title(f"Static. PSNR: {psnr_static:.2f}")
        plt.imshow(img_pred_static)
        plt.axis("off")
        plt.subplot(335)
        plt.title(f"Transient")
        plt.imshow(img_pred_transient)
        plt.axis("off")
        if "_rgb_fine_person" in results:
            plt.subplot(336)
            plt.title("Person")
            plt.axis("off")
            plt.imshow(img_pred_person)
        else:
            plt.subplot(336)
            plt.axis("off")

        plt.subplot(337)
        if mask_targ is not None:
            plt.title(f"Mask. AP: {average_precision:.4f}")
        else:
            plt.title("Mask.")
        plt.imshow(mask_pred)
        plt.axis("off")
        plt.subplot(338)
        plt.title(f"Mask: Transient.")
        plt.imshow(mask_transient)
        plt.axis("off")
        plt.subplot(339)
        plt.title(f"Mask: Person.")
        plt.imshow(mask_person)
        plt.axis("off")

    if visualise and not save:
        plt.show()

    results = {}
    results["figure"] = utils.plt_to_im(figure)
    plt.close()

    if output_clip:
        figure_clip, ax = plt.subplots(figsize=(8, 9), dpi = 300)
        figure_clip.suptitle(f"CLIP Sample: {sample_id}.\n")
        plt.tight_layout()
        plt.subplot(321)
        plt.title("GT")
        plt.imshow(img_gt)
        plt.axis("off")
        plt.subplot(322)
        plt.title("Prediction")
        plt.imshow(feat_map)
        plt.axis("off")
        for i in range(len(ds.image_encoder.positives)):
            plt.subplot(3,2,i+3)
            plt.title(ds.image_encoder.positives[i])
            plt.imshow(relevancy_maps[f"composited_{i}"].clamp(0, 1))
            plt.axis("off")
        results["figure_clip"] = utils.plt_to_im(figure_clip)
        plt.tight_layout()
        plt.close()
        
    if output_video:
        figure_clip, ax = plt.subplots(figsize=(8, 9))
        figure_clip.suptitle(f"VIDEO Sample: {sample_id}.\n")
        plt.tight_layout()
        plt.subplot(321)
        plt.title("GT")
        plt.imshow(img_gt)
        plt.axis("off")
        plt.subplot(322)
        plt.title("Prediction")
        plt.imshow(img_pred.clamp(0, 1))
        plt.axis("off")
        for i in range(len(ds.image_encoder.positives)):
            plt.subplot(3,2,i+3)
            plt.title(ds.image_encoder.positives[i])
            plt.imshow(video_relevancy_maps[f"video_composited_{i}"].clamp(0, 1))
            plt.axis("off")
        results["figure_video_clip"] = utils.plt_to_im(figure_clip)
        
    
    results["im_tran"] = img_pred_transient
    results["im_stat"] = img_pred_static
    results["im_pred"] = img_pred
    results["im_targ"] = img_gt
    results["psnr"] = psnr
    results["mask_pred"] = mask_pred
    results["mask_stat"] = mask_stat
    if output_person:
        results["mask_pers"] = mask_person
        results["im_pers"] = img_pred_person
    results["mask_tran"] = mask_transient
    if mask_targ is not None:
        results["average_precision"] = average_precision
    if output_clip:
        for i in range(len(relevancy_maps)):
            results[f"composited_{i}"] = relevancy_maps[f"composited_{i}"]
    if output_video:
        for i in range(len(relevancy_maps)):
            results[f"video_composited_{i}"] = video_relevancy_maps[f"video_composited_{i}"]

    for k in results:
        if k == "figure" or k == "figure_clip":
            continue
        if type(results[k]) == torch.Tensor:
            results[k] = results[k].to("cpu")

    return results


def evaluate(
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

    results = {
        k: []
        for k in [
            "avgpre",
            "psnr",
            "masks",
            "out",
            "hp",
        ]
    }

    if image_ids is None:
        image_ids = dataset.img_ids_test

    for i, sample_id in utils.tqdm(enumerate(image_ids), total=len(image_ids)):

        do_visualise = i % vis_i == 0

        tqdm.tqdm.write(f"Test sample {i}. Frame {sample_id}.")

        mask_targ, im_masked = mask_loader[sample_id]
        # ignore evaluation if no mask available
        if mask_targ.sum() == 0:
            print(f"No annotations for frame {sample_id}, skipping.")
            continue

        #results["hp"] = model.hparams
        #results["hp"]["git_eval"] = git.Repo(
        #    search_parent_directories=True
        #).head.object.hexsha

        if timestep_const is not None:
            timestep = sample_id
            sample_id = timestep_const
        else:
            timestep = sample_id
        out = evaluate_sample(
            dataset,
            sample_id,
            model=model,
            t=timestep,
            visualise=do_visualise,
            gt_masked=im_masked,
            mask_targ=mask_targ,
            save=save,
        )

        if save and do_visualise:
            results_im = out["figure"]
            os.makedirs(f"{save_dir}/per_sample", exist_ok=True)
            path = f"{save_dir}/per_sample/{sample_id}.png"
            plt.imsave(path, results_im)

            if ("figure_clip" in out.keys()):
                results_im_clip = out["figure_clip"]
                os.makedirs(f"{save_dir}/per_sample", exist_ok=True)
                path = f"{save_dir}/per_sample/{sample_id}_Img_{dataset.image_encoder.positives[0]}_{dataset.image_encoder.positives[1]}_clip.png"
                plt.imsave(path, results_im_clip)
            if ("figure_video_clip" in out.keys()):
                results_im_clip = out["figure_video_clip"]
                path = f"{save_dir}/per_sample/{sample_id}_Vid_{dataset.image_encoder.positives[0]}_{dataset.image_encoder.positives[1]}_video_clip.png"
                plt.imsave(path, results_im_clip)

        mask_pred = out["mask_pred"]

        results["avgpre"].append(out["average_precision"])

        results["psnr"].append(out["psnr"])
        results["masks"].append([mask_targ, mask_pred])
        results["out"].append(out)

    metrics_ = {
        "avgpre": {},
        "psnr": {},
    }
    for metric in metrics_:
        metrics_[metric] = np.array(
            [x for x in results[metric] if not np.isnan(x)]
        ).mean()

    results["metrics"] = metrics_

    if save:
        with open(f"{save_dir}/metrics.txt", "a") as f:
            lines = utils.write_summary(results)
            f.writelines(f"Epoch: {epoch}.\n")
            f.writelines(lines)

    print(f"avgpre: {results['metrics']['avgpre']}, PSNR: {results['metrics']['psnr']}")

    return results
