import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# OwlViT Detection
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt

import gc

from . import metrics, utils
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import tqdm
import torch.nn.functional as F

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

def load_owlvit(checkpoint_path="owlvit-large-patch14", device='cpu'):
    """
    Return: model, processor (for text inputs)
    """
    processor = OwlViTProcessor.from_pretrained(f"google/{checkpoint_path}")
    model = OwlViTForObjectDetection.from_pretrained(f"google/{checkpoint_path}")
    model.to(device)
    model.eval()
    
    return model, processor

def evaluate_open_voc_segmentation_sample(
    open_voc_segmenter,
    open_voc_processor,
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
    #rendered_img_tensor = F.interpolate(rendered_img_tensor.unsqueeze(0).permute(0, 3, 1, 2), scale_factor=scale, mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
    print(rendered_img_tensor.shape)
    #Save the rendered image
    rendered_img = Image.fromarray((rendered_img_tensor * 255).cpu().numpy().astype(np.uint8))
    #Save
    rendered_img.save(os.path.join(f"./{save_dir}/Rendered_{sample_id}.png"))
    
    # rendered_img = Image.open(img_gt_path)
    list_text_queries = [text_queries_dict]#[list(text_queries_dict.values())]
    print(list_text_queries)
    with torch.no_grad():
        inputs = open_voc_processor(text=list_text_queries, images=rendered_img, return_tensors="pt").to(model.device)
        outputs = open_voc_segmenter(**inputs)
        
    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([rendered_img.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = open_voc_processor.post_process_object_detection(outputs=outputs, threshold=0.0, target_sizes=target_sizes.to(model.device))
    scores = torch.sigmoid(outputs.logits)
    topk_scores, topk_idxs = torch.topk(scores, k=1, dim=1)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    topk_idxs = topk_idxs.squeeze(1).tolist()
    topk_boxes = results[i]['boxes'][topk_idxs]
    topk_scores = topk_scores.view(len(list_text_queries), -1)
    topk_labels = results[i]["labels"][topk_idxs]
    boxes, scores, labels = topk_boxes, topk_scores.squeeze(0), topk_labels
    
    # Print detected objects and rescaled box coordinates
    predictions = []
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {list_text_queries[0][label]} with confidence {score} at location {box}")
        predictions.append({"box": box, "score": score, "label": list_text_queries[0][label]})
        
    boxes = boxes.cpu().detach().numpy()
    normalized_boxes = copy.deepcopy(boxes)
    
    # # visualize pred
    size = rendered_img.size
    pred_dict = {
        "boxes": normalized_boxes,
        "size": [size[1], size[0]], # H, W
        "labels": [list_text_queries[0][idx] for idx in labels]
    }
    image_pil = rendered_img
    image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    image_with_box.save(os.path.join(f"./{save_dir}/OpenVoc_Segm_{sample_id}.png"))
    
    predictor = SamPredictor(build_sam(checkpoint="./sam_vit_h_4b8939.pth"))
    rendered_img_path = os.path.join(f"./{save_dir}/Rendered_{sample_id}.png")
    image = cv2.imread(rendered_img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    
    for i in range(boxes.shape[0]):
        boxes[i] = torch.Tensor(boxes[i])

    boxes = torch.tensor(boxes, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes,
        multimask_output = False,
    )
    for i in range(len(masks)):
        predictions[i]["mask"] = masks[i]
        print(predictions[i]["label"], predictions[i]["box"], torch.unique(masks[i], return_counts=True), masks[i].shape)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box in boxes:
        show_box(box.numpy(), plt.gca())
    plt.axis('off')
    plt.savefig(os.path.join(f"./{save_dir}/OpenVoc_Segm_{sample_id}_SAM.png"))
    

    for i in range(len(predictions)):
        pred_box = predictions[i]["box"]
        pred_mask = np.zeros((128, 228))
        #pred_mask[int(pred_box[1]):int(pred_box[3]), int(pred_box[0]):int(pred_box[2])] = 1
        pred_mask = predictions[i]["mask"].cpu().numpy()
        text_query = predictions[i]["label"]
        
        #Compute the metrics for the scene
        for i, t in enumerate(text_queries_dict):
            if t == text_query:
                text_query_ID = i
                break
        #text_query_ID = int(next((t for t in text_queries_dict if t == text_query), None)) #int(next((k for k, v in text_queries_dict.items() if v == text_query), None))
        mask_target_i = mask_targets[:, :, i]#mask_targets == text_query_ID
        mask_pred_i = pred_mask.astype(bool)
        print(mask_pred_i.dtype, mask_target_i.dtype)
        
        TP_i = (mask_target_i & mask_pred_i).sum().item()
        FP_i = (~mask_target_i & mask_pred_i).sum().item()
        FN_i = (mask_target_i & ~mask_pred_i).sum().item()
        
        scene_metrics[text_query]["TP"] += TP_i
        scene_metrics[text_query]["FP"] += FP_i
        scene_metrics[text_query]["FN"] += FN_i

    return scene_metrics

def evaluate_open_voc_segmentation(
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
    # load OWL-ViT model
    open_voc_segmenter, open_voc_processor = load_owlvit(checkpoint_path="owlvit-base-patch32", device="cuda:0")
    print("OWL-ViT model loaded successfully!!!")
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

        scene_metrics = evaluate_open_voc_segmentation_sample(
            open_voc_segmenter,
            open_voc_processor,
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
