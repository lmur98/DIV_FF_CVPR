import os
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy

import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

import sys

# Establecer PYTHONPATH dentro del script Python
sys.path.append("/home/lmur/FUSION_FIELDS/sam2")

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only make GPU:1 visible

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)

class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    if w == 0:
        w = 1
    if h == 0:
        h = 1
    if x + h > image.shape[0]:
        x = x - 1
    if y + w > image.shape[1]:
        y = y - 1
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img

def mask2segmap(masks, image):
    seg_img_list = []
    seg_map = -np.ones(image.shape[:2], dtype=np.int32)
    for i in range(len(masks)):
        mask = masks[i]
        seg_img = get_seg_img(mask, image)
        pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))
        seg_img_list.append(pad_seg_img)

        seg_map[masks[i]['segmentation']] = i
    seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
    seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')

    return seg_imgs, seg_map

def save_numpy(save_path, data):
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    np.save(save_path_s, data['seg_maps'])
    np.save(save_path_f, data['feature'])

def paint_seg_map(seg_map, show_path):
    seg_map = seg_map.copy()
    unique_classes = np.unique(seg_map)
    N_classes = len(np.unique(seg_map))
    colors = np.random.rand(N_classes, 3)
    colors[0] = [0, 0, 0]
    
    class_map = {k: i for i, k in enumerate(unique_classes)}
    seg_map_mapped = np.vectorize(class_map.get)(seg_map)
    cmap = ListedColormap(colors)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(seg_map_mapped, cmap=cmap, norm=Normalize(vmin=0, vmax=N_classes-1))
    plt.title('Segmentation map of the last frame')
    plt.savefig(show_path + '_seg_map.png')
    plt.close()
    
def get_bbox_tiles(seg_map, image):
    bbox_tiles = []
    for i in range(seg_map.max() + 1):
        mask = seg_map == i
        bbox = get_bbox_from_mask(mask)
        cropped_img = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        cropped_img = (torch.from_numpy(cropped_img.astype("float32")).permute(2,0,1) / 255.0).to('cuda')
        cropped_img = model.process(cropped_img).half()
        bbox_tiles.append(cropped_img)
    bbox_tiles = torch.stack(bbox_tiles, dim=0)
    return bbox_tiles

def get_bbox_from_mask(mask):
    if not np.any(mask):  # This returns True if all elements in the mask are 0
        # Return the bounding box for the entire image
        return 0, 0, mask.shape[1] - 1, mask.shape[0] - 1
    
    #Find the non-zero elements
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # Get the bounding box coordinates
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    if ymax - ymin == 0 or xmax - xmin == 0:
        print(f"Invalid bounding box with zero dimension: {(xmin, ymin, xmax, ymax)}")
        return xmin, ymin, xmax + 1, ymax + 1  # Adjust to ensure non-zero area
    
    # Return the bounding box as (xmin, ymin, xmax, ymax)
    return int(xmin), int(ymin), int(xmax), int(ymax)


def create(image_list, data_list, save_folder):
    assert image_list is not None, "image_list must be provided to generate features"
    mask_generator.predictor.model.to('cuda')

    for img, frame_ID in tqdm(zip(image_list, data_list), desc="Embedding images", leave=False):
        #Extract the masks
        frame_ID = frame_ID.split('.')[0]
        masks_last_frame = mask_generator.generate(img)  
        
        #Extract the embeddings of each mask
        seg_imgs, seg_map = mask2segmap(masks_last_frame, img)
        bbox_tiles = get_bbox_tiles(seg_map, img).to("cuda")
        tiles = seg_imgs.to("cuda")
        with torch.no_grad():
            clip_embed = model.encode_image(tiles)
            clip_bbox_embed = model.encode_image(bbox_tiles)
        clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
        clip_bbox_embed /= clip_bbox_embed.norm(dim=-1, keepdim=True)
        joined_embed = 0.75 * clip_embed + 0.25 * clip_bbox_embed #In the official code they have 0.75, 0.25
        joined_embed /= joined_embed.norm(dim=-1, keepdim=True)
        joined_embed = joined_embed.detach().cpu().half()

        #Save the embeddings and the masks
        save_path = os.path.join(save_folder, frame_ID)
        paint_seg_map(seg_map, save_path)
        save_numpy(save_path, {'feature': joined_embed, 'seg_maps': seg_map})
    

def seed_everything(seed_value):
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    seed_num = 42
    seed_everything(seed_num)

    root_path = '/home/lmur/FUSION_FIELDS/Lorenzo_Feature_Fields_v2/data/EPIC-Diff/'
    torch.set_default_dtype(torch.float32)
    
    model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    #SAM models
    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    image_predictor = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

    mask_generator = SAM2AutomaticMaskGenerator(
        model=image_predictor,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=225,
    )
    
    scene_list = ["P08_01", "P09_02", "P13_03", "P16_01", "P21_01"] #"P01_01" "P03_04", "P04_01", "P05_01", "P06_03", "P08_01", "P09_02", "P13_03", "P16_01", "P21_01"
    for scene in scene_list:
        print('----------------------------------------We are processing the scene', scene, '----------------------------------------')
        img_folder = os.path.join(root_path, scene, 'images')
        data_list = os.listdir(img_folder)
        data_list.sort()

        img_list = []
        for data_path in data_list:
            image_path = os.path.join(img_folder, data_path)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_list.append(image)
        print('The number of loaded images is:', len(img_list))
        save_folder = os.path.join('/home/lmur/FUSION_FIELDS/Lorenzo_Feature_Fields_v2/outputs', scene, 'SAM2bbox_CLIP_features')
        os.makedirs(save_folder, exist_ok=True)
        create(img_list, data_list, save_folder)
        print()