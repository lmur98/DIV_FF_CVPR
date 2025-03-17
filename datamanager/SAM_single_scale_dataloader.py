import typing

import torch
import torchvision.transforms as T
from datamanager.feature_dataloader import FeatureDataloader
from tqdm import tqdm
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
import os
from PIL import Image
from abc import ABC, ABCMeta, abstractmethod


class SAM_masks_with_CLIP(ABC):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        image_list: torch.Tensor,
        image_paths: typing.List[str],
        cache_path: str = None,
    ):
        assert "image_shape" in cfg
        self.emb_dim = 128
        self.cfg = cfg
        self.device = device
        self.cache_path = cache_path
        self.image_paths = image_paths #THIS DEPENDS IF IT IS VALIDATION OR TRAINING
        self.data = None
        self.load()
        
    def load(self):
        SAM_seg_maps = []
        SAM_feat_maps = {}
        SAM_img_IDs = []
        for img_path in self.image_paths:
            img_ID = img_path.split("/")[-1].split(".")[0].split("_")[-1]
            SAM_img_IDs.append(img_ID)
            filename = os.path.join(self.cache_path, 'frame_' + img_ID)
            #SAM_seg_maps.append(np.load(filename + "_s.npy")) #SEG MAP is (4, 245, 456)
            SAM_feat_maps[img_ID] = {}
            SAM_feat_maps[img_ID]['feat_map'] = np.load(filename + "_f.npy") #FEATURE MAP IS (N, 512)
            SAM_feat_maps[img_ID]['seg_map'] = np.load(filename + "_s.npy") #SEG MAP IS (4, 245, 456)
            #N goes from -1 to N. If it is -1, it is the border and they should be ignored in the loss function
        #SAM_seg_maps = torch.stack([torch.tensor(x) for x in SAM_seg_maps])
        #self.seg_maps = torch.tensor(SAM_seg_maps).to(self.device) # (N, 4, 245, 456) The 4 different segmentation maps
        self.SAM_img_ids = SAM_img_IDs
        self.SAM_feat_seg_maps = SAM_feat_maps
        #SAM_feat_maps = torch.stack([torch.tensor(x) for x in SAM_feat_maps])
        #print(SAM_feat_maps.shape, 'the shape of the feat maps')
    
    def __call__(self, img_points):
        img_IDs = [self.SAM_img_ids[int(x)] for x in img_points[:, 0]]
        #feat_maps = [self.feat_maps[x] for x in img_IDs]

        # img_points: (B, 3) # (img_ind, x, y)
        #img_scale = (
        #    self.data.shape[1] / self.cfg["image_shape"][0],
        #    self.data.shape[2] / self.cfg["image_shape"][1],
        #)
        out = []
        count_in_loss = []
        for i, ID in enumerate(img_IDs):
            feat_map = self.SAM_feat_seg_maps[ID]['feat_map']
            seg_map = self.SAM_feat_seg_maps[ID]['seg_map']
            x_ind = (img_points[i, 1] * 2).long() #2 is the scale
            y_ind = (img_points[i, 2] * 2).long()
            #In this version, we just take the default segmentation map
            seg_class = int(seg_map[x_ind, y_ind])
            if seg_class == -1:
                feat_in_pixel = torch.zeros(self.emb_dim).to(self.device)
                count_in_loss.append(False)
            else:
                feat_in_pixel = torch.tensor(feat_map[seg_class, :]).to(self.device)
                count_in_loss.append(True)
            out.append(feat_in_pixel)

        return torch.stack(out).half().to(self.device), torch.tensor(count_in_loss).to(self.device)

        #self.data is N, H, W, C
        #self.cgf["image_shape"] is H, W
        #x_ind, y_ind = (img_points[:, 1] * img_scale[0]).long(), (img_points[:, 2] * img_scale[1]).long()
        #return (self.data[img_points[:, 0].long(), x_ind, y_ind]).to(self.device)