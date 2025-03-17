import typing

import torch
import torchvision.transforms as T
from datamanager.feature_dataloader import FeatureDataloader
from tqdm import tqdm
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
import os
from PIL import Image


class Dinov2Dataloader(FeatureDataloader):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        image_list: torch.Tensor,
        image_paths: typing.List[str],
        cache_path: str = None,
    ):
        assert "image_shape" in cfg
        self.pca_components = 64
        self.num_channels = 384
        super().__init__(cfg, device, image_list, image_paths, cache_path)
    
    def create(self, image_list, image_paths):
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        h, w = 266, 476 #In order to be divisible by 14
        transform = T.Compose([T.Resize((h, w)),
                               T.Normalize(mean=[0.5], std=[0.5]),])

        dino_embeds = []
        print("We proceed to extract features")
        for image in tqdm(image_list, desc="dino", total=len(image_list), leave=False):
            #high_res_img_path = low_res_img_path.replace('frames', 'high_res_video').replace('.bmp', '.jpg').replace('IMG', 'frame')
            img_torch = transform(image).unsqueeze(0).to(self.device)
            B, C, H, W = img_torch.shape
            with torch.no_grad():
                patch_features = dinov2.forward_features(img_torch, masks = None)["x_norm_patchtokens"]
            patch_features = patch_features.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2).squeeze(0)

            dino_embeds.append(patch_features.cpu())

        self.data = torch.stack(dino_embeds, dim=0)
        
        print("We proceed to PCA reduction")
        pca = PCA(n_components=self.pca_components, copy=False)
        N, C, H, W = self.data.shape
        all_features = self.data.permute(0, 2, 3, 1).reshape(-1, C).numpy()
        X = pca.fit_transform(all_features)
        reduced_feats = torch.Tensor(X).view(N, H, W, self.pca_components) 
        #print('PCA DINOv2', reduced_feats.shape)
        for ft in range(reduced_feats.shape[0]):
            reduced_feats[ft] = torch.nn.functional.normalize(reduced_feats[ft], dim=-1)
        self.data = reduced_feats
        print("Features shape (PCA) completed!!: ", self.data.shape)

    def create_low_res(self, image_list, image_paths):
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        h, w = 140, 238 #In order to be divisible by 14
        transform = T.Compose([T.Resize((h, w)),
                               T.Normalize(mean=[0.5], std=[0.5]),])

        dino_embeds = []
        for image in tqdm(image_list, desc="dino", total=len(image_list), leave=False):
            img_torch = transform(image).unsqueeze(0).to(self.device)
            B, C, H, W = img_torch.shape
            with torch.no_grad():
                patch_features = dinov2.forward_features(img_torch, masks = None)["x_norm_patchtokens"]
            patch_features = patch_features.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)
            if self.upsample: #TO DO
                patch_features = torch.nn.functional.interpolate(patch_features, (128, 228)).squeeze(0)
            else:
                patch_features = patch_features.squeeze(0)
            dino_embeds.append(patch_features.cpu())

        self.data = torch.stack(dino_embeds, dim=0)
        
        print("We proceed to PCA reduction")
        pca = PCA(n_components=self.pca_components, copy=False)
        N, C, H, W = self.data.shape
        all_features = self.data.permute(0, 2, 3, 1).reshape(-1, C).numpy()
        X = pca.fit_transform(all_features)
        self.data = torch.Tensor(X).view(N, H, W, self.pca_components) 
        #self.data is torch.Size([752, 64, 10, 17]) #N, C, H, W
        print("Features shape (PCA) completed!!: ", self.data.shape)
        
    def __call__(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        img_scale = (
            self.data.shape[1] / self.cfg["image_shape"][0],
            self.data.shape[2] / self.cfg["image_shape"][1],
        )
        #self.data is N, H, W, C
        #self.cgf["image_shape"] is H, W
        x_ind, y_ind = (img_points[:, 1] * img_scale[0]).long(), (img_points[:, 2] * img_scale[1]).long()
        return (self.data[img_points[:, 0].long(), x_ind, y_ind]).to(self.device)