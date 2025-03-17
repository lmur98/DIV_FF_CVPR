import typing

import torch
from datamanager.dino_extractor import ViTExtractor
from datamanager.feature_dataloader import FeatureDataloader
from tqdm import tqdm
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
import os

#import sys
#sys.path.append('/home/lmur/Feature_Fields/EgoVideo')
#from backbone.model.setup_model import *
from einops import rearrange
import argparse
import pandas as pd
import typing

import torch
import torchvision
from datamanager.feature_dataloader import FeatureDataloader
from tqdm import tqdm
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
import os
from PIL import Image
import json


class EgoVideoDataloader(FeatureDataloader):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        image_list: torch.Tensor,
        image_paths: typing.List[str],
        cache_path: str = None,
    ):
        assert "image_shape" in cfg
        self.upsample = False
        self.pca_components = 256
        self.num_channels = 1408
        self.sampled_frames = 4
        self.temporal_stride = 16
        self.video_patch_size = 14
        self.patches_per_frame = 256
        super().__init__(cfg, device, image_list, image_paths, cache_path)
    
    def create(self, image_list, image_paths):
        egovideo, tokenizer = build_vision_model(ckpt_path = '/home/lmur/Feature_Fields/EgoVideo/ckpt_4frames.pth', num_frames = 4)
        egovideo = egovideo.eval().to(self.device).to(torch.float16)

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Resize((224, 224)),
                                                    torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),])

        egovideo_embeds = []
        for low_res_img_path in tqdm(image_paths, desc="EgoVideo_features", total=len(image_list), leave=False):
            high_res_img_path = low_res_img_path.replace('frames', 'high_res_video').replace('.bmp', '.jpg').replace('IMG', 'frame')
            frame_number = int(high_res_img_path.split('/')[-1].split('_')[-1].split('.')[0])
            video_frames = [frame_number + i for i in range(0, self.sampled_frames * self.temporal_stride, self.temporal_stride)]
            video_frames_paths = []
            for frame in video_frames:
                # Creamos un nuevo filename con el número de frame formateado a 10 dígitos
                new_filename = f"frame_{str(frame).zfill(10)}.jpg"
                new_frame_path = os.path.join(os.path.dirname(high_res_img_path), new_filename)
                video_frames_paths.append(new_frame_path)
            
            video = []
            for frame in video_frames_paths:
                video.append(transform(Image.open(frame)).to(self.device))
            video = torch.stack(video, dim=1).unsqueeze(0) #B, C, T, H, W
            
            with torch.no_grad():
                video_features, text_features = egovideo(video, text = None, mask = None)        
            egoVideo_3D_MAP = rearrange(video_features, 'b (t l) c -> b t l c',  t=self.sampled_frames, l=self.patches_per_frame, c=self.num_channels)
            egoVideo_3D_MAP = egoVideo_3D_MAP.permute(0, 3, 1, 2)
            egoVideo_3D_MAP = egoVideo_3D_MAP.view(video.shape[0], self.num_channels, self.sampled_frames, int(self.patches_per_frame**0.5), int(self.patches_per_frame**0.5)).squeeze(0)

            egovideo_embeds.append(egoVideo_3D_MAP.cpu())

        self.data = torch.stack(egovideo_embeds, dim=0)        
        print("We proceed to PCA reduction", self.data.shape)
        #pca = PCA(n_components=self.pca_components, copy=False)
        N, C, T, H, W = self.data.shape
        all_features = self.data.permute(0, 2, 3, 4, 1).reshape(-1, C).numpy()
        print("Shape of all features before PCA: ", all_features.shape)
        
        batch_size = 1000000
        inc_pca = IncrementalPCA(n_components=self.pca_components)
        for i in range(0, all_features.shape[0], batch_size):
            print("Fitting PCA on batch: ", i)
            inc_pca.partial_fit(all_features[i:i+batch_size])
        # Transformar los datos después de ajustar el modelo PCA
        pca_features = np.empty((all_features.shape[0], self.pca_components))  # Crear un array para los datos transformados
        for i in range(0, all_features.shape[0], batch_size):
            print("Transforming PCA on batch:", i)
            pca_features[i:i+batch_size] = inc_pca.transform(all_features[i:i+batch_size])
        self.data = torch.Tensor(pca_features).view(N, T, H, W, self.pca_components) 
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