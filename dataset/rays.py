import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset

from .utils import *
from encoders.openclip_encoder import OpenCLIPNetwork
from datamanager.pyramid_embedding_dataloader import PyramidEmbeddingDataloader
from datamanager.dino_dataloader import DinoDataloader
from datamanager.egovideo_dataloader import EgoVideoDataloader
from datamanager.dinov2_dataloader import Dinov2Dataloader
from datamanager.SAM_single_scale_dataloader import SAM_masks_with_CLIP
from datamanager.VIDEO_pyramid_embedding import VIDEO_PyramidEmbeddingDataloader
from datamanager.VIDEO_SAM_dataloader import SAM_masks_with_EgoVIDEO
import os.path as osp
from pathlib import Path
from colmap_converter.frames import resize_image

def load_meta(root, name="meta.json"):
    """Load meta information per scene and frame (nears, fars, poses etc.)."""
    path = os.path.join(root, name)
    with open(path, "r") as fp:
        ds = json.load(fp)
    for k in ["nears", "fars", "images", "poses"]:
        ds[k] = {int(i): ds[k][i] for i in ds[k]}
        if k == "poses":
            ds[k] = {i: np.array(ds[k][i]) for i in ds[k]}
    ds["intrinsics"] = np.array(ds["intrinsics"])
    return ds


class EPICDiff(Dataset):
    def __init__(self, vid, root="data/EPIC-Diff", split=None):

        self.root = os.path.join(root, vid)
        self.vid = vid
        self.img_w = 228 # 456 # 228
        self.img_h = 128 # 256 # 128
        self.split = split
        self.val_num = 1
        self.transform = torchvision.transforms.ToTensor()
        self.init_meta()
        self.video_activated = True
        self.sampled_frames = 4
        self.temporal_stride = 8
        
        self.image_encoder = OpenCLIPNetwork()
        #self.image_encoder = OpenCLIPNetwork(clip_model_type = 'hf-hub:timm/ViT-SO400M-14-SigLIP-384')
        if (self.split == "train" or self.split == "val"):
            images, videos = [], []
            images_path = []
            if (self.split == "train"):
                idx_split = self.img_ids_train
            else:
                idx_split = self.img_ids_val
            for idx in idx_split:
                #img_path = os.path.join(self.root, "frames", self.image_paths[idx])
                img_path = os.path.join(self.root, "high_res_video", self.image_paths[idx])
                img_path = img_path.replace('bmp', 'jpg').replace('IMG', 'frame')
                img = Image.open(img_path)
                img_w, img_h = img.size
                img = self.transform(img)  # (3, h, w)
                img = img.reshape(-1, 3, img_h, img_w)  # (N_images-1, 3, h, w) RGB
                images += [img]
                images_path.append(img_path)
                
                if self.video_activated:
                    frame_number = int(img_path.split('/')[-1].split('_')[-1].split('.')[0])
                    video_frames = [frame_number + i for i in range(0, self.sampled_frames * self.temporal_stride, self.temporal_stride)]
                    video_frames_paths = []
                    for frame in video_frames:
                        # Creamos un nuevo filename con el número de frame formateado a 10 dígitos
                        new_filename = f"frame_{str(frame).zfill(10)}.jpg"
                        new_frame_path = os.path.join(os.path.dirname(img_path), new_filename)
                        video_frames_paths.append(new_frame_path)
                    video = []
                    for frame in video_frames_paths:
                        frame = self.transform(Image.open(frame))
                        video.append(frame)
                    video = torch.stack(video, dim = 1).unsqueeze(0) #B, C, T, H, W
                    videos += [video]
                    
            images = torch.cat(images, 0)
            if self.video_activated:
                videos = torch.cat(videos, 0)
            

            # self.image_encoder = OpenCLIPNetwork()
            cache_dir = f"outputs/{vid}"
            if (self.split == "train"):
                #clip_cache_path = Path(osp.join(cache_dir, f"train_clip_{self.image_encoder.name}"))   
                clip_cache_path = Path(osp.join(cache_dir, f"SAM2bbox_CLIP_features_dim128"))
                dino_cache_path = Path(osp.join(cache_dir, "train_dinov2_64PCA.npy"))
                IntHots_video_cache_path = Path(osp.join(cache_dir, "train_egovideo"))
                PATCH_video_cache_path = Path(osp.join(cache_dir, "train_egovideo"))
                SAM_video_cache_path = Path(osp.join(cache_dir, "SAM_EgoVIDEO_features_dim128"))
            else:
                #clip_cache_path = Path(osp.join(cache_dir, f"val_clip_{self.image_encoder.name}"))   
                clip_cache_path = Path(osp.join(cache_dir, f"SAM2bbox_CLIP_features_dim128"))
                dino_cache_path = Path(osp.join(cache_dir, "val_dinov2_64PCA.npy"))
                IntHots_video_cache_path = Path(osp.join(cache_dir, "val_egovideo"))
                PATCH_video_cache_path = Path(osp.join(cache_dir, "val_egovideo"))
                SAM_video_cache_path = Path(osp.join(cache_dir, "SAM_EgoVIDEO_features_dim128"))
        
            #torch.cuda.empty_cache()
            #self.dino_dataloader = DinoDataloader(
            #    image_list=images,
            #    device=torch.device('cuda:0'),
            #    cfg={"image_shape": list(images.shape[2:4])},
            #    cache_path=dino_cache_path,
            #)
            
            print('Lets extract Dino features')
            torch.cuda.empty_cache()
            self.dinov2_dataloader = Dinov2Dataloader(
                image_list=images,
                image_paths=images_path,
                device=torch.device('cuda:0'),
                cfg={"image_shape": list(images.shape[2:4])},
                cache_path=dino_cache_path,
            )
            print("Dino dataloader created", self.dinov2_dataloader.data.shape)
            print()

            print('Lets extract CLIP + SAM features')
            torch.cuda.empty_cache()
            self.SAM_CLIP = SAM_masks_with_CLIP(
                image_list=images,
                image_paths=images_path,
                device = torch.device('cuda:0'),
                cfg={
                    "image_shape": list(images.shape[2:4]),
                },
                cache_path=clip_cache_path,
            )
            print()
            
        

            """print('CLIP CACHE PATH', clip_cache_path, len(images))
            torch.cuda.empty_cache()
            self.clip_interpolator = PyramidEmbeddingDataloader(
                image_list=images,
                device = torch.device('cuda:0'),
                cfg={
                    "tile_size_range": [0.05, 0.5],
                    "tile_size_res": 7,
                    "stride_scaler": 0.5,
                    "image_shape": list(images.shape[2:4]),
                    "model_name": self.image_encoder.name,
                },
                cache_path=clip_cache_path,
                model=self.image_encoder,
            )
            print("Clip interpolator created")"""
            
            
            
            torch.cuda.empty_cache()
            if self.video_activated:
                """self.egovideo_dataloader_SAM = SAM_masks_with_EgoVIDEO(
                    image_list=images,
                    image_paths=images_path,
                    device = torch.device('cuda:0'),
                    cfg={
                        "image_shape": list(images.shape[2:4]),
                    },
                    cache_path=SAM_video_cache_path,
                )"""

                self.egovideo_dataloader_PATCH = VIDEO_PyramidEmbeddingDataloader(
                    image_list=videos,
                    image_paths=images_path,
                    device = torch.device('cuda:0'),
                    cfg={
                        "mode": "per_patches",
                        "tile_size_range": [0.3, 0.3], #[0.45, 0.45],
                        "tile_size_res": 1,
                        "stride_scaler": 0.5, #0.5,
                        "image_shape": list(images.shape[2:4]),
                        "model_name": "EgoVideo",
                    },
                    cache_path=PATCH_video_cache_path.with_name(PATCH_video_cache_path.name + "_all_patches_stride0dot3"),
                    model="EgoVideo",
                    transform = "EgoVideo_Transform",
                )
                print("EgoVideo patches created")
                print()
                self.egovideo_dataloader_IntHots = VIDEO_PyramidEmbeddingDataloader(
                    image_list=videos,
                    image_paths=images_path,
                    device = torch.device('cuda:0'),
                    cfg={
                        "mode": "Int_Hots",
                        "image_shape": list(images.shape[2:4]),
                        "model_name": "EgoVideo",
                    },
                    cache_path=IntHots_video_cache_path.with_name(IntHots_video_cache_path.name + "_IntHots"),
                    model="EgoVideo",
                    transform="EgoVideo_Transform",
                )
                print("EgoVideo IntHots created")
                print()
                
            torch.cuda.empty_cache()

    def imshow(self, index):
        plt.imshow(self.imread(index))
        plt.axis("off")
        plt.show()

    def imread(self, index):
        return plt.imread(os.path.join(self.root, "frames", self.image_paths[index]))

    def x2im(self, x, type_="np"):
        """Convert numpy or torch tensor to numpy or torch 'image'."""
        w = self.img_w
        h = self.img_h
        if len(x.shape) == 2 and x.shape[1] == 3:
            x = x.reshape(h, w, 3)
        else:
            x = x.reshape(h, w)
        if type(x) == torch.Tensor:
            x = x.detach().cpu()
            if type_ == "np":
                x = x.numpy()
        elif type(x) == np.array:
            if type_ == "pt":
                x = torch.from_numpy(x)
        return x

    def rays_per_image(self, idx, pose=None):
        """Return sample with rays, frame index etc."""
        sample = {}
        if pose is None:
            sample["c2w"] = c2w = torch.FloatTensor(self.poses_dict[idx])
        else:
            sample["c2w"] = c2w = pose

        sample["im_path"] = self.image_paths[idx]

        img = Image.open(os.path.join(self.root, "frames", self.image_paths[idx]))
        # img_w, img_h = img.size
        img = self.transform(img)  # (3, h, w) 
        img_w, img_h = img.shape[2], img.shape[1]       
        img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB

        directions = get_ray_directions(img_h, img_w, self.K)
        rays_o, rays_d = get_rays(directions, c2w)

        c2c = torch.zeros(3, 4).to(c2w.device)
        c2c[:3, :3] = torch.eye(3, 3).to(c2w.device)
        rays_o_c, rays_d_c = get_rays(directions, c2c)

        rays_t = idx * torch.ones(len(rays_o), 1).long()

        rays = torch.cat(
            [
                rays_o,
                rays_d,
                self.nears[idx] * torch.ones_like(rays_o[:, :1]),
                self.fars[idx] * torch.ones_like(rays_o[:, :1]),
                rays_o_c,
                rays_d_c,
            ],
            1,
        )

        sample["rays"] = rays
        sample["img_wh"] = torch.LongTensor([img_w, img_h])
        sample["ts"] = rays_t
        sample["rgbs"] = img

        # Create ray_indices for CLIP interpolation
        if (self.split == "train" or self.split == "val"):
            if (self.split == "train"):
                idx_to_append = self.img_ids_train.index(idx)
            elif (self.split == "val"):
                idx_to_append = self.img_ids_val.index(idx)
            else:
                idx_to_append = self.img_ids_test.index(idx)     
            indices = []
            for i in range(img_h):
                for j in range(img_w):
                    indices.append([idx_to_append, i, j])
            indices = torch.tensor(indices)
            sample["indices"] = indices
    
        return sample

    def init_meta(self):
        """Load meta information, e.g. intrinsics, train, test, val split etc."""
        meta = load_meta(self.root)
        self.img_ids = meta["ids_all"]
        self.img_ids_train = meta["ids_train"]
        self.img_ids_test = meta["ids_test"]
        self.img_ids_val = meta["ids_val"]
        self.poses_dict = meta["poses"]
        self.nears = meta["nears"]
        self.fars = meta["fars"]
        self.image_paths = meta["images"]
        self.K = meta["intrinsics"]

        if self.split == "train":
            # create buffer of all rays and rgb data
            self.rays = []
            self.rgbs = []
            self.ts = []
            self.indices = []

            for idx in self.img_ids_train:
                sample = self.rays_per_image(idx)
                self.rgbs += [sample["rgbs"]]
                self.rays += [sample["rays"]]
                self.ts += [sample["ts"]]
                self.indices += [sample["indices"]]

            self.rays = torch.cat(self.rays, 0)  # ((N_images-1)*h*w, 8)
            self.rgbs = torch.cat(self.rgbs, 0)  # ((N_images-1)*h*w, 3)
            self.ts = torch.cat(self.ts, 0)
            self.indices = torch.cat(self.indices, 0)

    def __len__(self):
        if self.split == "train":
            # rays are stored concatenated
            return len(self.rays)
        if self.split == "val":
            # evaluate only one image, sampled from val img ids
            return 1
        else:
            # choose any image index
            return max(self.img_ids)

    def __getitem__(self, idx, pose=None):

        if self.split == "train":
            # samples selected from prefetched train data
            sample = {
                "rays": self.rays[idx],
                "ts": self.ts[idx, 0].long(),
                "rgbs": self.rgbs[idx],
                "indices": self.indices[idx]
            }

        elif self.split == "val":
            # for tuning hyperparameters, tensorboard samples
            idx = random.choice(self.img_ids_val)
            sample = self.rays_per_image(idx, pose)

        elif self.split == "test":
            # evaluating according to table in paper, chosen index must be in test ids
            assert idx in self.img_ids_test
            sample = self.rays_per_image(idx, pose)

        else:
            # for arbitrary samples, e.g. summary video when rendering over all images
            sample = self.rays_per_image(idx, pose)

        return sample
