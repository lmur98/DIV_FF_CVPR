import json
import typing

import numpy as np
import os
import torch
from datamanager.feature_dataloader import FeatureDataloader
from encoders.image_encoder import BaseImageEncoder
from tqdm import tqdm
from sklearn.decomposition import PCA, IncrementalPCA

import sys
sys.path.append('/home/lmur/FUSION_FIELDS')
from EgoVideo.backbone.model.setup_model import build_video_model, build_text_model

import cv2
import torchvision


class VIDEO_PatchEmbeddingDataloader(FeatureDataloader):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        model: BaseImageEncoder,
        image_list: torch.Tensor = None,
        image_paths: typing.List[str] = None,
        cache_path: str = None,
        transform: typing.Any = None,
    ):
        
        assert "mode" in cfg
        print("EGOVIDEO con el modo", cfg["mode"], 'el output es', cache_path)
        
        self.extraction_type = cfg["mode"]
        if self.extraction_type == "Int_Hots":
            self.cfg = cfg
            self.emb_dim = 512
            self.images_paths = image_paths
            scene_id = str(cache_path).split('/')[-2]
            detections_file = os.path.join('/home/lmur/FUSION_FIELDS/Lorenzo_Feature_Fields_v2/data/EPIC-Diff', scene_id, 'hand_obj_detections.json')
            self.detections = json.load(open(detections_file, 'r'))
            
        elif self.extraction_type == "per_patches":
            assert "tile_ratio" in cfg
            assert "stride_ratio" in cfg
            assert "image_shape" in cfg
            assert "model_name" in cfg
            self.kernel_size = int(cfg["image_shape"][0] * cfg["tile_ratio"])
            self.stride = int(self.kernel_size * cfg["stride_ratio"])
            print('In EgoVideo', cfg["image_shape"], 'the image shape with a kernel of', self.kernel_size, 'and a stride of', self.stride)
            
            self.padding = self.kernel_size // 2
            self.center_x = (
                (self.kernel_size - 1) / 2
                - self.padding
                + self.stride
                * np.arange(
                    np.floor((cfg["image_shape"][0] + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride + 1)
                )
            )
            self.center_y = (
                (self.kernel_size - 1) / 2
                - self.padding
                + self.stride
                * np.arange(
                    np.floor((cfg["image_shape"][1] + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride + 1)
                )
            )
            self.center_x = torch.from_numpy(self.center_x)
            self.center_y = torch.from_numpy(self.center_y)
            print('CENTER X of the video', self.center_x)
            print('CENTER Y of the video', self.center_y)
            self.start_x = self.center_x[0].float()
            self.start_y = self.center_y[0].float()

            self.model = model
            self.video_transform = transform
            self.embed_size = 512 #self.model.embedding_dim
            self.sub_batchsize = 96
        super().__init__(cfg, device, image_list, image_paths, cache_path)

    def load(self):
        if self.extraction_type == "per_patches":
            #self.cache_path in this case is level_0.npy for example
            cache_info_path = self.cache_path.with_suffix(".info")
            data_path = self.cache_path.with_name(self.cache_path.stem + ".npy")
            print("Loading data from ", data_path, "en el modo", self.extraction_type)
            if not cache_info_path.exists():
                raise FileNotFoundError
            
            data = torch.from_numpy(np.load(data_path)) #Loads with the PCA reduction
            for ft in range(data.shape[0]):
                data[ft] = torch.nn.functional.normalize(data[ft].float(), dim=-1)
            self.data = data.float()
            print("EgoVideo per Patch features:", self.data.shape, 'loaded!!!')
            
            
        elif self.extraction_type == "Int_Hots":
            VIDEO_feat_maps = {}
            VIDEO_img_IDs = []
            for img_path in self.images_paths:
                img_ID = img_path.split("/")[-1].split(".")[0].split("_")[-1]
                VIDEO_img_IDs.append(img_ID)
                filename = os.path.join(self.cache_path, img_ID)
                #SAM_seg_maps.append(np.load(filename + "_s.npy")) #SEG MAP is (4, 245, 456)
                VIDEO_feat_maps[img_ID] = {}
                VIDEO_feat_maps[img_ID]['feat_map'] = np.load(filename + "_feats.npy") #FEATURE MAP IS (1, 512)
                VIDEO_feat_maps[img_ID]['seg_map'] = np.load(filename + "_masks.npy") #SEG MAP IS (245, 456)

                #N goes from -1 to N. If it is -1, it is the border and they should be ignored in the loss function
            #SAM_seg_maps = torch.stack([torch.tensor(x) for x in SAM_seg_maps])
            #self.seg_maps = torch.tensor(SAM_seg_maps).to(self.device) # (N, 4, 245, 456) The 4 different segmentation maps
            self.VIDEO_img_ids = VIDEO_img_IDs
            self.VIDEO_feat_seg_maps = VIDEO_feat_maps
            print("En el modo IntHotspots, loaded data shape: ", len(self.VIDEO_img_ids))
            

    def create(self, image_list, images_path):
        if self.extraction_type == "per_patches":
            video_encoder, _ = build_video_model(ckpt_path = '/home/lmur/FUSION_FIELDS/EgoVideo/ckpt_4frames.pth', num_frames = 4)
            self.model = video_encoder.eval().to("cuda:0").to(torch.float16)
            self.video_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                                                    torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),])

            assert self.model is not None, "model must be provided to generate features"
            assert image_list is not None, "image_list must be provided to generate features"

            unfold_func = torch.nn.Unfold(
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ).to(self.device)

            img_embeds = []
            for img in tqdm(image_list, desc="Extracting PATCH features of videos", leave=False):
                img_embeds.append(self._embed_EgoVIDEO_tiles(img, unfold_func))
            self.data = torch.from_numpy(np.stack(img_embeds))
            
            
            os.makedirs(self.cache_path.parent, exist_ok=True)
            cache_info_path = self.cache_path.with_suffix(".info")
            with open(cache_info_path, "w") as f:
                f.write(json.dumps(self.cfg))
            print('Guardamos el EgoVideo PATCH en', self.cache_path)
            np.save(self.cache_path, self.data)
        
        elif self.extraction_type == "Int_Hots":
            video_encoder, _ = build_video_model(ckpt_path = '/home/lmur/FUSION_FIELDS/EgoVideo/ckpt_4frames.pth', num_frames = 4)
            self.model = video_encoder.eval().to("cuda:0").to(torch.float16)
            self.video_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                                                    torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),])
            
            assert self.model is not None, "model must be provided to generate features"
            assert image_list is not None, "image_list must be provided to generate features"
            
            print("Guardamos el EgoVideo IntHotspot en", self.cache_path)
            os.makedirs(self.cache_path.parent, exist_ok=True)
            cache_info_path = self.cache_path.with_suffix(".info")
            with open(cache_info_path, "w") as f:
                f.write(json.dumps(self.cfg))
            
            for v_i, video in tqdm(enumerate(image_list), desc = "Extracting IntHotspots features of videos", leave = False):
                last_frame_path = images_path[v_i]
                last_frame_id = last_frame_path.split('/')[-1].split('.')[0].split('_')[-1]
                last_frame_detections = self.find_frame(self.detections, last_frame_id)
                video_descriptor, mask_interaction = self._embed_EgoVIDEO_interaction(video, last_frame_detections)
                save_path = os.path.join(self.cache_path, last_frame_id)

                save_path_feats = save_path + "_feats.npy"
                save_path_masks = save_path + "_masks.npy"
                np.save(save_path_feats, video_descriptor)
                np.save(save_path_masks, mask_interaction)
                
    def find_frame(self, data, frame_number):
        for entry in data:
            if entry['frame_number'] == frame_number:
                return entry
        return None       

    def __call__(self, img_points):
        if self.extraction_type == "per_patches":
            img_points = img_points.cpu()
            img_ind, img_points_x, img_points_y = img_points[:, 0], img_points[:, 1], img_points[:, 2]
            img_points_x = img_points_x * 2 #Scale factor 228 -> 456. The img_points come from the low res image, but the stride is in the reference of the high res image
            img_points_y = img_points_y * 2 #Scale factor 128 -> 256
            
            x_ind = torch.floor((img_points_x - (self.start_x)) / self.stride).long()
            y_ind = torch.floor((img_points_y - (self.start_y)) / self.stride).long()
            #return self._interp_inds(img_ind, x_ind, y_ind, img_points_x, img_points_y)
            img_ind = img_ind.to(self.data.device)  # self.data is on cpu to save gpu memory, hence this line
            # Extraer el único tile (solo un tile en la escala única)

            tile_feat = self.data[img_ind, x_ind, y_ind].to(self.device)
            return tile_feat #We count all the tiles in the loss
        
        elif self.extraction_type == "Int_Hots":
            img_IDs = [self.VIDEO_img_ids[int(x)] for x in img_points[:, 0]]

            out = []
            count_in_loss = []
            for i, ID in enumerate(img_IDs):
                feat_map = self.VIDEO_feat_seg_maps[ID]['feat_map']
                seg_map = self.VIDEO_feat_seg_maps[ID]['seg_map']
                x_ind = (img_points[i, 1] * 2).long() #2 is the scale
                y_ind = (img_points[i, 2] * 2).long()
                
                #In this version, we just take the default segmentation map
                seg_class = int(seg_map[x_ind, y_ind])
                if seg_class == 0: #There is no interaction hotspot, we don't count it in the loss
                    feat_in_pixel = torch.zeros(self.emb_dim).to(self.device)
                    count_in_loss.append(False)
                else: #There is an interaction hotspot, we count it in the loss
                    feat_in_pixel = torch.tensor(feat_map.squeeze()).to(self.device)
                    count_in_loss.append(True)
                out.append(feat_in_pixel)

            return torch.stack(out).to(self.device), torch.tensor(count_in_loss).to(self.device)
        
        else:
            raise ValueError("Error en el Video Patch Embedding Dataloader")
    
    def _embed_EgoVIDEO_interaction(self, video, detections):
        #VIDEO is (C, T, H, W)
        with torch.no_grad():
            video = self.video_transform(video).unsqueeze(0).to(torch.float16).to("cuda:0")
            video_feats, _ = self.model(video, None, None)
            video_feats = video_feats.detach().cpu().numpy()
        
        mask = np.zeros((self.cfg["image_shape"][0], self.cfg["image_shape"][1]))
        #Mask with the interaction hotspot
        if detections is not None:
            pixels_for_context = 10
            if detections['left_hand'] is not None:
                has_int = False
                for obj in detections['objects']:
                    int_bbox, is_int = self.intersection_box(detections['left_hand'], obj)
                    if is_int:
                        int_bbox = self.add_context_to_bbox(int_bbox, pixels_for_context)
                        mask[int_bbox[1]:int_bbox[3], int_bbox[0]:int_bbox[2]] = 1
                        has_int = True
                if not has_int:
                    hand_bbox = detections['left_hand']
                    hand_bbox = self.add_context_to_bbox(hand_bbox, pixels_for_context)
                    mask[hand_bbox[1]:hand_bbox[3], hand_bbox[0]:hand_bbox[2]] = 1

            if detections['right_hand'] is not None:
                has_int = False
                for obj in detections['objects']:
                    int_bbox, is_int = self.intersection_box(detections['right_hand'], obj)
                    if is_int:
                        int_bbox = self.add_context_to_bbox(int_bbox, pixels_for_context)
                        mask[int_bbox[1]:int_bbox[3], int_bbox[0]:int_bbox[2]] = 1
                        has_int = True
                if not has_int:
                    hand_bbox = detections['right_hand']
                    hand_bbox = self.add_context_to_bbox(hand_bbox, pixels_for_context)
                    mask[hand_bbox[1]:hand_bbox[3], hand_bbox[0]:hand_bbox[2]] = 1
                    
        return video_feats, mask

    def add_context_to_bbox(self, bbox, context):
        bbox[0] = int(max(0, bbox[0] - context))
        bbox[1] = int(max(0, bbox[1] - context))
        bbox[2] = int(min(456, bbox[2] + context))
        bbox[3] = int(min(256, bbox[3] + context))
        return bbox

    def intersection_box(self, boxA, boxB):
        # Calculate the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        # Check if there is an intersection by ensuring width and height are positive
        if xA < xB and yA < yB:
            # There is an intersection, return the coordinates and True
            return [xA, yA, xB, yB], True
        else:
            # No intersection
            return None, False
        
    def union_box(self, boxA, boxB):
        # Calculate the (x, y)-coordinates of the union rectangle
        xA = min(boxA[0], boxB[0])
        yA = min(boxA[1], boxB[1])
        xB = max(boxA[2], boxB[2])
        yB = max(boxA[3], boxB[3])
        
        # Return the union box
        return [xA, yA, xB, yB]

    def _embed_EgoVIDEO_tiles(self, video, unfold_func):
        # image augmentation: slow-ish (0.02s for 600x800 image per augmentation)
        #VIDEO is (C, T, H, W)
        unfolded_video = []
        for t in range(video.shape[1]):
            frame = video[:, t, :, :].unsqueeze(0) #(1, C, H, W)
            unfolded_frame = unfold_func(frame).permute(2, 0, 1).reshape(-1, 3, self.kernel_size, self.kernel_size).to("cuda") #(Batch tiles, C, H_kernel, W_kernel)
            unfolded_video.append(unfolded_frame)
        unfolded_video = torch.stack(unfolded_video, dim=2)
        
        video_embeds = []
        with torch.no_grad():
            for batch in range(unfolded_video.shape[0]):
                unfolded_video_transform = self.video_transform(unfolded_video[batch])
                video_feats, _ = self.model(unfolded_video_transform.unsqueeze(0), None, None)
                video_embeds.append(video_feats)
        video_embeds = torch.cat(video_embeds, dim=0)        
        
        clip_embeds = video_embeds.reshape((self.center_x.shape[0], self.center_y.shape[0], -1))
        clip_embeds = torch.concat((clip_embeds, clip_embeds[:, [-1], :]), dim=1)
        clip_embeds = torch.concat((clip_embeds, clip_embeds[[-1], :, :]), dim=0)
        return clip_embeds.detach().cpu().numpy()
