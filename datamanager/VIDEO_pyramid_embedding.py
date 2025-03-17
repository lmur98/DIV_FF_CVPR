import json
import os
from pathlib import Path

import numpy as np
import torch
from datamanager.feature_dataloader import FeatureDataloader
from datamanager.patch_embedding_dataloader import PatchEmbeddingDataloader
from datamanager.VIDEO_patch_embedding_dataloader import VIDEO_PatchEmbeddingDataloader
from encoders.image_encoder import BaseImageEncoder
from tqdm import tqdm
import typing
from sklearn.decomposition import PCA, IncrementalPCA


class VIDEO_PyramidEmbeddingDataloader(FeatureDataloader):
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
        
        self.mode = cfg["mode"]
        if self.mode == "per_patches":
            assert "tile_size_range" in cfg
            assert "tile_size_res" in cfg
            assert "stride_scaler" in cfg
            assert "image_shape" in cfg
            assert "model_name" in cfg
            self.tile_sizes = torch.linspace(*cfg["tile_size_range"], cfg["tile_size_res"]).to(device)
            self.strider_scaler_list = [self._stride_scaler(tr.item(), cfg["stride_scaler"]) for tr in self.tile_sizes]
            print('EgoVideo tile sizes', self.tile_sizes, 'and stride scaler', self.strider_scaler_list)

        self.model = model
        self.embed_size = 512 #PCA Reduction !! #self.model.embedding_dim
        self.egovideo_dim = 512
        self.feature_type = "EGOVIDEO"
        self.data_dict = {}
        self.video_transform = transform
        super().__init__(cfg, device, image_list, image_paths, cache_path)

    def __call__(self, img_points, scale = None):
        return self._single_scale(img_points)

    def _stride_scaler(self, tile_ratio, stride_scaler):
        return np.interp(tile_ratio, [0.05, 0.15], [1.0, stride_scaler])

    def load(self):
        # don't create anything, PatchEmbeddingDataloader will create itself
        cache_info_path = self.cache_path.with_suffix(".info")

        # check if cache exists
        if not cache_info_path.exists():
            raise FileNotFoundError

        # if config is different, remove all cached content
        with open(cache_info_path, "r") as f:
            cfg = json.loads(f.read())
        if cfg != self.cfg:
            for f in os.listdir(self.cache_path):
                os.remove(os.path.join(self.cache_path, f))
            raise ValueError("Config mismatch")

        raise FileNotFoundError  # trigger create

    def create(self, image_list, image_paths):
        if self.mode == "per_patches":
            os.makedirs(self.cache_path, exist_ok=True)
            for i, tr in enumerate(tqdm(self.tile_sizes, desc="Scales")):
                stride_scaler = self.strider_scaler_list[i]
                self.data_dict[i] = VIDEO_PatchEmbeddingDataloader(
                    cfg={
                        "mode": "per_patches",
                        "tile_ratio": tr.item(),
                        "stride_ratio": stride_scaler,
                        "image_shape": self.cfg["image_shape"],
                        "model_name": self.cfg["model_name"],
                    },
                    device=self.device,
                    model="EgoVIDEO",
                    image_list=image_list,
                    image_paths=image_paths,
                    cache_path =Path(self.cache_path),
                    transform="EgoVIDEO_transform",
                )

        elif self.mode == "Int_Hots":
            os.makedirs(self.cache_path, exist_ok=True)
            self.data_dict[0] = VIDEO_PatchEmbeddingDataloader(
                cfg={
                    "mode": "Int_Hots",
                    "image_shape": self.cfg["image_shape"],
                    "model_name": self.cfg["model_name"],
                },
                device=self.device,
                model="EgoVIDEO",
                image_list=image_list,
                image_paths=image_paths,
                cache_path=Path(self.cache_path),
                transform="EgoVIDEO_transform",
            )
        else:
            raise ValueError("Error en el VIDEO Pyramid Embedding Dataloader")
            
        """_, _, _, EgoVideo_ch = self.data_dict[0].data.shape
        #If the features have still the original CLIP dimension, we proceed with a PCA reduction
        if EgoVideo_ch != self.embed_size:
            all_dict_feats = []
            n_feats_per_scale = []
            for i, tr in enumerate(tqdm(self.tile_sizes, desc="Scales")):
                N, H, W, _ = self.data_dict[i].data.shape
                flatten_feats = self.data_dict[i].data.reshape(-1, self.egovideo_dim)
                all_dict_feats.append(flatten_feats)
                n_feats_per_scale.append({'N': N, 'H': H, 'W': W})
            all_dict_feats = torch.cat(all_dict_feats, dim=0).numpy()
            
            print('We proceed with a incremental PCA reduction of the VIDEO embeddings')
            batch_size = 100000
            inc_pca = IncrementalPCA(n_components=self.embed_size, batch_size=batch_size)
            for i in range(0, all_dict_feats.shape[0], batch_size):
                inc_pca.partial_fit(all_dict_feats[i:i+batch_size])
            pca_features = np.empty((all_dict_feats.shape[0], self.embed_size))  # Crear un array para los datos transformados
            for i in range(0, all_dict_feats.shape[0], batch_size):
                pca_features[i:i+batch_size] = inc_pca.transform(all_dict_feats[i:i+batch_size])
            
            #Save the pca components
            pca_weights_file = f"{self.cache_path}/VIDEO_pca_weights.npz"
            np.savez(pca_weights_file, 
                    components = inc_pca.components_, 
                    mean = inc_pca.mean_, 
                    explained_variance = inc_pca.explained_variance_)

            pca_dict_features = {}
            for scale in range(len(n_feats_per_scale)):
                N, H, W = n_feats_per_scale[scale].values()
                pca_dict_features[scale] = torch.from_numpy(pca_features[:N*H*W]).reshape(N, H, W, self.embed_size)
                pca_features = pca_features[N*H*W:]

            for i, tr in enumerate(tqdm(self.tile_sizes, desc="Scales")):
                np.save(f"{self.cache_path}/VIDEO_level_{i}_pca.npy", pca_dict_features[i].numpy())
        #If the features have already the PCA dimension, we load the weights
        else:
            self.pca_weights = np.load(f"{self.cache_path}/VIDEO_pca_weights.npz")"""

    def save(self):
        cache_info_path = self.cache_path.with_suffix(".info")
        with open(cache_info_path, "w") as f:
            f.write(json.dumps(self.cfg))
        # don't save anything, PatchEmbeddingDataloader will save itself
        pass

    def _single_scale(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        # return: (B, 512), some random scale (between 0, 1)
        img_points = img_points.to(self.device)
        single_scale_feats = self.data_dict[0](img_points)
        return single_scale_feats

