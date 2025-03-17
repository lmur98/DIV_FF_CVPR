import typing

import torch
from datamanager.dino_extractor import ViTExtractor
from datamanager.feature_dataloader import FeatureDataloader
from tqdm import tqdm
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
import os


class DinoDataloader(FeatureDataloader):
    dino_model_type = "dino_vits8"
    dino_stride = 8
    dino_load_size = 500
    dino_layer = 11
    dino_facet = "key"
    dino_bin = False

    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        image_list: torch.Tensor,
        cache_path: str = None,
    ):
        assert "image_shape" in cfg
        super().__init__(cfg, device, image_list, cache_path)

    def create(self, image_list):
        print(self.cache_path.parent, self.cache_path, 'queeee es esto')
        print(image_list[0].shape, 'shape before the preprocessing')
        extractor = ViTExtractor(self.dino_model_type, self.dino_stride)
        preproc_image_lst = extractor.preprocess(image_list, self.dino_load_size)[0].to(self.device)

        dino_embeds = []
        for image in tqdm(preproc_image_lst, desc="dino", total=len(image_list), leave=False):
            print(image.shape)
            with torch.no_grad():
                descriptors = extractor.extract_descriptors(
                    image.unsqueeze(0),
                    [self.dino_layer],
                    self.dino_facet,
                    self.dino_bin,
                )
            print(descriptors.shape, 'shape of the descriptors')
            descriptors = descriptors.reshape(extractor.num_patches[0], extractor.num_patches[1], -1)
            print(descriptors.shape, 'y la shape de los descriptores')
            dino_embeds.append(descriptors.cpu().detach())

        self.data = torch.stack(dino_embeds, dim=0)
        
        """
        print('The size of the dino features is', self.data.shape, self.data.device, self.data.dtype)
        #Save the DINO features
        sequence_id = self.cache_path.split('/')[-1]
        dino_save_path = os.pah.join('/home/lmur/Feature_Fields/Language-Aware-NeuralDiff/data/EPIC-Diff', sequence_id, 'features', 'full_dino_features.npy')
        print('Saving DINO features to', dino_save_path)
        np.save(dino_save_path, self.data.cpu().numpy())
        
        #pca = PCA(n_components=64, copy=False)
        N, H, W, C = self.data.shape
        self.data = self.data.view(-1, C).numpy()
        print("Features shape: ", self.data.shape)
        #X = pca.fit_transform(self.data)
        batch_size = 100000
        inc_pca = IncrementalPCA(n_components=64)
        for i in range(0, self.data.shape[0], batch_size):
            print("Fitting PCA on batch: ", i)
            inc_pca.partial_fit(self.data[i:i+batch_size])
        X = inc_pca.transform(self.data)
        print("Features shape (PCA): ", X.shape)
        self.data = torch.Tensor(X).view(N, H, W, 64)
        """          

    def __call__(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        img_scale = (
            self.data.shape[1] / self.cfg["image_shape"][0],
            self.data.shape[2] / self.cfg["image_shape"][1],
        )
        x_ind, y_ind = (img_points[:, 1] * img_scale[0]).long(), (img_points[:, 2] * img_scale[1]).long()
        return (self.data[img_points[:, 0].long(), x_ind, y_ind]).to(self.device)
