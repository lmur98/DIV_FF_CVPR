import torch
from torch import nn
#import tinycudann as tcnn
import numpy as np

class Lerf(nn.Module):
    def __init__(
        self,
        W = 256,
    ):
        super().__init__()          
        self.W = W

        """
        # clip layers
        self.clip_net = tcnn.Network(
            n_input_dims=self.hash_enc.n_output_dims + 1,
            n_output_dims=512,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.W,
                "n_hidden_layers": 4,
            },
        )

        # dino layers
        self.dino_net = tcnn.Network(
            n_input_dims=self.hash_enc.n_output_dims,
            n_output_dims=64,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.W,
                "n_hidden_layers": 1,
            },
        ) """ 

        self.clip_net = nn.Sequential(nn.Linear(self.W // 2, self.W // 2), nn.ReLU(),
                                      nn.Linear(self.W // 2, self.W // 2), nn.ReLU(),
                                      nn.Linear(self.W // 2, self.W // 2), nn.ReLU(),
                                      nn.Linear(self.W // 2, self.W // 2), nn.ReLU(),
                                      nn.Linear(self.W // 2, self.W // 2), nn.ReLU(),
                                      nn.Linear(self.W // 2, 128)) #MODIFIEDD!!

        # dino layers
        #self.dino_net = nn.Sequential(nn.Linear(self.W // 2, self.W // 2), nn.ReLU(), 
        #                              nn.Linear(self.W // 2, self.W // 2), nn.ReLU(), 
        #                              nn.Linear(self.W // 2, self.W // 2), nn.ReLU(), 
        #                              nn.Linear(self.W//2, 64)) #, nn.Tanh()) #MODIFIEDD!! 

    #def forward(self, xyz, scales):
    def forward(self, xyz):
        """
        Inputs:
            xyz: Sampled positions [N_Rays, N_lerf_samples, 3]
            scales: CLIP scale for each sample [N_Rays, N_lerf_samples, 1]
        """
        # clip and dino models
        hash_output = xyz.view(-1, 128)
        clip_output = self.clip_net(hash_output.float()).view(*xyz.shape[:-1], -1)
        #clip_output = clip_output / clip_output.norm(dim=-1, keepdim=True) # (N_Rays, N_lerf_samples, 128)
        #dino_output = self.dino_net(hash_output.float()).view(*xyz.shape[:-1], -1) # (N_Rays, N_lerf_samples, 64) #MODIFIEDD!!
        #dino_output = None
        #hash_output = hash_output.view(*xyz.shape[:-1], -1) # (N_Rays, N_lerf_samples, N_hash_outs)
        return clip_output

    """#def get_output_from_hashgrid(self, xyz, hash_output, scale):
    def get_output_from_hashgrid(self, encoding, scale):
        #hash_output = hash_output.view(-1, self.clip_net.n_input_dims -1
        #hash_output = encoding.view(-1, 128)
        #hash_output = self.hash_enc(encoding.view(-1, 3))
        hash_output = encoding.view(-1, 128)
        hash_output = self.lerf_mlp(hash_output.float())
        clip_output = self.clip_net(torch.cat([hash_output, scale.view(-1, 1)], dim=-1).float()).view(*encoding.shape[:-1], -1)   #MODIFIEDD!!
        #clip_output = clip_output / clip_output.norm(dim=-1, keepdim=True) # (N_Rays, N_lerf_samples, 128)
        return clip_output"""
        
class Video_Lerf(nn.Module):
    def __init__(
        self,
        W = 256,
    ):
        super().__init__()          
        self.W = W
        
        """levels = 16 # 16
        hash_size = 19 # 19
        start_res = 16
        end_res = 512
        growth = np.exp((np.log(end_res) - np.log(start_res)) / (levels - 1))
        self.hash_enc = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": levels,
                "n_features_per_level": 8,
                "log2_hashmap_size": hash_size,
                "base_resolution": start_res,
                "per_level_scale": growth,
            },
        )"""
        self.egovideo_net = nn.Sequential(nn.Linear(self.W // 2, self.W // 2), nn.ReLU(),
                                          nn.Linear(self.W // 2, self.W // 2), nn.ReLU(),
                                          nn.Linear(self.W // 2, self.W // 2), nn.ReLU(),
                                          nn.Linear(self.W // 2, self.W // 2), nn.ReLU(),
                                          nn.Linear(self.W // 2, 512))
        self.dino_net = nn.Sequential(nn.Linear(self.W//2, self.W // 2), nn.ReLU(), 
                                      nn.Linear(self.W//2, 64)) #, nn.Tanh()) #MODIFIEDD!! 

    
    def forward(self, xyz):
        #hash_output = self.hash_enc(xyz.view(-1, 3))
        hash_output = xyz.view(-1, 128)
        egovideo_output = self.egovideo_net(hash_output.float()).view(*xyz.shape[:-1], -1)
        dino_output = self.dino_net(hash_output.float()).view(*xyz.shape[:-1], -1)
        
        return egovideo_output, dino_output
    

