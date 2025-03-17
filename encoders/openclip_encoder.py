from dataclasses import dataclass, field
from typing import Tuple, Type

import torch
import torchvision
import os

import sys
sys.path.append('/home/lmur/FUSION_FIELDS/Lorenzo_Feature_Fields_v2/autoencoder')
from autoencoder.model import Autoencoder

try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"

from encoders.image_encoder import BaseImageEncoder
# from nerfstudio.viewer.viewer_elements import ViewerText

import sys
sys.path.append('/home/lmur/FUSION_FIELDS')
from EgoVideo.backbone.model.setup_model import build_video_model, build_text_model

class OpenCLIPNetwork(BaseImageEncoder):
    def __init__(self, clip_model_type: str = "ViT-B-16", clip_model_pretrained: str = "laion2b_s34b_b88k", clip_n_dims: int = 512, negatives: Tuple[str] = ("object", "things", "stuff", "hands")): 
        super().__init__()
        self.clip_model_type = clip_model_type
        self.clip_model_pretrained = clip_model_pretrained
        self.clip_n_dims = clip_n_dims
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
            self.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        self.tokenizer = open_clip.get_tokenizer(self.clip_model_type)
        self.model = model.to("cuda")

        # self.positive_input = ViewerText("LERF Positives", "", cb_hook=self.gui_cb)

        self.positives = [""] # self.positive_input.value.split(";")
        self.negatives = negatives
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

        autoencoder_weights_path = os.path.join('/home/lmur/FUSION_FIELDS/Lorenzo_Feature_Fields_v2/autoencoder/ckpt', 'P13_03_SAM2_bbox128', 'best_ckpt.pth')
        checkpoint = torch.load(autoencoder_weights_path, map_location='cuda:0')
        SAM_autoencoder = Autoencoder(encoder_hidden_dims = [256, 128], decoder_hidden_dims = [196, 256, 256, 512]).to("cuda:0")
        SAM_autoencoder.load_state_dict(checkpoint, strict = True)
        SAM_autoencoder.eval()
        self.IMAGE_SAM_autoencoder = SAM_autoencoder

        #autoencoder_weights_path = os.path.join('/home/lmur/FUSION_FIELDS/Lorenzo_Feature_Fields_v2/autoencoder/ckpt', 'P01_VIDEO_128', 'best_ckpt.pth')
        #checkpoint = torch.load(autoencoder_weights_path, map_location='cuda:0')
        #SAM_autoencoder = Autoencoder(encoder_hidden_dims = [256, 128], decoder_hidden_dims = [196, 256, 256, 512]).to("cuda:0")
        #SAM_autoencoder.load_state_dict(checkpoint, strict = True)
        #SAM_autoencoder.eval()
        #self.VIDEO_SAM_autoencoder = SAM_autoencoder

        self.text_encoder, self.video_tokenizer = build_text_model(ckpt_path = '/home/lmur/FUSION_FIELDS/EgoVideo/ckpt_4frames.pth', num_frames = 4)
        self.text_encoder.eval().to("cuda:0").to(torch.float16)
        self.video_negatives = ["general task", "undistic movement", "unclear action", "background"] #["object", "things", "stuff", "texture"] #["#C C does something undeffined", "#C C is moving", "#C C does an unclear action", "Nothing happens"]#["#C C is moving", "#C does an unclear action", "#C C does a general task", "Nothing happens"]
        with torch.no_grad():
            self.video_neg_embeds = []
            for phrase in self.video_negatives:
                text = self.video_tokenizer(phrase, max_length=20,truncation=True,padding = 'max_length',return_tensors = 'pt')
                text_input = text.input_ids.to("cuda:0")
                mask = text.attention_mask.to("cuda:0")
                _, text_features = self.text_encoder(None, text_input, mask)
                self.video_neg_embeds.append(text_features)
            self.video_neg_embeds = torch.cat(self.video_neg_embeds, dim = 0)
        
        
    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.clip_model_type, self.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.clip_n_dims
    
    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        
    def set_video_positives(self, text_list):
        self.video_positives = text_list    
        with torch.no_grad():
            self.video_pos_embeds = []
            for phrase in self.positives:
                text = self.video_tokenizer(phrase, max_length=20,truncation=True,padding = 'max_length',return_tensors = 'pt')
                text_input = text.input_ids.to("cuda:0")
                mask = text.attention_mask.to("cuda:0")
                _, text_features = self.text_encoder(None, text_input, mask)
                self.video_pos_embeds.append(text_features)
            self.video_pos_embeds = torch.cat(self.video_pos_embeds, dim = 0)

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
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
            :, 0, :
        ]
        
    def get_video_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        print('COMPARANDO LA VIDEO RELEVANCY')
        phrases_embeds = torch.cat([self.video_pos_embeds, self.video_neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
            :, 0, :
        ]
        
        

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)
