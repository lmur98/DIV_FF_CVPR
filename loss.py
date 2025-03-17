import torch
from torch import nn


class Loss(nn.Module):
    """
    Name abbreviations:
        c_l: coarse color loss
        f_l: fine color loss
        b_l: beta loss
        s_l: sigma loss
    """

    def __init__(self, lambda_u=0.01):
        """
        lambda_u: regularisation for sigmas.
        """
        super().__init__()
        self.lambda_u = lambda_u

    def forward(self, inputs, targets, clip_targets, CLIP_ignore_mask, dino_targets, video_targets_PATCH, video_targets_IntHots, video_ignore_mask, is_nerf_eval=False):
        ret = {}
        ret["c_l"] = 0.5 * ((inputs["rgb_coarse"] - targets) ** 2).mean()
        if "rgb_fine" in inputs:
            ret["f_l"] = (
                (inputs["rgb_fine"] - targets) ** 2
                / (2 * inputs["beta"].unsqueeze(1) ** 2)
            ).mean()
            ret["b_l"] = torch.log(inputs["beta"]).mean()
            ret["s_l"] = self.lambda_u * inputs["transient_sigmas"].mean()
            ret["s_l"] = ret["s_l"] + self.lambda_u * inputs["person_sigmas"].mean()

        # LERF loss
        # print("CLIP GT: ", clip_targets)
        # print("CLIP OUTS: ", inputs["clip"])
        if "clip" in inputs:
            loss = torch.nn.functional.huber_loss(inputs["clip"], clip_targets, delta=1.25, reduction="none")
            masked_loss = loss * CLIP_ignore_mask.unsqueeze(1).float()
            ret["clip_l"] = masked_loss.sum(dim=1).mean()
            
        if "dino" in inputs:
            dino_distances = ((inputs["dino"] - dino_targets) ** 2).sum(dim=1)
            ret["dino_l"] = dino_distances.mean() #Args distill lamda    

        if 'video' in inputs:
            dist_video_IntHots = torch.nn.functional.huber_loss(inputs["video"], video_targets_IntHots, delta = 1.25, reduction = 'none')
            video_ignore_mask = video_ignore_mask.unsqueeze(1).float()
            dist_video_PATCH = torch.nn.functional.huber_loss(inputs["video"], video_targets_PATCH, delta = 1.25, reduction = 'none')
            
            ret["video"] = (dist_video_PATCH).sum(dim=1).mean() #+ (video_ignore_mask * dist_video_IntHots).sum(dim=1).mean() #+ dist_video_PATCH.sum(dim=1).mean()
            
        return ret
