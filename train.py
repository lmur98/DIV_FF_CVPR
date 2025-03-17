import os
from collections import defaultdict

import matplotlib.pyplot as plt
import pytorch_lightning
import torch
from torch.optim import Adam, RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import dataset
import model
import extra_utils
from evaluation.metrics import *
from loss import Loss
from opt import get_opts
from extra_utils import *
from einops import rearrange

def on_load_checkpoint(self, checkpoint: dict) -> None:
    state_dict = checkpoint["state_dict"]
    model_state_dict = self.state_dict()
    is_changed = False
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print(f"Skip loading parameter: {k}, "
                            f"required shape: {model_state_dict[k].shape}, "
                            f"loaded shape: {state_dict[k].shape}")
                state_dict[k] = model_state_dict[k]
                is_changed = True
        else:
            print(f"Dropping parameter {k}")
            is_changed = True

    if is_changed:
        print('removing optimiser states and LREmbedding')
        checkpoint.pop("optimizer_states", None)


def init_model(ckpt_path, dataset=None, hparams=None, vid=None):
    if (ckpt_path is not None):
        ckpt = torch.load(ckpt_path, map_location="cpu")

    model = NeuralDiffSystem(
        hparams, train_dataset=dataset, val_dataset=dataset
    )
    if (ckpt_path is not None):
        try:
            model.load_state_dict(ckpt["state_dict"])
        except Exception as e:
            warnings.warn('Some model components were not loaded from checkpoint. \
                Loading with `strict=False`.'.replace('  ', '')
            )
            model.load_state_dict(ckpt["state_dict"], strict=False)

    return model

class NeuralDiffSystem(pytorch_lightning.LightningModule):
    def __init__(self, hparams, train_dataset=None, val_dataset=None):
        super().__init__()
        print(hparams)
        self.my_hparams = hparams
        if self.my_hparams.deterministic:
            extra_utils.set_deterministic()

        # for avoiding reinitialization of dataloaders when debugging/using notebook
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.loss = Loss()

        self.models_to_train = []
        self.embedding_xyz = model.PosEmbedding(
            hparams.N_emb_xyz - 1, hparams.N_emb_xyz
        )
        self.embedding_dir = model.PosEmbedding(
            hparams.N_emb_dir - 1, hparams.N_emb_dir
        )

        self.embeddings = {
            "xyz": self.embedding_xyz,
            "dir": self.embedding_dir,
        }

        self.embedding_t = model.LREEmbedding(
            N=hparams.N_vocab, D=hparams.N_tau, K=hparams.lowpass_K
        )
        self.embeddings["t"] = self.embedding_t
        self.models_to_train += [self.embedding_t]

        self.embedding_a = torch.nn.Embedding(hparams.N_vocab, hparams.N_a)
        self.embeddings["a"] = self.embedding_a
        self.models_to_train += [self.embedding_a]

        self.nerf_coarse = model.NeuralDiff(
            "coarse",
            in_channels_xyz=6 * hparams.N_emb_xyz + 3,
            in_channels_dir=6 * hparams.N_emb_dir + 3,
            W=hparams.model_width,
        )
        self.models = {"coarse": self.nerf_coarse}
        if hparams.N_importance > 0:
            self.nerf_fine = model.NeuralDiff(
                "fine",
                in_channels_xyz=6 * hparams.N_emb_xyz + 3,
                in_channels_dir=6 * hparams.N_emb_dir + 3,
                encode_dynamic=True,
                in_channels_a=hparams.N_a,
                in_channels_t=hparams.N_tau,
                beta_min=hparams.beta_min,
                W=hparams.model_width,
            )
            self.models["fine"] = self.nerf_fine

        # Add LERF model
        if (hparams.use_clip):
            self.static_lerf = model.Lerf(W = hparams.lerf_model_width,).to(torch.device("cuda:0"))
            self.models["static_lerf"] = self.static_lerf
            self.transient_lerf = model.Lerf(W = hparams.lerf_model_width,).to(torch.device("cuda:0"))
            self.models["transient_lerf"] = self.transient_lerf
            self.person_lerf = model.Lerf(W = hparams.lerf_model_width,).to(torch.device("cuda:0"))
            self.models["person_lerf"] = self.person_lerf
        
        if hparams.video_active:
            #self.video_lerf = model.Video_Lerf(W = hparams.lerf_model_width,).to(torch.device("cuda:0"))
            #self.models["video_lerf"] = self.video_lerf
            self.static_video = model.Video_Lerf(W = hparams.lerf_model_width,).to(torch.device("cuda:0"))
            self.models["static_video"] = self.static_video
            self.transient_video = model.Video_Lerf(W = hparams.lerf_model_width,).to(torch.device("cuda:0"))
            self.models["transient_video"] = self.transient_video
            self.person_video = model.Video_Lerf(W = hparams.lerf_model_width,).to(torch.device("cuda:0"))
            self.models["person_video"] = self.person_video
        
        self.models_to_train += [self.models]
        self.automatic_optimization = False

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, rays, ts, clip_scales, test_time=False, disable_perturb=False, image_encoder=None):
        perturb = 0 if test_time or disable_perturb else self.my_hparams.perturb
        noise_std = 0 if test_time or disable_perturb else self.my_hparams.noise_std
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.my_hparams.chunk):
            rendered_ray_chunks = model.render_rays(
                models=self.models,
                embeddings=self.embeddings,
                rays=rays[i : i + self.my_hparams.chunk],
                ts=ts[i : i + self.my_hparams.chunk],
                clip_scales=clip_scales[i : i + self.my_hparams.chunk],
                img_h=self.img_h,
                intrinsics_fy=self.intrinsics_fy,
                N_samples=self.my_hparams.N_samples,
                perturb=perturb,
                noise_std=noise_std,
                N_importance=self.my_hparams.N_importance,
                N_lerf_samples=self.my_hparams.N_lerf_samples,
                chunk=self.my_hparams.chunk,
                hp=self.my_hparams,
                test_time=test_time,
                image_encoder=image_encoder,
            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage, reset_dataset=False):
        kwargs = {"root": self.my_hparams.root}
        kwargs["vid"] = self.my_hparams.vid
        print("Loading datasets...")
        if (self.train_dataset is None and self.val_dataset is None) or reset_dataset:
            self.train_dataset = dataset.EPICDiff(split="train", **kwargs)
            self.val_dataset = dataset.EPICDiff(split="val", **kwargs)
            self.img_h = self.train_dataset.img_h
            self.intrinsics_fy = self.train_dataset.K[1][1]

    def configure_optimizers(self):
        eps = 1e-8
        self.optimizer = Adam(
            get_parameters(self.models_to_train),
            lr=self.my_hparams.lr,
            eps=eps,
            weight_decay=self.my_hparams.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.my_hparams.num_epochs, eta_min=eps
        )
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.my_hparams.num_workers,
            batch_size=self.my_hparams.batch_size,
            pin_memory=True,
        )

    def val_dataloader(self):
        # batch_size=1 for validating one image (H*W rays) at a time
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.my_hparams.num_workers,
            batch_size=1,
            pin_memory=True,
        )
        
    def freeze_geometry_layers(self):
        models_to_freeze = ["coarse", "fine"]
        for model in self.models:
            if model in models_to_freeze:
                for param in self.models[model].parameters():
                    param.requires_grad = False

        
    def unfreeze_geometry_layers(self):
        for model in self.models:
            for param in self.models[model].parameters():
                param.requires_grad = True
                
    def count_trainable_parameters(self):
        trainable_params = 0
        for model_name, submodel in self.models.items():
            for name, param in submodel.named_parameters():
                if param.requires_grad:
                    trainable_params += param.numel()  # numel() devuelve el número total de elementos
        return trainable_params

    def training_step(self, batch, batch_nb):
        opt = self.optimizers()
        
        if self.my_hparams.use_clip:
            global_step = self.global_step
            if global_step == 0:
                self.freeze_geometry_layers()
                print('There are', self.count_trainable_parameters(), 'trainable parameters')
                
            elif global_step == 5000: #The loss starts to saturate after 10k steps, which is half epoch aprox
                self.unfreeze_geometry_layers()
                print('There are', self.count_trainable_parameters(), 'trainable parameters')
            
        opt.zero_grad()
        rays, rgbs, ts, indices = batch["rays"], batch["rgbs"], batch["ts"], batch["indices"]
        #clip_gt, clip_scales = self.train_dataset.clip_interpolator(indices.to(torch.device("cpu")))
        SAM_clip_gt, SAM_ignore_masks = self.train_dataset.SAM_CLIP(indices.to(torch.device("cpu")))
        clip_scales = torch.ones_like(ts) # (H*W)
        # print("GT: ", clip_gt)
        dino_gt = self.train_dataset.dinov2_dataloader(indices.to(torch.device("cpu"))).to(torch.device("cuda:0"))
        
        if self.my_hparams.video_active:
            egovideo_gt_IntHots, egovideo_ignore_masks = self.train_dataset.egovideo_dataloader_IntHots(indices.to(torch.device("cpu"))) #(B, T, C)
            egovideo_gt_IntHots = egovideo_gt_IntHots.to(torch.device("cuda:0"))
            egovideo_ignore_masks = egovideo_ignore_masks.to(torch.device("cuda:0"))
            egovideo_gt_PATCH = self.train_dataset.egovideo_dataloader_PATCH(indices.to(torch.device("cpu"))) #(B, T, C)
        else:
            egovideo_gt_PATCH, egovideo_gt_IntHots, egovideo_ignore_masks = None, None, None
        
        results = self(rays, ts, clip_scales)
        # print("RES: ", results["clip"])
        loss_d = self.loss(results, rgbs, SAM_clip_gt.float(), SAM_ignore_masks, dino_gt,  egovideo_gt_PATCH, egovideo_gt_IntHots, egovideo_ignore_masks) # Important, GT dtype is half, we need to convert it to float
        loss = sum(l for l in loss_d.values())

        with torch.no_grad():
            psnr_ = psnr(results["rgb_fine"], rgbs)

        loss.backward()
        opt.step()

        self.log("lr", self.optimizer.param_groups[0]["lr"])
        self.log("train/loss", loss)
        for k, v in loss_d.items():
            self.log(f"train/{k}", v, prog_bar=True)
        self.log("train/psnr", psnr_, prog_bar=True)

        return loss

    def render(self, sample, img_h, fy, image_encoder, t=None, device=None):
        self.img_h = img_h
        self.intrinsics_fy = fy
        rays, rgbs, ts = (
            sample["rays"].cuda(),
            sample["rgbs"].cuda(),
            sample["ts"].cuda(),
        )

        if t is not None:
            if type(t) is torch.Tensor:
                t = t.cuda()
            ts = torch.ones_like(ts) * t

        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        ts = ts.squeeze()  # (H*W)
        clip_scales = torch.ones_like(ts) # (H*W)
        clip_scales = rearrange(clip_scales, "n1 -> n1 1 1")
        with torch.no_grad():
            results = self(rays, ts, clip_scales, test_time=True, image_encoder=image_encoder)

        if device is not None:
            for k in results:
                results[k] = results[k].to(device)

        return results

    def validation_step(self, batch, batch_nb, is_debug=False):
        rays, rgbs, ts, indices = batch["rays"], batch["rgbs"], batch["ts"], batch["indices"]

        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        ts = ts.squeeze()  # (H*W)
        indices = indices.squeeze().to(torch.device("cpu"))
        #clip_gt, clip_scales = self.val_dataset.clip_interpolator(indices)
        SAM_clip_gt, SAM_ignore_masks = self.train_dataset.SAM_CLIP(indices.to(torch.device("cpu")))
        clip_scales = torch.ones_like(ts) # (H*W)
        dino_gt = self.val_dataset.dinov2_dataloader(indices).to(torch.device("cuda:0"))
        
        if self.my_hparams.video_active:
            egovideo_gt_IntHots, egovideo_ignore_masks = self.val_dataset.egovideo_dataloader_IntHots(indices.to(torch.device("cpu"))) #(B, T, C)
            egovideo_gt_IntHots = egovideo_gt_IntHots.to(torch.device("cuda:0"))
            egovideo_ignore_masks = egovideo_ignore_masks.to(torch.device("cuda:0"))
            egovideo_gt_PATCH = self.val_dataset.egovideo_dataloader_PATCH(indices.to(torch.device("cpu"))) #(B, T, C)

        else:
            egovideo_gt_PATCH, egovideo_gt_IntHots, egovideo_ignore_masks = None, None, None
        # disable perturb (used during training), but keep loss for tensorboard
        results = self(rays, ts, clip_scales, disable_perturb=True)
        loss_d = self.loss(results, rgbs, SAM_clip_gt, SAM_ignore_masks, dino_gt, egovideo_gt_PATCH, egovideo_gt_IntHots, egovideo_ignore_masks)
        loss = sum(l for l in loss_d.values())
        log = {"val_loss": loss}

        if batch_nb == 0:
            WH = batch["img_wh"].view(1, 2)
            W, H = WH[0, 0].item(), WH[0, 1].item()
            img = (
                results["rgb_fine"].view(H, W, 3)[:, :, :3].permute(2, 0, 1).cpu()
            )  # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
            depth = visualize_depth(results["depth_fine"].view(H, W))  # (3, H, W)
            stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)
            # if self.logger is not None:
            #     self.logger.experiment.add_images(
            #         "val/GT_pred_depth", stack, self.global_step
            #     )

        psnr_ = psnr(results["rgb_fine"], rgbs)
        log["val_psnr"] = psnr_

        if is_debug:
            # then visualise in jupyter
            log["images"] = stack
            log["results"] = results

            f, p = plt.subplots(1, 3, figsize=(15, 15))
            for i in range(3):
                im = stack[i]
                p[i].imshow(im.permute(1, 2, 0).cpu())
                p[i].axis("off")
            plt.show()

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        mean_psnr = torch.stack([x["val_psnr"] for x in outputs]).mean()

        self.log("val/loss", mean_loss)
        self.log("val/psnr", mean_psnr, prog_bar=True)


def init_trainer(hparams, logger=None, checkpoint_callback=None):
    if checkpoint_callback is None:
        checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(
            filepath=os.path.join(f"ckpts/{hparams.exp_name}", "{epoch:d}"),
            monitor="val/psnr",
            mode="max",
            save_top_k=-1,
        )
    project_name = hparams.vid + "_width_" + str (hparams.model_width) + "_clip_" + str (hparams.use_clip) + "_CLIPwidth_" + str (hparams.model_width) 
    logger = pytorch_lightning.loggers.WandbLogger(project=project_name)
    # logger = pytorch_lightning.loggers.TestTubeLogger(
    #     save_dir="logs",
    #     name=hparams.exp_name,
    #     debug=False,
    #     create_git_tag=False,
    #     log_graph=False,
    # )

    trainer = pytorch_lightning.Trainer(
        max_epochs=hparams.num_epochs,
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=hparams.ckpt_path,
        logger=logger,
        weights_summary=None,
        progress_bar_refresh_rate=50,
        gpus=hparams.num_gpus,
        accelerator="ddp" if hparams.num_gpus > 1 else None,
        num_sanity_val_steps=1,
        benchmark=True,
        profiler="simple" if hparams.num_gpus == 1 else None,
    )

    return trainer


def main(hparams):
    system = NeuralDiffSystem(hparams)
    #weights_path = '/home/lmur/Feature_Fields/ckpts_official_NeuralDiff/ckpts/rel/P01_01/epoch=9.ckpt'
    #weights = torch.load(weights_path, map_location='cpu')
    #system.load_state_dict(weights['state_dict'], strict=True)
    trainer = init_trainer(hparams)
    trainer.fit(system)


if __name__ == "__main__":
    hparams = get_opts()
    print(hparams)
    main(hparams)
