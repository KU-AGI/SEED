import os
from typing import Any, List
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist

import json

import hydra
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, seed_everything
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import pyrootutils

from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
import torch.nn.functional as F
from pytorch_lightning.strategies import DDPStrategy, DDPFullyShardedStrategy
from einops import rearrange
import transformers
from transformers import CLIPVisionModelWithProjection, CLIPTokenizer

from pytorch_lightning import loggers as pl_loggers
from functools import partial
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.utilities import grad_norm

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt

from models.seed_qformer.vit import Block
from models.seed_llama_tokenizer import ImageTokenizer

from datamodules.seed_llama_datamodule import SEEDDataModule

from calculate_clip_score import calculate_clip_s_for_folder
from utils.config import build_config

from lavis.models import load_model
from lavis.common.dist_utils import is_dist_avail_and_initialized

from diffusers import DPMSolverMultistepScheduler

from models.slot_attention.slot_attn import MultiHeadSTEVESA

from scipy.optimize import linear_sum_assignment
from transformers import AutoTokenizer, AutoModelForCausalLM

from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DiffusionPipeline,
)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

BOI_TOKEN = "<img>"
EOI_TOKEN = "</img>"
IMG_TOKEN = "<img_{:05d}>"

IMG_FLAG = "<image>"
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 8192
IMAGE_ID_SHIFT = 32000

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]

class DINOBackbone(nn.Module):
    def __init__(self, dinov2):
        super().__init__()
        self.dinov2 = dinov2

    def forward(self, x):
        enc_out = self.dinov2.forward_features(x)
        return rearrange(
            enc_out["x_norm_patchtokens"], 
            "b (h w ) c -> b c h w",
            h=int(np.sqrt(enc_out["x_norm_patchtokens"].shape[-2]))
        )

class SlotTrainingWrapper(LightningModule):
    """Training wrapper for Slot

    Args:
        LightningModule (cfg, model): model should be ImageTokenizer
    """    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        diffusion_precision = "fp16" if cfg.optimizer.diffusion_precision else "fp32"
        pretrained_model_name = "stabilityai/stable-diffusion-2-1"
        self.diffusion_model = StableDiffusionPipeline.from_pretrained(pretrained_model_name,
                                                                torch_dtype=dict(fp16=torch.float16, fp32=torch.float32)[diffusion_precision])
        # Change to DDPMScheduler
        self.diffusion_model.scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name, subfolder="scheduler", torch_dtype=dict(fp16=torch.float16, fp32=torch.float32)[diffusion_precision])

        # For diffusion DDP
        self.feature_extractor = self.diffusion_model.feature_extractor

        if self.cfg.stage2.unclip:
            self.image_encoder = self.diffusion_model.image_encoder
            self.image_normalizer = self.diffusion_model.image_normalizer
            self.image_noising_scheduler = self.diffusion_model.image_noising_scheduler
        else:
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
            self.image_normalizer = None
            self.image_noising_scheduler = None
        if self.diffusion_model.text_encoder is not None:
            self.clip_tokenizer = self.diffusion_model.tokenizer
        else:
            self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        if self.diffusion_model.text_encoder is not None:
            self.text_encoder = self.diffusion_model.text_encoder
        self.unet = self.diffusion_model.unet
        self.scheduler = self.diffusion_model.scheduler
        self.vae = self.diffusion_model.vae

        # Scheduler for validation
        scheduler_args = {}
        if "variance_type" in self.scheduler.config:
            variance_type = self.scheduler.config.variance_type
            
            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type
            
        self.val_schduler = DPMSolverMultistepScheduler.from_config(
            self.scheduler.config, **scheduler_args
        )

        # For logging
        self.stage = cfg.experiment.stage
        
        # Define slot attention
        slot_attn_config = MultiHeadSTEVESA.load_config(cfg.slot_cfg_path)
        self.slot_attention = MultiHeadSTEVESA.from_config(slot_attn_config)            

        self.out_linear = self.slot_attention.out_linear
        self.out_linear_4096 = nn.Linear(768, 4096)

        self.transform_256 = transforms.Resize((256, 256), antialias=True)

        self.normalize_diffusion = transforms.Normalize(mean=[0.5], std=[0.5])
        self.normalize_vit = transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                            )

        self.save_path = None

        # Load backbone
        dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.backbone = DINOBackbone(dinov2).eval()

        self.pos_embed = nn.Parameter(torch.zeros(1, 32, 768))
        self.causal_transformer = nn.ModuleList([
            Block(dim=768,
                    num_heads=12,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6)) for _ in range(1)
        ])

        llama_ver = "meta-llama/Llama-2-7b-chat-hf"
        self.llama_embedding = AutoModelForCausalLM.from_pretrained(llama_ver).model.embed_tokens
        self.llama_embedding = self.llama_embedding.half().eval()
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_ver)

    def setup(self, stage):
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Freeze llama embedding
        for param in self.llama_embedding.parameters():
            param.requires_grad = False

        # Diffusion frozen
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        if self.image_normalizer is not None:
            for param in self.image_normalizer.parameters():
                param.requires_grad = False
        if self.text_encoder is not None:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        # Freeze VAE
        for param in self.vae.parameters():
            param.requires_grad = False

        if hasattr(self.cfg.stage1, "unfreeze_unet"):
            # casting to float32
            self.unet = self.unet.to(dtype=torch.float32)
        for name, param in self.unet.named_parameters():
            # If self.cfg.stage1.unfreeze_unet exists, unfreeze cross attention layer
            if hasattr(self.cfg.stage1, "unfreeze_unet") and self.cfg.stage1.unfreeze_unet:
                if any(x in name for x in ["attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out"]):
                    print(f"Unfreeze {name}")
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False

        ## make dump folder
        os.makedirs(self.cfg.result_file_path, exist_ok=True)

    def save_config(self):
        config_save_path = os.path.join(self.logger.log_dir, "config.yaml")
        with open(config_save_path, "w") as f:
            json.dump(self.cfg, f, indent=4)
    
    def get_clip_text_embedding(self, batch_text):
        """CLIP text embedding

        Args:
            batch_text (List): List contains text. [b, 32]

        Returns:
            float: clip text embedding [b, 1024]
        """        
        gt_text_clip_embeddings = []
        with torch.no_grad():
            for idx in range(len(batch_text)):
                gt_text_clip_embeddings.append(
                    self.clip_tokenizer(batch_text[idx]).squeeze().to(self.device)
                )
            gt_text_clip_embeddings = torch.stack(gt_text_clip_embeddings, dim=0)

            # gt_img_clip_embeddings = self.model_clip.encode_image(batch.img.to(self.device))
            gt_text_clip_embeddings = self.image_encoder.encode_text(
                gt_text_clip_embeddings.to(self.device)
            )
        return gt_text_clip_embeddings
    
    def get_clip_img_embedding(self, batch_img):
        """CLIP image embedding

        Args:
            batch_img (torch.Tensor): Image tensor [b, 3, 224, 224]

        Returns:
            float: clip image embedding [b, 1024]
        """        
        return self.image_encoder(batch_img).image_embeds.to(self.device)
 
    def all_gather_with_grad(self, tensors):
        """
        Performs all_gather operation on the provided tensors.
        Graph remains connected for backward grad computation.
        """
        # Queue the gathered tensors
        world_size = torch.distributed.get_world_size()
        # There is no need for reduction in the single-proc case
        if world_size == 1:
            return tensors

        # tensor_all = GatherLayer.apply(tensors)
        tensor_all = GatherLayer.apply(tensors)

        return torch.cat(tensor_all, dim=0)
    
    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        # if use distributed training
        if not is_dist_avail_and_initialized():
            return tensor

        tensors_gather = [
            torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output

    def compute_cost_matrix(self, embeddings_1, embeddings_2):
        # Normalize embeddings to compute cosine similarity
        embeddings_1_norm = F.normalize(embeddings_1, p=2, dim=-1)
        embeddings_2_norm = F.normalize(embeddings_2, p=2, dim=-1)
        
        # Cosine similarity is computed as the dot product of normalized vectors
        cost_matrix = torch.matmul(embeddings_1_norm, embeddings_2_norm.transpose(-1, -2))
        
        # Convert cosine similarity to a cost (negative similarity)
        cost_matrix = -cost_matrix  # We want to minimize the negative similarity, i.e., maximize similarity
        
        return cost_matrix  # [32, n]

    def hungarian_matching(self, cost_matrix):
        cost = cost_matrix.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)

        return torch.tensor(row_ind, device=self.device), torch.tensor(col_ind, device=self.device)

    def bipartite_matching_loss(self, cost_matrix, indicies):
        row_ind, col_ind = indicies
        total_cost = cost_matrix[row_ind, col_ind].mean()
        return total_cost

    def comput_contrastive_loss(self, slots, text):
        visual_embedding_batch = self.out_linear_4096(slots)  # shape: [b, 32, 4096]
        text_tokens_list = self.llama_tokenizer(text).input_ids

        total_loss = 0.0

        for visual_embedding, text_token in zip(visual_embedding_batch, text_tokens_list):
            text_token = torch.tensor(text_token).to(self.device)
            text_embedding = self.llama_embedding(text_token)  # shape: [b, n, 4096] 

            # Compute cost matrix based on cosine similarity
            cost_matrix = self.compute_cost_matrix(visual_embedding, text_embedding) # shape : [b, 32, n]

            # Optimal matching using Hungarian algorithm
            indices = self.hungarian_matching(cost_matrix) # [(row_ind, col_ind), ...]

            # Compute bipartite matching loss
            loss = self.bipartite_matching_loss(cost_matrix, indices) # scalar
            total_loss += loss

        total_loss /= len(visual_embedding_batch)

        return total_loss

    def apply_causal_transformer(self, slots):
        pos_embed_applied_slot = slots + self.pos_embed.repeat(slots.size(0), 1, 1)
        # Apply Causal Transformer
        for blk in self.causal_transformer:
            pos_embed_applied_slot = blk(pos_embed_applied_slot, use_causal_mask=True)
        return pos_embed_applied_slot

    def get_diffusion_text_contrastive_loss(self, batch, batch_idx: int, is_validation=False):
        pixel_values = self.transform_256(batch[0])
        pixel_values = self.normalize_diffusion(pixel_values)

        # Convert images to latent space
        model_input = self.vae.encode(pixel_values).latent_dist.sample()
        model_input = model_input * self.vae.config.scaling_factor

        # Sample noise that we'll add to the model input
        noise = torch.randn_like(model_input)

        bsz, channels, height, width = model_input.shape
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
        )
        timesteps = timesteps.long()

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = self.scheduler.add_noise(
            model_input, noise, timesteps)

        # timestep is not used, but should we?
        backbone_input = self.normalize_vit(batch[0])
        feat = self.backbone(backbone_input)

        slots, attn = self.slot_attention(feat[:, None], apply_out_linear=False)  # for the time dimension
        # Remove T dimension
        slots = rearrange(slots, "bs 1 n_s n_d -> bs n_s n_d")

        causal_slots = self.apply_causal_transformer(slots)

        slots_1024 = self.out_linear(causal_slots)

        # Predict the noise residual
        model_pred = self.unet(
            noisy_model_input, timesteps, slots_1024,
        ).sample

        # Get the target for loss depending on the prediction type
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(
                model_input, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler.config.prediction_type}")

        # Compute instance loss
        loss_diffusion = F.mse_loss(model_pred.float(),
                            target.float(), reduction="mean")

        ###============== Image-text Contrastive ===================###
        # Image feat from slot attention
        loss_itc = self.comput_contrastive_loss(causal_slots, batch[1])

        loss = loss_diffusion + loss_itc
        # DEBUG : Only test for causal transformer
        # loss = loss_diffusion

        cur_stage = "train" if not is_validation else "val"
        self.log(
            f"{cur_stage}/loss_diffusion",
            loss_diffusion,
            on_step=False if is_validation else True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            f"{cur_stage}/loss_itc",
            loss_itc,
            on_step=False if is_validation else True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            f"{cur_stage}/loss",
            loss,
            on_step=False if is_validation else True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss, slots_1024

    def on_train_start(self):
        print(f"\n====Traing Stage {self.stage}====")
        if self.stage == 2 and self.cfg.stage2.bypass_codebook:
            print("\n====Bypass codebook====")

        print("Save config")
        self.save_config()
    
    def training_step(self, batch, batch_idx: int):
        if len(batch) == 3:
            image, text, image_id = batch
        elif len(batch) == 2:
            image, text = batch

        self.B = image.shape[0]

        loss, _ = self.get_diffusion_text_contrastive_loss(batch, batch_idx)
        
        return loss

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        # {'grad_2.0_norm/weight': 0.0003, 'grad_2.0_norm/bias': 0.0, 'grad_2.0_norm_total': 0.0003}
        norm_slot = grad_norm(self.slot_attention, norm_type=2)
        for norm in norm_slot.keys():
            self.logger.experiment.add_scalar(
                f"grad_norm/slot_attention/{norm}",
                norm_slot[norm],
                global_step=self.global_step,
            )
        norm_unet = grad_norm(self.unet, norm_type=2)
        for norm in norm_unet.keys():
            self.logger.experiment.add_scalar(
                f"grad_norm/unet/{norm}",
                norm_unet[norm],
                global_step=self.global_step,
            )
    
    def on_validation_epoch_start(self):
        return

    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int, save_path=None):
        if self.logger is not None and isinstance(self.logger, pl.loggers.TensorBoardLogger):
            tb_log_dir = self.logger.log_dir
        else:
            tb_log_dir = self.cfg.result_file_path  # Fallback directory if logger is not set

        self.save_path = f"{tb_log_dir}/reconstructed_images"
        os.makedirs(self.save_path, exist_ok=True)
        
        self.save_path = f"{tb_log_dir}/reconstructed_images/epoch_{self.current_epoch}"
        os.makedirs(self.save_path, exist_ok=True)

        img, text, image_id = batch
        loss, slots = self.get_diffusion_text_contrastive_loss(batch, batch_idx, is_validation=True)

        # Change validation scheduler
        cur_scheduler = self.scheduler
        self.diffusion_model.scheduler = self.val_schduler

        reconstructed_images = self.diffusion_model(
            prompt_embeds=slots,
            height=256,
            width=256,
            guidance_scale=1.3,
            # num_inference_steps=25,
            num_inference_steps=100,
        ).images

        # Return it to original scheduler
        self.diffusion_model.scheduler = cur_scheduler

        for img, cur_id in zip(reconstructed_images, image_id):
            # save PIL image to save_path
            img.save(f"{self.save_path}/{cur_id}")

    def on_validation_epoch_end(self):
        if self.logger is not None and isinstance(self.logger, pl_loggers.TensorBoardLogger):
            tb_log_dir = self.logger.log_dir
        else:
            tb_log_dir = self.cfg.result_file_path

        original_image_dir = self.cfg.dataset.val_config.root_dir
        generated_image_dir = self.save_path
        clip_score = calculate_clip_s_for_folder(original_image_dir, generated_image_dir)
        
        print(f"clip score: {clip_score}")
        self.log_dict({
            'clip_score': clip_score,
        },on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        self.log(
            "clip_score_coco_karpathy",
            clip_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-8)
        lr = self.cfg.optimizer.max_lr
        betas = (self.cfg.hyperparameters.beta_1, self.cfg.hyperparameters.beta_2)
        weight_decay = self.cfg.hyperparameters.weight_decay

        num_parameters = 0
        p_wd, p_non_wd = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
            num_parameters += p.data.nelement()
        # for n, p in self.slot_attention.named_parameters():
        #     if not p.requires_grad:
        #         continue
        #     # if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
        #     #     p_non_wd.append(p)
        #     else:
        #         p_wd.append(p)
        #     num_parameters += p.data.nelement()
        print(f"number of parameters: {num_parameters}")

        optim_params = [
            {
                "params": p_wd,
                "weight_decay": float(weight_decay),
            },
            {"params": p_non_wd, "weight_decay": 0},
        ]

        optimizer = torch.optim.AdamW(
            optim_params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )

        #scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=100, num_training_steps=5000)
        num_training_steps = self.cfg.experiment.total_training_steps
        num_warmup_steps = self.cfg.experiment.num_warmup_steps
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config,}
        
if __name__ == "__main__":
    cfg, cfg_yaml = build_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    seed_everything(cfg.experiment.seed, workers=True)

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    os.makedirs(cfg.result_file_path, exist_ok=True)

    datamodule = SEEDDataModule(cfg, transform=transform, use_coco_val=cfg.dataset.val_config.use_coco_val)
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    cfg.experiment.total_training_steps = datamodule.total_training_steps

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.result_file_path)
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval="step")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor="clip_score_coco_karpathy" if cfg.experiment.stage == 2 else "val/loss_itc_mean",
        mode="max" if cfg.experiment.stage == 2 else "min",
        every_n_epochs=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        accelerator=device,
        num_nodes=cfg.dist.n_nodes,
        devices=cfg.dist.n_gpus,
        strategy="ddp",
        max_epochs=cfg.experiment.max_epochs,
        deterministic=cfg.experiment.deterministic,
        logger=tb_logger,
        log_every_n_steps=cfg.experiment.log_every_n_steps,
        val_check_interval=cfg.experiment.val_check_interval,
        # check_val_every_n_epoch=cfg.experiment.check_val_every_n_epoch,
        enable_checkpointing=cfg.experiment.enable_checkpointing,
        num_sanity_val_steps=cfg.experiment.num_sanity_val_steps,
        precision=str(cfg.optimizer.precision),
        callbacks=[ModelSummary(max_depth=3), lr_logger], # + [checkpoint_callback] if cfg.experiment.enable_checkpointing else [],
        accumulate_grad_batches=cfg.experiment.grad_accumulation,
        gradient_clip_val=cfg.experiment.grad_clip_val,
    )

    if cfg.load_weight and cfg.resume:
        raise ValueError("Only checkpoint or finetune")

    if cfg.load_weight:
        wrapper = SlotTrainingWrapper.load_from_checkpoint(cfg.weight_path, cfg=cfg, strict=False).to(device)
        print("Loaded model from checkpoint")
    else:
        wrapper = SlotTrainingWrapper(cfg).to(device)
        if cfg.experiment.stage == 1:
            print(f"Stage 1 init from {cfg.stage1.init}")
        elif cfg.experiment.stage == 2:
            print("Stage 2 init from Scratch")

    wrapper.setup("fit")
    
    if cfg.resume:
        # Resume training
        if cfg.weight_path is None:
            raise ValueError("checkpoint_path is None")
        else:
            print(f"Resume training from {cfg.weight_path}")
            trainer.fit(
                wrapper, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
                ckpt_path=cfg.weight_path
            )
    else:
        print("Start training")
        trainer.fit(
            wrapper, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
        )
    
    trainer.strategy.barrier()