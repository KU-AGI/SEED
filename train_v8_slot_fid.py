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
from models.seed_qformer.qformer_quantizer import VectorQuantizer2, NSVQ

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

from slots_test.slot_heatmap import get_heatmap

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
        self.stage = cfg.experiment.stage

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
        self.vae = self.diffusion_model.vae

        # Scheduler for validation
        scheduler_args = {}
        if "variance_type" in self.scheduler.config:
            variance_type = self.scheduler.config.variance_type
            
            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type
            
        self.schduler = DPMSolverMultistepScheduler.from_config(
            self.scheduler.config, **scheduler_args
        )
        self.diffusion_model.scheduler = self.schduler

        # For logging
        self.stage = cfg.experiment.stage
        
        # Define slot attention
        slot_attn_config = MultiHeadSTEVESA.load_config(cfg.slot_cfg_path)
        self.slot_attention = MultiHeadSTEVESA.from_config(slot_attn_config)            
        self.slot_size = slot_attn_config['slot_size']
        self.num_slots = slot_attn_config['num_slots']

        self.out_linear_1024 = self.slot_attention.out_linear

        self.image_size = cfg.stage1.image_size
        self.transform_256 = transforms.Resize((self.image_size, self.image_size), antialias=True)

        self.normalize_diffusion = transforms.Normalize(mean=[0.5], std=[0.5])
        self.normalize_vit = transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                            )

        self.save_path = None

        # Load backbone
        dinov2 = torch.hub.load("facebookresearch/dinov2", cfg.stage1.dino_model_name)
        self.backbone = DINOBackbone(dinov2).eval()

    def get_diffusion_noisy_model_input(self, batch):
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

        return noisy_model_input, noise, timesteps, model_input

    def get_slot_embedding(self, batch, batch_idx: int):

        # timestep is not used, but should we?
        backbone_input = self.normalize_vit(batch[0])
        feat = self.backbone(backbone_input)

        slots, attn = self.slot_attention(feat[:, None], apply_out_linear=False)  # for the time dimension
        # Remove T dimension
        slots = rearrange(slots, "bs 1 n_s n_d -> bs n_s n_d")
        attn = rearrange(attn, "bs 1 n_s n_d -> bs n_s n_d")
        return slots, attn

    def get_diffusion_loss(self, batch, batch_idx: int, is_validation=False):
        noisy_model_input, noise, timesteps, model_input = self.get_diffusion_noisy_model_input(batch)
        logging_dict = {}

        slots, attn = self.get_slot_embedding(batch, batch_idx)

        slots_1024 = self.out_linear_1024(slots)

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
        logging_dict["loss_diffusion"] = loss_diffusion
        
        # loss logging
        cur_stage = "train" if not is_validation else "val"
        for key in logging_dict.keys():
            self.log(
                f"{cur_stage}/{key}",
                logging_dict[key],
                on_step=False if is_validation else True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        
        return loss_diffusion, slots, slots_1024, attn
    
    def calculate_rec_loss(self, rec, target):
        target = target / target.norm(dim=-1, keepdim=True)
        rec = rec / rec.norm(dim=-1, keepdim=True)
        rec_loss = (1 - (target * rec).sum(dim=-1)).mean()
        return rec_loss

    def apply_transformer(self, slots, transformer_blocks):
        pos_embed_applied_slot = slots + self.pos_embed.repeat(slots.size(0), 1, 1)
        # Apply Causal Transformer
        for blk in transformer_blocks:
            pos_embed_applied_slot = blk(pos_embed_applied_slot, use_causal_mask=False)
        return pos_embed_applied_slot

    def get_codebook_util(self, indices):
        num_codes = self.quantize.n_e
        uniques = indices.unique().numel()
        return (uniques / num_codes) * 100

    @torch.no_grad()
    def test_step(self, batch, batch_idx: int, save_path=None):
        if self.logger is not None and isinstance(self.logger, pl.loggers.TensorBoardLogger):
            tb_log_dir = self.logger.log_dir
        else:
            tb_log_dir = self.cfg.result_file_path  # Fallback directory if logger is not set

        self.save_path = f"{tb_log_dir}/reconstructed_images"
        os.makedirs(self.save_path, exist_ok=True)
        
        self.save_path = f"{tb_log_dir}/reconstructed_images/epoch_{self.current_epoch}"
        os.makedirs(self.save_path, exist_ok=True)

        img, text, image_id = batch
        _, _, slots_1024, attn = self.get_diffusion_loss(batch, batch_idx, is_validation=True)

        pixel_values = self.transform_256(img)
        heatmap = get_heatmap(pixel_values, attn)

        reconstructed_images = self.diffusion_model(
            prompt_embeds=slots_1024,
            height=self.image_size,
            width=self.image_size,
            guidance_scale=1.3,
            # num_inference_steps=25,
            num_inference_steps=100,
        ).images

        for img, cur_id in zip(reconstructed_images, image_id):
            # save PIL image to save_path
            img.save(f"{self.save_path}/{cur_id}")

    def on_test_epoch_end(self):
        if self.logger is not None and isinstance(self.logger, pl_loggers.TensorBoardLogger):
            tb_log_dir = self.logger.log_dir
        else:
            tb_log_dir = self.cfg.result_file_path

        original_image_dir = self.cfg.dataset.val_config.root_dir
        generated_image_dir = self.save_path
        try:
            clip_score = calculate_clip_s_for_folder(original_image_dir, generated_image_dir)
        except Exception as e:
            self.print(f"Error: {e}")
            clip_score = 0
        
        self.print(f"clip score: {clip_score}")
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

if __name__ == "__main__":
    # cfg, cfg_yaml = build_config()
    cfg = OmegaConf.load("pretrained/seed_slot_layer1_dino_12head_dataaug_learnablequery_200epoch_crossattn_unfreeze/config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    seed_everything(cfg.experiment.seed, workers=True)

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    os.makedirs(cfg.result_file_path, exist_ok=True)

    datamodule = SEEDDataModule(cfg, transform=transform, use_coco_val=cfg.dataset.val_config.use_coco_val)
    datamodule.setup()
    test_dataloader = datamodule.val_dataloader()

    strategy = DDPStrategy(
        find_unused_parameters=cfg.experiment.find_unused_parameters,
    )

    trainer = pl.Trainer(
        accelerator=device,
        num_nodes=cfg.dist.n_nodes,
        devices=cfg.dist.n_gpus,
        strategy=strategy,
        deterministic=True,
        precision=str(cfg.optimizer.precision),
    )

    if cfg.load_weight and cfg.resume:
        raise ValueError("Only checkpoint or finetune")

    if cfg.load_weight:
        wrapper = SlotTrainingWrapper.load_from_checkpoint(cfg.weight_path, cfg=cfg, strict=False)
    else:
        wrapper = SlotTrainingWrapper(cfg)
        if cfg.experiment.stage == 1:
            print(f"Stage 1 init from Scratch")
        elif cfg.experiment.stage == 2:
            print(f"Stage 2 init from {cfg.weight_path}")

    wrapper.setup("fit")
    trainer.test(wrapper, dataloaders=test_dataloader)

    trainer.strategy.barrier()
