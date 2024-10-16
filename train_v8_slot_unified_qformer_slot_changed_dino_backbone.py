import os
from typing import Any, List
import torch
from IPython.core.completer import back_latex_name_matcher
from skimage.restoration.uft import image_quad_norm
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

from calculate_clip_score import calculate_clip_s_for_folder, calculate_lpips_for_folder, calculate_psnr_for_folder, \
    calculate_ssim_for_folder
from utils.config import build_config

# from lavis.models import load_model
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

from vector_quantize_pytorch import VectorQuantize

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

        # Load tokenizer
        self.image_tokenizer = ImageTokenizer(
            model_path=cfg.checkpoint_path.model_path,
            diffusion_model_path=None,  # Diffusion model is loaded in TrainingWrapper
            device="cpu",  # For PyTorch Lightning
            load_diffusion=False,
            vq_type=cfg.stage2.vq.type,
            discarding_thre=cfg.stage2.vq.discarding_threshold,
            from_pretrained=True if cfg.checkpoint_path.model_path is not None else False,
            vit_precision=cfg.optimizer.vit_precision,
            diffusion_precision=cfg.optimizer.diffusion_precision,
            legacy=cfg.stage2.vq.legacy,
        )

        # Backbone is from ImageTokenizer
        dinov2 = torch.hub.load("facebookresearch/dinov2", cfg.stage1.dino_model_name)
        self.backbone = DINOBackbone(dinov2).eval()
        self.visual_embedding_encoder = nn.Linear(1024, 1408)
        self.image_tokenizer.model.visual_encoder = None

        # self.backbone = self.image_tokenizer.model.visual_encoder
        self.out_layer_norm = nn.LayerNorm(768)
        self.out_linear_1024 = nn.Linear(768, 1024)

        diffusion_precision = "fp16" if cfg.optimizer.diffusion_precision else "fp32"
        pretrained_model_name = "stabilityai/stable-diffusion-2-1"
        self.diffusion_model = StableDiffusionPipeline.from_pretrained(pretrained_model_name,
                                                                       torch_dtype=
                                                                       dict(fp16=torch.float16, fp32=torch.float32)[
                                                                           diffusion_precision])
        # Change to DDPMScheduler
        self.diffusion_model.scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name, subfolder="scheduler",
            torch_dtype=dict(fp16=torch.float16, fp32=torch.float32)[diffusion_precision])

        # For diffusion DDP
        self.feature_extractor = self.diffusion_model.feature_extractor

        self.image_encoder = None
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

        self.image_size = cfg.stage1.image_size
        self.transform_256 = transforms.Resize((self.image_size, self.image_size), antialias=True)

        self.normalize_diffusion = transforms.Normalize(mean=[0.5], std=[0.5])
        self.normalize_vit = transforms.Normalize(
            mean=(0.43216, 0.394666, 0.37645),
            std=(0.22803, 0.22145, 0.216989),
        )

        self.save_path = None

        # We don't use MSE CLIP loss
        self.image_tokenizer.model.image_down = None
        self.image_tokenizer.model.distill_image_proj = None

        # Use unused model for stage 1, Quantize is not used
        self.image_tokenizer.model.encode_task_layer = None
        self.image_tokenizer.model.decode_task_layer = None
        self.image_tokenizer.model.quantize = None
        self.image_tokenizer.model.blocks = None
        self.image_tokenizer.model.blocks_image = None

        # itc loss
        if hasattr(self.cfg.stage1, "itc_weight") and \
                self.cfg.stage1.itc_weight is not None and \
                self.cfg.stage1.itc_weight > 0:

            self.use_itc = True
            self.temp = nn.Parameter(0.07 * torch.ones([]))
        else:
            self.use_itc = False

        if self.stage == 2:
            self.codebook_embed_dim = self.cfg.stage2.vq.codebook_embed_dim
            self.n_embed = self.cfg.stage2.vq.n_embed

            print(f"n_embed: {self.n_embed}, codebook_embed_dim: {self.codebook_embed_dim}")
            # TODO : config slot_embed
            slot_embed = 1024

            if self.cfg.stage2.vq.vq_type == "vq":
                self.quantize = VectorQuantize(
                    dim=slot_embed,
                    codebook_size=self.n_embed,
                    codebook_dim=self.codebook_embed_dim
                )
            elif self.cfg.stage2.vq.vq_type == "residual_vq":
                from vector_quantize_pytorch import ResidualVQ
                self.quantize = ResidualVQ(
                    dim=slot_embed,
                    num_quantizers=self.cfg.stage2.vq.num_quantizers,
                    codebook_size=self.n_embed,
                    codebook_dim=self.codebook_embed_dim,
                    shared_codebook=True,
                )

            self.pos_embed = nn.Parameter(torch.zeros(1, 32, slot_embed))
            self.blocks = nn.ModuleList([
                Block(dim=slot_embed,
                      # num_heads=12,
                      num_heads=16,
                      mlp_ratio=4.0,
                      qkv_bias=True,
                      qk_scale=None,
                      drop=0.0,
                      attn_drop=0.0,
                      drop_path=0.0,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6)) for _ in range(self.cfg.stage2.blocks_layers)
            ])

            if self.cfg.stage2.use_blocks_image and \
                    self.cfg.stage2.blocks_image_layers is not None:

                self.use_blocks_image = True
                self.pos_embed_image = nn.Parameter(torch.zeros(1, 32, slot_embed))
                self.blocks_image = nn.ModuleList([
                    Block(dim=slot_embed,
                          num_heads=16,
                          mlp_ratio=4.0,
                          qkv_bias=True,
                          qk_scale=None,
                          drop=0.0,
                          attn_drop=0.0,
                          drop_path=0.0,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6)) for _ in
                    range(self.cfg.stage2.blocks_image_layers)
                ])
            else:
                self.use_blocks_image = False

    def setup(self, stage):
        # Freeze backbone
        if hasattr(self, "backbone") and self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Diffusion frozen
        if hasattr(self, "image_normalizer") and self.image_normalizer is not None:
            for param in self.image_normalizer.parameters():
                param.requires_grad = False
        if hasattr(self, "text_encoder") and self.text_encoder is not None:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        # Freeze VAE
        for param in self.vae.parameters():
            param.requires_grad = False

        if self.stage == 1:
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

        # Freezing for stage 2
        if self.stage == 2:
            if hasattr(self.cfg.stage2, "unfreeze_unet"):
                # casting to float32
                self.unet = self.unet.to(dtype=torch.float32)
            for name, param in self.unet.named_parameters():
                # If self.cfg.stage2.unfreeze_unet exists, unfreeze cross attention layer
                if hasattr(self.cfg.stage2, "unfreeze_unet") and self.cfg.stage2.unfreeze_unet:
                    if any(x in name for x in ["attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out"]):
                        print(f"Unfreeze {name}")
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    param.requires_grad = False

            # Freeze stage 1 model
            for param in self.image_tokenizer.model.Qformer.parameters():
                param.requires_grad = False

            # Allow to train the out layer norm and out linear
            if hasattr(self.cfg.stage2, "unfreeze_linear") and \
                    self.cfg.stage2.unfreeze_linear:
                for param in self.out_linear_1024.parameters():
                    param.requires_grad = True
            else:
                for param in self.out_linear_1024.parameters():
                    param.requires_grad = False

        ## make dump folder
        os.makedirs(self.cfg.result_file_path, exist_ok=True)

    def save_config(self):
        config_save_path = os.path.join(self.logger.log_dir, "config.yaml")
        with open(config_save_path, "w") as f:
            json.dump(self.cfg, f, indent=4)

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
        # if usedistributed training
        if not is_dist_avail_and_initialized():
            return tensor

        tensors_gather = [
            torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output

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

    def get_image_feats(self, batch, batch_idx: int):
        if len(batch) == 3:
            image, text, image_id = batch
        elif len(batch) == 2:
            image, text = batch
        else:
            raise ValueError(f"Unknown batch size {len(batch)}")

        # Normalize image
        image = self.normalize_vit(image)

        with torch.no_grad():
            image_embeds = self.backbone(image)  # [b, 1024, 16, 16]

        image_embeds = rearrange(image_embeds, "b d h w -> b (h w) d")  # [b, 256, 1024]

        image_embeds = self.visual_embedding_encoder(image_embeds)  # [b, 256, 1408]

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        query_tokens = self.image_tokenizer.model.query_tokens.expand(image_embeds.shape[0], -1, -1)

        # Assume image_embeds.shape[0] is the batch size (b) and you have 32 tokens (n)
        b, n, _ = query_tokens.shape

        query_output = self.image_tokenizer.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            use_slot=True,  # Important! Use slot
        )

        image_feats = query_output.last_hidden_state

        return image_feats

    def get_text_feats(self, batch, batch_idx: int):
        if len(batch) == 3:
            image, text, image_id = batch
        elif len(batch) == 2:
            image, text = batch

        text_tokens = self.image_tokenizer.model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        text_output = self.image_tokenizer.model.Qformer.bert(
            text_tokens.input_ids.to(self.device),
            attention_mask=text_tokens.attention_mask.to(self.device),
            return_dict=True,
            use_slot=True,
        )

        text_feat = text_output.last_hidden_state

        return text_feat

    def get_itc_loss_use_last_token(self, batch, batch_idx: int, is_validation=False, image_feats=None):
        if image_feats is None:
            image_feats = self.get_image_feats(batch, batch_idx)

        b = image_feats.size(0)

        image_feats = F.normalize(image_feats, dim=-1)
        image_feats = rearrange(image_feats[:, -1, :], "b d -> b 1 d").contiguous()

        text_feats = self.get_text_feats(batch, batch_idx)
        text_feats = F.normalize(text_feats, dim=-1)
        text_feats = text_feats[:, 0, :].contiguous()

        image_feats_all = self.all_gather_with_grad(image_feats)
        text_feats_all = self.all_gather_with_grad(text_feats)

        sim_q2t = torch.matmul(
            rearrange(image_feats, "bs n d -> bs 1 n d"),
            rearrange(text_feats_all, "(bs ngpus) d -> (bs ngpus) d 1", bs=b)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # Always use last token
        # sim_i2t = sim_q2t[:, :, -1]
        sim_i2t = sim_q2t
        # Debug : Test Original BLIP-2 loss
        # sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            rearrange(text_feats, "bs d -> bs 1 1 d"),
            rearrange(image_feats_all, "(bs ngpus) n d -> (bs ngpus) d n", bs=b)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # Always use last token
        # sim_t2i = sim_t2q[:, :, -1]
        sim_t2i = sim_t2q
        # Debug : Test Original BLIP-2 loss
        # sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        rank = dist.get_rank()
        bs = image_feats.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            self.device
        )

        loss_itc = (
                           F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                           + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
                   ) / 2

        return loss_itc

    def get_slot_embedding(self, batch, batch_idx: int, image_feats=None):
        if image_feats is None:
            image_feats = self.get_image_feats(batch, batch_idx)
        slots = self.out_layer_norm(image_feats)
        slots_1024 = self.out_linear_1024(slots)

        return slots, slots_1024

    def get_stage1_loss(self, batch, batch_idx: int, is_validation=False):
        noisy_model_input, noise, timesteps, model_input = self.get_diffusion_noisy_model_input(batch)
        logging_dict = {}

        image_feats = self.get_image_feats(batch, batch_idx)

        # itc loss calculation
        if self.use_itc:
            loss_itc = self.get_itc_loss_use_last_token(batch, batch_idx, is_validation, image_feats)
            logging_dict["loss_itc"] = loss_itc

        # Diffusion loss calculation
        slots, slots_1024 = self.get_slot_embedding(batch, batch_idx, image_feats)

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

        # Combine loss
        loss = loss_diffusion
        if self.use_itc:
            loss += self.cfg.stage1.itc_weight * loss_itc
            logging_dict["loss"] = loss

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

        return loss, image_feats, slots, slots_1024

    def calculate_rec_loss(self, rec, target):
        target = target / target.norm(dim=-1, keepdim=True)
        rec = rec / rec.norm(dim=-1, keepdim=True)
        rec_loss = (1 - (target * rec).sum(dim=-1)).mean()
        return rec_loss

    def apply_transformer(self, slots, transformer_blocks, pos_embed):
        pos_embed_applied_slot = slots + pos_embed.repeat(slots.size(0), 1, 1)
        # Apply Causal Transformer
        for blk in transformer_blocks:
            pos_embed_applied_slot = blk(pos_embed_applied_slot, use_causal_mask=False)
        return pos_embed_applied_slot

    def get_codebook_util(self, indices):
        num_codes = self.quantize.n_e
        uniques = indices.unique().numel()
        return (uniques / num_codes) * 100

    def get_quantize_loss(self, batch, batch_idx: int, is_validation=False):
        logging_dict = {}

        with torch.no_grad():
            slots, slots_1024 = self.get_slot_embedding(batch, batch_idx)

        # Quantize slots
        quant, embed_ind, loss_codebook = self.quantize(slots_1024)

        if self.cfg.stage2.vq.vq_type == "residual_vq":
            loss_codebook = loss_codebook.sum()

        logging_dict["loss_codebook"] = loss_codebook

        # Flatten the quantized slots
        embed_ind = embed_ind.reshape(slots.shape[0], -1)

        slots_1024_blocks_applied = self.apply_transformer(quant, self.blocks, self.pos_embed)

        loss_recon = F.mse_loss(slots_1024, slots_1024_blocks_applied, reduction="mean")
        logging_dict["loss_recon"] = loss_recon
        logging_dict["sim_768"] = F.cosine_similarity(slots_1024, slots_1024_blocks_applied, dim=-1).mean()

        # Compute diffusion loss
        noisy_model_input, noise, timesteps, model_input = self.get_diffusion_noisy_model_input(batch)

        # Predict the noise residual
        # By using the reconstructed slot
        if self.use_blocks_image:
            slots_1024_blocks_image_applied = self.apply_transformer(
                quant, self.blocks_image, self.pos_embed_image
            )

            # Calculate diffusion loss by using blocks_image applied slot
            model_pred = self.unet(
                noisy_model_input, timesteps, slots_1024_blocks_image_applied,
            ).sample
        else:
            # Don't use blocks_image, directly use reconstructed slot
            model_pred = self.unet(
                noisy_model_input, timesteps, slots_1024_blocks_applied,
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

        # Combine loss
        loss = self.cfg.stage2.loss_weight.loss_codebook * loss_codebook + \
               self.cfg.stage2.loss_weight.loss_recon * loss_recon + \
               self.cfg.stage2.loss_weight.loss_diffusion * loss_diffusion
        logging_dict["loss"] = loss

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

        return loss, embed_ind, slots_1024_blocks_applied

    def on_train_start(self):
        self.print(f"\n====Traing Stage {self.stage}====")
        if self.stage == 2 and self.cfg.stage2.bypass_codebook:
            self.print("\n====Bypass codebook====")

        self.print("Save config")
        self.save_config()

    def training_step(self, batch, batch_idx: int):
        if len(batch) == 3:
            image, text, image_id = batch
        elif len(batch) == 2:
            image, text = batch

        self.B = image.shape[0]

        if self.stage == 1:
            loss, *_ = self.get_stage1_loss(batch, batch_idx)
        elif self.stage == 2:
            loss, *_ = self.get_quantize_loss(batch, batch_idx)

        return loss

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        # {'grad_2.0_norm/weight': 0.0003, 'grad_2.0_norm/bias': 0.0, 'grad_2.0_norm_total': 0.0003}
        for i in range(0, 12, 2):
            norm_slot = grad_norm(self.image_tokenizer.model.Qformer.bert.encoder.layer[i], norm_type=2)
            for norm in norm_slot.keys():
                self.logger.experiment.add_scalar(
                    f"grad_norm/bert_slot_{i}/{norm}",
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

        norm_out_linear_1024 = grad_norm(self.out_linear_1024, norm_type=2)
        for norm in norm_out_linear_1024.keys():
            self.logger.experiment.add_scalar(
                f"grad_norm/out_linear_1024/{norm}",
                norm_out_linear_1024[norm],
                global_step=self.global_step,
            )

        if hasattr(self, "blocks"):
            norm_blocks = grad_norm(self.blocks, norm_type=2)
            for norm in norm_blocks.keys():
                self.logger.experiment.add_scalar(
                    f"grad_norm/blocks/{norm}",
                    norm_blocks[norm],
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
        if self.stage == 1:
            _, _, _, slots_1024 = self.get_stage1_loss(batch, batch_idx, is_validation=True)
        else:
            _, _, slots_1024 = self.get_quantize_loss(batch, batch_idx, is_validation=True)

        # Change validation scheduler
        cur_scheduler = self.scheduler
        self.diffusion_model.scheduler = self.val_schduler

        reconstructed_images = self.diffusion_model(
            prompt_embeds=slots_1024,
            height=self.image_size,
            width=self.image_size,
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
        try:
            clip_score = calculate_clip_s_for_folder(original_image_dir, generated_image_dir)
            self.log(
                "metrics/clip_score",
                clip_score,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
        except Exception as e:
            self.print(f"Error: {e}")
            clip_score = 0

        try:
            lpips = calculate_lpips_for_folder(original_image_dir, generated_image_dir)
            self.log(
                "metrics/lpips",
                lpips,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
        except Exception as e:
            self.print(f"Error: {e}")
            lpips = 0

        try:
            psnr = calculate_psnr_for_folder(original_image_dir, generated_image_dir)
            self.log(
                "metrics/psnr",
                psnr,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
        except Exception as e:
            self.print(f"Error: {e}")
            psnr = 0

        try:
            ssim = calculate_ssim_for_folder(original_image_dir, generated_image_dir)
            self.log(
                "metrics/ssim",
                ssim,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
        except Exception as e:
            self.print(f"Error: {e}")
            ssim = 0

        self.log_dict({
            'clip_score': clip_score,
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

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
            '''
            if not p.requires_grad:
                continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
            '''
            p_wd.append(p)
            num_parameters += p.data.nelement()

        self.print(f"number of parameters: {num_parameters}")

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

        # scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=100, num_training_steps=5000)
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
                "lr_scheduler": lr_scheduler_config, }


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
        monitor="clip_score_coco_karpathy",
        mode="max",
        every_n_epochs=1,
        save_last=True,
    )

    if not cfg.experiment.enable_checkpointing:
        print("############### WARNING: Checkpointing is disabled ###############")

    strategy = DDPStrategy(
        find_unused_parameters=cfg.experiment.find_unused_parameters,
    )

    trainer = pl.Trainer(
        accelerator=device,
        num_nodes=cfg.dist.n_nodes,
        devices=cfg.dist.n_gpus,
        strategy=strategy,
        max_epochs=cfg.experiment.max_epochs,
        deterministic=cfg.experiment.deterministic,
        logger=tb_logger,
        log_every_n_steps=cfg.experiment.log_every_n_steps,
        val_check_interval=cfg.experiment.val_check_interval,
        # check_val_every_n_epoch=cfg.experiment.check_val_every_n_epoch,
        enable_checkpointing=cfg.experiment.enable_checkpointing,
        num_sanity_val_steps=cfg.experiment.num_sanity_val_steps,
        precision=str(cfg.optimizer.precision),
        callbacks=[ModelSummary(max_depth=3), lr_logger, checkpoint_callback],
        accumulate_grad_batches=cfg.experiment.grad_accumulation,
        gradient_clip_val=cfg.experiment.grad_clip_val,
    )

    if cfg.load_weight and cfg.resume:
        raise ValueError("Only checkpoint or finetune")

    if cfg.load_weight:
        wrapper = SlotTrainingWrapper.load_from_checkpoint(cfg.weight_path, cfg=cfg, strict=False)
    else:
        wrapper = SlotTrainingWrapper(cfg)
        if cfg.experiment.stage == 1:
            print(f"Stage 1 init from {cfg.checkpoint_path.model_path}")
        elif cfg.experiment.stage == 2:
            print(f"Stage 2 init from {cfg.weight_path}")

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
