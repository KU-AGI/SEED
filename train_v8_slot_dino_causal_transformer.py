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

class SEEDTrainingWrapper(LightningModule):
    """Training wrapper for SEED

    Args:
        LightningModule (cfg, model): model should be ImageTokenizer
    """    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # ImageTokenizer model
        # Target model to train
        self.image_tokenizer = ImageTokenizer(
            model_path=cfg.checkpoint_path.model_path,
            diffusion_model_path=cfg.checkpoint_path.diffusion_model_path,
            load_diffusion=cfg.stage2.load_diffusion,
            vq_type=cfg.stage2.vq.type,
            discarding_thre=cfg.stage2.vq.discarding_threshold,
            from_pretrained=True if cfg.stage1.init == "SEED" else False,
            vit_precision=cfg.optimizer.vit_precision,
            diffusion_precision=cfg.optimizer.diffusion_precision,
            unclip=cfg.stage2.unclip,
            legacy=cfg.stage2.vq.legacy,
        )

        # Debug
        self.image_tokenizer.model = self.image_tokenizer.model.to("cpu")

        self.B = None

        self.transform_224 = transforms.Resize((224, 224), antialias=True)

        # For diffusion DDP
        if self.image_tokenizer.diffusion_model is not None:
            self.feature_extractor = self.image_tokenizer.diffusion_model.feature_extractor
            if self.cfg.stage2.unclip:
                self.image_encoder = self.image_tokenizer.diffusion_model.image_encoder
                self.image_normalizer = self.image_tokenizer.diffusion_model.image_normalizer
                self.image_noising_scheduler = self.image_tokenizer.diffusion_model.image_noising_scheduler
            else:
                self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
                self.image_normalizer = None
                self.image_noising_scheduler = None
            if self.image_tokenizer.diffusion_model.text_encoder is not None:
                self.tokenizer = self.image_tokenizer.diffusion_model.tokenizer
            else:
                self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            if self.image_tokenizer.diffusion_model.text_encoder is not None:
                self.text_encoder = self.image_tokenizer.diffusion_model.text_encoder
            self.unet = self.image_tokenizer.diffusion_model.unet
            self.scheduler = self.image_tokenizer.diffusion_model.scheduler
            self.vae = self.image_tokenizer.diffusion_model.vae

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
        self.pil_to_tensor = transforms.ToTensor()
        self.sample_image_ind = 0
        self.logged_original_image = set()
        
        self.stage = cfg.experiment.stage
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        if cfg.stage2.recon_s is not None:
            self.recon_s = cfg.stage2.recon_s
        else:
            self.recon_s = False

        # Test for query token xaiver uniform initialization
        if cfg.stage1.use_slot:
            if not cfg.resume and not cfg.load_weight:
                torch.nn.init.xavier_uniform_(self.image_tokenizer.model.query_tokens)
        
        # slot out linear
        self.out_layer_norm = nn.LayerNorm(768)
        self.out_linear = nn.Linear(768, 1024)

        self.transform_256 = transforms.Resize((256, 256), antialias=True)

        self.normalize_diffusion = transforms.Normalize(mean=[0.5], std=[0.5])
        self.normalize_vit = transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                            )

        self.save_path = None
        
        # DEBUG : Use subset of bert layer
        if self.cfg.stage1.using_layer is not None:
            # self.image_tokenizer.model.Qformer.bert.encoder.layer = self.image_tokenizer.model.Qformer.bert.encoder.layer[cfg.stage1.using_layer:cfg.stage1.using_layer + 1]
            # self.image_tokenizer.model.Qformer.bert.encoder.config.num_hidden_layers = 1

            # self.slot_attention = self.image_tokenizer.model.Qformer.bert.encoder.layer[0].crossattention

            # Debug
            slot_attn_config = MultiHeadSTEVESA.load_config(cfg.slot_cfg_path)
            self.slot_attention = MultiHeadSTEVESA.from_config(slot_attn_config)            
            # self.slot_attention = MultiHeadSTEVESA.from_pretrained("/home/zheedong/Projects/latent-slot-diffusion/lsd_log/coco_32slot/stable_lsd/checkpoint-49000/multiheadstevesa")

            dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            self.backbone = DINOBackbone(dinov2)

            for param in self.backbone.parameters():
                param.requires_grad = False

            # Test
            # from diffusers import (
            #     AutoencoderKL,
            #     DDPMScheduler,
            #     UNet2DConditionModel,
            # )
            # model_name = "stabilityai/stable-diffusion-2-1"
            # self.test_vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
            # self.test_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
            # self.test_unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", revision=None)

    def setup(self, stage):
        # Setup training parameter
        self.image_tokenizer.model.train()
        for param in self.image_tokenizer.model.parameters():
            param.requires_grad = False
        self.image_tokenizer.model = self.image_tokenizer.model.to("cpu")

        # Freeze ViT Encoder
        for param in self.image_tokenizer.model.visual_encoder.parameters():
            param.requires_grad = False
        self.image_tokenizer.model.visual_encoder = self.image_tokenizer.model.visual_encoder.to("cpu")
        for param in self.image_tokenizer.model.ln_vision.parameters():
            param.requires_grad = False
        self.image_tokenizer.model.ln_vision = self.image_tokenizer.model.ln_vision.to("cpu")

        # Diffusion frozen
        if self.image_tokenizer.diffusion_model is not None:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            if self.image_normalizer is not None:
                for param in self.image_tokenizer.diffusion_model.image_normalizer.parameters():
                    param.requires_grad = False
            if self.image_noising_scheduler is not None:
                for param in self.image_tokenizer.diffusion_model.text_encoder.parameters():
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
                    
            for param in self.vae.parameters():
                param.requires_grad = False

        if self.stage == 1:
            if self.cfg.stage1.init == "BLIP-2":
                print("Load init weights from BLIP-2")
                blip_model = load_model("blip2", "pretrain")
                # Update the model with the weights
                filtered_state_dict = {k: v for k, v in blip_model.state_dict().items() if k in self.image_tokenizer.model.state_dict()}
                self.image_tokenizer.model.load_state_dict(filtered_state_dict, strict=False)

            elif self.cfg.stage1.init == "SEED":
                print("Load init weights from SEED")

            print("Set stage 2 model not trainable")
            for param in self.image_tokenizer.model.quantize.parameters():
                param.requires_grad = False
            for param in self.image_tokenizer.model.encode_task_layer.parameters():
                param.requires_grad = False
            for param in self.image_tokenizer.model.decode_task_layer.parameters():
                param.requires_grad = False
            for param in self.image_tokenizer.model.blocks.parameters():
                param.requires_grad = False
            for param in self.image_tokenizer.model.blocks_image.parameters():
                param.requires_grad = False
            for param in self.image_tokenizer.model.image_down.parameters():
                param.requires_grad = False
            for param in self.image_tokenizer.model.distill_image_proj.parameters():
                param.requires_grad = False
                
            print("Move stage 2 model to cpu")
            self.image_tokenizer.model.quantize = self.image_tokenizer.model.quantize.to("cpu")
            self.image_tokenizer.model.encode_task_layer = self.image_tokenizer.model.encode_task_layer.to("cpu")
            self.image_tokenizer.model.decode_task_layer = self.image_tokenizer.model.decode_task_layer.to("cpu")
            self.image_tokenizer.model.blocks = self.image_tokenizer.model.blocks.to("cpu")
            self.image_tokenizer.model.blocks_image = self.image_tokenizer.model.blocks_image.to("cpu")
            self.image_tokenizer.model.image_down = self.image_tokenizer.model.image_down.to("cpu")
            self.image_tokenizer.model.distill_image_proj = self.image_tokenizer.model.distill_image_proj.to("cpu")
        elif self.stage == 2:
            print("Freeze stage 1 model")
            for name, param in self.image_tokenizer.model.Qformer.named_parameters():
                param.requires_grad = False
            self.image_tokenizer.model.query_tokens.requires_grad = False

            # Random initialize stage 2 model weights
            if self.cfg.stage1.init == "trained":
                print("Random initialize stage 2 model weights")
                self.random_initialize_stage2_model_weights()
            elif "init_stage_2" in self.cfg.stage2 and self.cfg.stage2.init_stage_2:
                print("Load stage 2 model weights from checkpoint")
            elif not self.cfg.resume and not self.cfg.load_weight:
                print("Random initialize stage 2 model weights")
                self.random_initialize_stage2_model_weights()
            else:
                print("Load stage 2 model weights from checkpoint")
            
        ## make dump folder
        os.makedirs(self.cfg.result_file_path, exist_ok=True)
    
    def random_initialize_stage2_model_weights(self):
        """Random initialize stage 2 model weights
        """        
        print("Random initialize stage 2 model weights") 
        # unFreeze stage 2 model and initialize with random weights
        for param in self.image_tokenizer.model.encode_task_layer.parameters():
            #nn.init.xavier_uniform_(param) 
            nn.init.normal_(param, mean=0.0, std=0.02)
        for param in self.image_tokenizer.model.quantize.parameters():
            nn.init.normal_(param, mean=0.0, std=0.02)
        for param in self.image_tokenizer.model.decode_task_layer.parameters():
            nn.init.normal_(param, mean=0.0, std=0.02)
        for param in self.image_tokenizer.model.blocks_image.parameters():
            nn.init.normal_(param, mean=0.0, std=0.02)
        for param in self.image_tokenizer.model.image_down.parameters():
            nn.init.normal_(param, mean=0.0, std=0.02)
        for param in self.image_tokenizer.model.distill_image_proj.parameters():
            nn.init.normal_(param, mean=0.0, std=0.02)
        
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
                    self.tokenizer(batch_text[idx]).squeeze().to(self.device)
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

    def get_causal_embeddings(self, image):
        return self.image_tokenizer.model.get_causal_embeddings(image, self.cfg.stage1.use_slot)

    def logging_train_stage2(self, generation_embedding_cosine_similarity, loss_dict, is_val=False):
        stage = "val" if is_val else "train"
        self.log(
            f"{stage}/unweighted_loss_generation_embed",
            loss_dict["unweighted_loss_generation_embed"].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True if is_val else False,
        )

        self.log(
            f"{stage}/generation_embedding_mse_loss",
            loss_dict["loss_generation_embed"].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True if is_val else False,
        )

        if "loss_embed" in loss_dict.keys():
            self.log(
                f"{stage}/unweighted_loss_embed",
                loss_dict["unweighted_loss_embed"].mean(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

            self.log(
                f"{stage}/codebook_loss_embed",
                loss_dict["loss_embed"].mean(),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True if is_val else False,
            )

        self.log(
            f"{stage}/unweighted_loss_recon",
            loss_dict["unweighted_loss_recon"].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True if is_val else False,
        )

        self.log(
            f"{stage}/reconstruction_loss",
            loss_dict["loss_recon"].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True if is_val else False,
        )

        self.log(
            f"{stage}/total_loss",
            loss_dict["loss"].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True if is_val else False,
        )

        if "perplexity" in loss_dict.keys():
            self.log(
                f"{stage}/codebook_perplexity",
                loss_dict["perplexity"].mean(),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True if is_val else False,
            )

        if "num_replaced_codebook" in loss_dict.keys():
            self.log(
                f"{stage}/num_replaced_codebook",
                loss_dict["num_replaced_codebook"],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True if is_val else False,
            )
 
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

    def get_stage_1_loss_use_last_token(self, batch, batch_idx: int):
        """
            Contrastive loss using last token of the query_output
        Args:
            batch (_type_): _description_
            batch_idx (int): _description_
            is_validation (bool, optional): _description_. Defaults to False.
        """
        if len(batch) == 3:
            image, text, image_id = batch
        elif len(batch) == 2:
            image, text = batch

        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=False):
                original_data_type = image.dtype
                # TODO : precision issue
                if self.cfg.optimizer.vit_precision == "fp16":
                    image = image.half()

                image_embeds = self.image_tokenizer.model.ln_vision(
                    self.image_tokenizer.model.visual_encoder(image)
                )

                image_embeds = image_embeds.to(original_data_type)


        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        query_tokens = self.image_tokenizer.model.query_tokens.expand(image_embeds.shape[0], -1, -1)

        # Assume image_embeds.shape[0] is the batch size (b) and you have 32 tokens (n)
        b, n, _ = query_tokens.shape

        query_output = self.image_tokenizer.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            use_slot=self.cfg.stage1.use_slot,
            use_caual=self.cfg.stage1.use_causal,
        )

        # Use last hidden state
        # We have 32 tokens, and use last token as image embedding
        # [b, 32, 768]
        # TODO: Use 'final' causal embedding? Does it mean to use last token embedding?
        # Debug
        image_feats = rearrange(query_output.last_hidden_state[:, -1, :], "b d -> b 1 d").contiguous()
        image_feats = F.normalize(image_feats, dim=-1)

        text_tokens = self.image_tokenizer.model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.cfg.dataset.text_max_length,
            return_tensors="pt",
        )

        text_output = self.image_tokenizer.model.Qformer.bert(
            text_tokens.input_ids.to(self.device),
            attention_mask=text_tokens.attention_mask.to(self.device),
            return_dict=True,
        )

        # CLS token
        # [b, 768]
        text_feat = F.normalize(text_output.last_hidden_state[:, 0, :], dim=-1)

        ###============== Image-text Contrastive ===================###
        # Compute for each query token
        # image_feats_all = self.concat_all_gather(
        image_feats_all = self.all_gather_with_grad(
            image_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        # text_feat_all = self.concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]
        text_feat_all = self.all_gather_with_grad(text_feat)  # [batch_size*num_gpu, embed_dim]

        # image_feats.unsqueeze(1) : [batch_size, 1, num_query_tokens, embed_dim]
        # text_feat_all.unsqueeze(-1) : [batch_size*num_gpu, embed_dim, 1] => broadcast to [batch_size, batch_size*num_gpu, embed_dim, 1]
        # Last two dimensions are broadcasted to all other dimensions
        # [j, 1, n, m] x [k, m, p] => [j, k, n, p]
        # https://pytorch.org/docs/stable/generated/torch.matmul.html
        # sim_q2t : [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_q2t = torch.matmul(
            rearrange(image_feats, "bs n d -> bs 1 n d"), rearrange(text_feat_all, "(bs ngpus) d -> (bs ngpus) d 1", bs=b)
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
            rearrange(text_feat, "bs d -> bs 1 1 d"), rearrange(image_feats_all, "(bs ngpus) n d -> (bs ngpus) d n", bs=b)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # Always use last token
        # sim_t2i = sim_t2q[:, :, -1]
        sim_t2i = sim_t2q
        # Debug : Test Original BLIP-2 loss
        # sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        rank = dist.get_rank()
        bs = image.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            image.device
        )

        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        self.log(
                "train/loss_itc",
                loss_itc,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return loss_itc

    def debug_lsd_loss(self, batch, batch_idx: int, is_validation=False):
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

        slots, attn = self.slot_attention(feat[:, None])  # for the time dimension
        slots = slots[:, 0]

        # Predict the noise residual
        model_pred = self.unet(
            noisy_model_input, timesteps, slots,
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
        loss = F.mse_loss(model_pred.float(),
                            target.float(), reduction="mean")

        if not is_validation:
            self.log(
                "train/loss_diffusion",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return loss, slots

    def get_stage_1_diffusion_loss(self, batch, batch_idx: int):
        """
            MSE loss with diffusion
        Args:
            batch (_type_): _description_
            batch_idx (int): _description_
            is_validation (bool, optional): _description_. Defaults to False.
        """
        if len(batch) == 3:
            image, text, image_id = batch
        elif len(batch) == 2:
            image, text = batch

        # Diffusion training target

        # Convert images to latent space
        pixel_values = self.transform_256(batch[0])
        pixel_values = self.normalize_diffusion(pixel_values)

        model_input = self.vae.encode(pixel_values).latent_dist.sample()
        model_input = model_input * self.vae.config.scaling_factor
        
        # Sample noise that we'll add to the model input
        # randn_like !!!
        noise = torch.randn_like(model_input)

        bsz, channels, height, width = model_input.shape
        # Sample a random timesteps for each image 
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (bsz,), device=self.device
        )
        timesteps = timesteps.long()
        
        # Add noise to the model input according to the noise magnitude at each timesteps
        noisy_model_input = self.scheduler.add_noise(
            model_input, noise, timesteps
        )

        backbone_input = self.normalize_vit(batch[0])
        image_embeds = self.image_tokenizer.model.visual_encoder(backbone_input)

        # DEBUG
        if self.cfg.stage1.using_layer is not None:
            slots, attn = self.slot_attention(image_embeds[:, None])
            slots = slots[:, 0]

            # slots = self.slot_attention(
            #     query_tokens,
            #     encoder_hidden_states=image_embeds,
            #     encoder_attention_mask=image_atts,
            #     use_slot=True,
            #     use_causal_mask=False,
            # )
            # slots = slots[0]
        else:
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)

            query_tokens = self.image_tokenizer.model.query_tokens.expand(image_embeds.shape[0], -1, -1)

            # Assume image_embeds.shape[0] is the batch size (b) and you have 32 tokens (n)
            b, n, _ = query_tokens.shape

            slots = self.image_tokenizer.model.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                use_slot=self.cfg.stage1.use_slot,
                use_causal=self.cfg.stage1.use_causal,
            ).last_hidden_state
            # For slot out linear
            slots = self.out_layer_norm(slots)
            slots = self.out_linear(slots)

        model_pred = self.unet(
            noisy_model_input,
            timesteps,
            slots,
        ).sample

        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(
                model_input, noise, timesteps
            )
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler.config.prediction_type}"
            )
        
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        self.log(
                "train/loss_diffusion",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return loss, slots

    @torch.no_grad()
    def check_image_text_similarity(self, batch, batch_idx: int, save_dir="image_text_similarity"):
        if len(batch) == 3:
            image, text, image_id = batch
        elif len(batch) == 2:
            image, text = batch
        rank = dist.get_rank()

        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=False):
                # TODO : precision issue
                if self.cfg.optimizer.vit_precision == "fp16":
                    image = image.half()

                image_embeds = self.image_tokenizer.model.ln_vision(
                    self.image_tokenizer.model.visual_encoder(image)
                )

                if self.cfg.optimizer.vit_precision == "fp16":
                    image_embeds = image_embeds.float()

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        query_tokens = self.image_tokenizer.model.query_tokens.expand(image_embeds.shape[0], -1, -1)

        # Assume image_embeds.shape[0] is the batch size (b) and you have 32 tokens (n)
        b, n, _ = query_tokens.shape

        query_output = self.image_tokenizer.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            use_slot=self.cfg.stage1.use_slot,
            use_causal=self.cfg.stage1.use_causal,
        )

        # Use last hidden state
        # We have 32 tokens, and use last token as image embedding
        # [b, 32, 768]
        image_feats = F.normalize(query_output.last_hidden_state, dim=-1)

        text_tokens = self.image_tokenizer.model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.cfg.dataset.text_max_length,
            return_tensors="pt",
        )

        text_output = self.image_tokenizer.model.Qformer.bert(
            text_tokens.input_ids.to(self.device),
            attention_mask=text_tokens.attention_mask.to(self.device),
            return_dict=True,
        )

        # CLS token
        # [b, 768]
        text_feat = F.normalize(text_output.last_hidden_state[:, 0, :], dim=-1)

        ###============== Image-text Contrastive ===================###

        # Original BLIP-2 loss
        # Compute for each query token
        image_feats_all = image_feats # [batch_size, num_query_tokens, embed_dim]
        text_feat_all = text_feat  # [batch_size, embed_dim]

        # image_feats.unsqueeze(1) : [batch_size, 1, num_query_tokens, embed_dim]
        # text_feat_all.unsqueeze(-1) : [batch_size*num_gpu, embed_dim, 1] => broadcast to [batch_size, batch_size*num_gpu, embed_dim, 1]
        # Last two dimensions are broadcasted to all other dimensions
        # [j, 1, n, m] x [k, m, p] => [j, k, n, p]
        # https://pytorch.org/docs/stable/generated/torch.matmul.html
        # sim_q2t : [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_q2t = torch.matmul(
            rearrange(image_feats, "bs n d -> bs 1 n d"), rearrange(text_feat_all, "bs_X_ngpus d -> bs_X_ngpus d 1")
            # image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        ########### 1. Debug: for check the similarity ############
        # Softmax for each row
        dump = []
        for token_num in range(32):
            dump.append(F.softmax(sim_q2t[:, :, token_num], dim=1))
        dump = torch.stack(dump, dim=2)
        positive_token_similarity = torch.diagonal(dump, dim1=0, dim2=1).mean(dim=1)
        # Save positive_token_similarity as bar graph
        plt.figure(figsize=(18, 6))
        bars = plt.bar(list(range(32)), positive_token_similarity.cpu().numpy(), color='blue')
        plt.xlabel('Token Number')
        plt.ylabel('Value')
        plt.title('Positive Token Similarity')
        plt.xticks(list(range(32)))  # Ensure all keys are shown in the x-axis
        # Add a table of values next to the bars
        cell_text = [[f"{val:.4f}"] for val in positive_token_similarity.cpu().numpy()]
        plt.table(cellText=cell_text, colLabels=["Value"], loc='right', cellLoc='center')

        # Adjust layout to make room for the table:
        plt.subplots_adjust(right=0.5)
        plt.savefig(f"{save_dir}/positive_token_similarity_i2t_batch{batch_idx}_rank{rank}.png")

        ############################################################
        # Debug: for check the similarity
        count_dict = {}
        for token_num in range(32):
            count_dict[token_num] = 0
            for row in range(b):
                _, ind = sim_q2t[:, :, token_num][row].max(-1)
                if row == ind:
                    print(f"In token {token_num}, in row {row}, max index is {ind}")
                    count_dict[token_num] += 1
        print(count_dict)

        # Extracting keys and values
        keys = list(count_dict.keys())
        values = list(count_dict.values())

        # Plotting the histogram
        plt.figure(figsize=(10, 6))
        bars = plt.bar(keys, values, color='blue')
        plt.xlabel('Token Number')
        plt.ylabel('Value')
        plt.title('Histogram of Token Values')
        plt.xticks(keys)  # Ensure all keys are shown in the x-axis

        # Adding the text on top of each bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom')

        os.makedirs(f"{save_dir}", exist_ok=True)
        plt.savefig(f"{save_dir}/token_histogram_image_text_batch{batch_idx}_rank{rank}.png")
        # plt.show()
        ############################################################

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            rearrange(text_feat, "bs d -> bs 1 1 d"), rearrange(image_feats_all, "bs_X_ngpus n d -> bs_X_ngpus d n")
            # text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()

        # Debug: for check the similarity
        count_dict = {}
        for token_num in range(32):
            count_dict[token_num] = 0
            for row in range(b):
                _, ind = sim_t2q[:, :, token_num][row].max(-1)
                if row == ind:
                    print(f"In token {token_num}, in row {row}, max index is {ind}")
                    count_dict[token_num] += 1
        print(count_dict)

        # Softmax for each row
        dump = []
        for token_num in range(32):
            dump.append(F.softmax(sim_t2q[:, :, token_num], dim=1))
        dump = torch.stack(dump, dim=2)
        positive_token_similarity = torch.diagonal(dump, dim1=0, dim2=1).mean(dim=1)
        # Save positive_token_similarity as bar graph
        plt.figure(figsize=(18, 6))
        bars = plt.bar(list(range(32)), positive_token_similarity.cpu().numpy(), color='blue')
        plt.xlabel('Token Number')
        plt.ylabel('Value')
        plt.title('Positive Token Similarity')
        plt.xticks(list(range(32)))  # Ensure all keys are shown in the x-axis
        # Add a table of values next to the bars
        cell_text = [[f"{val:.4f}"] for val in positive_token_similarity.cpu().numpy()]
        plt.table(cellText=cell_text, colLabels=["Value"], loc='right', cellLoc='center')

        # Adjust layout to make room for the table:
        plt.subplots_adjust(right=0.5)
        plt.savefig(f"{save_dir}/positive_token_similarity_t2i_batch{batch_idx}_rank{rank}.png")

        # Extracting keys and values
        keys = list(count_dict.keys())
        values = list(count_dict.values())

        # Plotting the histogram
        plt.figure(figsize=(10, 6))
        bars = plt.bar(keys, values, color='blue')
        plt.xlabel('Token Number')
        plt.ylabel('Value')
        plt.title('Histogram of Token Values')
        plt.xticks(keys)  # Ensure all keys are shown in the x-axis

        # Adding the text on top of each bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom')

        plt.savefig(f"{save_dir}/token_histogram_text_image_batch{batch_idx}_rank{rank}.png")

        loss_mean = 0
        rank = dist.get_rank()
        if rank == 0:
            for token in range(32):
                bs = image.size(0)
                targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
                    image.device
                )

                sim_i2t = sim_q2t[:, :, token] / self.temp
                sim_t2i = sim_t2q[:, :, token] / self.temp

                loss_itc = (
                    F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                    + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
                ) / 2
                print(f"Loss I2T in Token {token}: {loss_itc}")
                loss_mean += loss_itc

                self.log(
                        f"val/loss_itc_{token}",
                        loss_itc,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                    )

            loss_mean /= 32
            self.log(
                    "val/loss_itc_mean",
                    loss_mean,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
        return

    def calculate_rec_loss(self, rec, target):
        target = target / target.norm(dim=-1, keepdim=True)
        rec = rec / rec.norm(dim=-1, keepdim=True)
        rec_loss = (1 - (target * rec).sum(dim=-1)).mean()
        return rec_loss

    def get_stage_2_loss(self, batch, batch_idx: int, is_validation=False):
        """_summary_

        Args:
            batch (_type_): _description_
            batch_idx (int): _description_
        """        
        #------------------------
        # Stage 2 Training
        #------------------------
        device = self.device
        if len(batch) == 3:
            img, text, image_id = batch
        elif len(batch) == 2:
            img, text = batch

        #------------------------
        # Stage 2 - 1 : Codebook Training
        #------------------------
        with torch.no_grad():
            causal_embeddings = self.get_causal_embeddings(img)

        # TODO: query_output should be trained to be similar with text embedding
        # Image embedding is cross attentioned.
        # Notice: query_output_down is match to clip embedding?
        # [b, 32, 32]
        query_output_down = self.image_tokenizer.model.encode_task_layer(causal_embeddings)

        # quant [b, 32, 32], loss_embed [b, 32, 768], embed_ind [b, 32]
        bypass_codebook = self.cfg.stage2.bypass_codebook

        if bypass_codebook:
            # Bypass codebook
            if is_validation:
                print("Bypass codebook")
            quant = query_output_down
            loss_embed = 0.0
            embed_ind = None
            perplexity = None
        else:
            # Quantize
            if is_validation:
                print("Quantize")
            quant, loss_embed, embed_ind, perplexity = self.image_tokenizer.model.quantize(query_output_down)
            embed_ind = embed_ind.reshape(quant.shape[0], -1)

        # codebook replacement
        replacement_num_batches = self.cfg.stage2.vq.replacement_num_batches
        #if ((self.global_step + 1) % replacement_num_batches == 0) & (self.global_step <= replacement_num_batches - 2 * replacement_num_batches):
        if ((batch_idx + 1) % replacement_num_batches == 0) & self.cfg.stage2.vq.replace_codes:
            num_replaced_codebook = self.image_tokenizer.model.quantize.replace_unused_codebooks(replacement_num_batches)
        else:
            num_replaced_codebook = -1
        #------------------------
        # Stage 2 - 2 : Reconstruction Caual Embedding
        #------------------------

        # quant embedding dimension is [b, 32, 32]
        # decoder_task_layer upscale it to [b, 32, 768]
        # [b, 32, 32] => [b, 32, 768]
        query_output_up = self.image_tokenizer.model.decode_task_layer(quant)

        if self.recon_s:
            pos_embed = self.image_tokenizer.model.pos_embed.repeat(query_output_up.shape[0], 1, 1)
            query_output_up_pos = query_output_up + pos_embed
            for blk in self.image_tokenizer.model.blocks:
                query_output_up_pos = blk(query_output_up_pos)
            recon_s = query_output_up_pos
            loss_recon = self.calculate_rec_loss(recon_s, causal_embeddings)

            query_output_up_pos_image = self.image_tokenizer.model.get_transformer_decoded_embedding(query_output_up)
        else:
            # Transformer decoder
            query_output_up_pos_image = self.image_tokenizer.model.get_transformer_decoded_embedding(query_output_up)
            
            # query_output_up_pos_image should be similar to original causal_embeddings
            # Maximize cosine similarity between query_output_up_pos_image and causal_embeddings

            #loss_recon = F.cosine_similarity(query_output_up, causal_embeddings).mean()
            loss_recon = self.calculate_rec_loss(query_output_up, causal_embeddings)
        
        #------------------------
        # Stage 2 - 3 : Reconstruction Generation Embedding
        #------------------------

        # MLP
        # query_output_up = causal_embeddings
        reverse_output_proj = self.image_tokenizer.model.get_mlp_decoded_embedding(query_output_up_pos_image)

        if self.cfg.optimizer.precision == '32':
            gt_img_clip_embeddings = self.get_clip_img_embedding(img.half()).float()
        else:
            gt_img_clip_embeddings = self.get_clip_img_embedding(img)
    
        loss_generation_embed = F.mse_loss(reverse_output_proj, gt_img_clip_embeddings)

        # dict for logging
        loss_dict = {}

        # logging unweighted loss
        loss_dict["unweighted_loss_generation_embed"] = loss_generation_embed
        loss_dict["unweighted_loss_recon"] = loss_recon
        if not bypass_codebook:
            loss_dict["unweighted_loss_embed"] = loss_embed

        # Loss balance
        loss_total = 0.0
        weighted_loss_generation_embed = loss_generation_embed * self.cfg.stage2.loss_weight.loss_generation_embedding
        weighted_loss_recon = loss_recon * self.cfg.stage2.loss_weight.loss_recon
        weighted_loss_embed = loss_embed * self.cfg.stage2.loss_weight.loss_codebook

        loss_total += weighted_loss_recon + weighted_loss_generation_embed
        # For NSVQ, codebook loss (loss_embed) should be used only for logging
        if self.cfg.stage2.vq.type != 'nsvq':
            loss_total += weighted_loss_embed

        loss_dict["loss_generation_embed"] = weighted_loss_generation_embed
        loss_dict["loss_recon"] = weighted_loss_recon
        loss_dict["loss"] = loss_total

        if not bypass_codebook:
            loss_dict["loss_embed"] = weighted_loss_embed
            loss_dict["perplexity"] = perplexity

        if num_replaced_codebook > 0:
            loss_dict["num_replaced_codebook"] = num_replaced_codebook

        #------------------------
        # Logging
        #------------------------
        generation_embedding_cosine_similarity = F.cosine_similarity(reverse_output_proj, gt_img_clip_embeddings).mean()

        self.logging_train_stage2(generation_embedding_cosine_similarity, loss_dict, is_val=is_validation)

        return loss_total, reverse_output_proj

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

        if self.stage == 1:
            if self.cfg.stage1.use_diffusion_loss:
                # loss, _ = self.get_stage_1_diffusion_loss(batch, batch_idx)
                loss, _ = self.debug_lsd_loss(batch, batch_idx)
            else:
                loss = self.get_stage_1_loss_use_last_token(batch, batch_idx)
        elif self.stage == 2:
            loss, _ = self.get_stage_2_loss(batch, batch_idx)
        
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

        if self.cfg.experiment.stage == 1:
            norms_0 = grad_norm(self.image_tokenizer.model.Qformer.bert.encoder.layer[0].attention.self.value, norm_type=2)
            for norm in norms_0.keys():
                self.logger.experiment.add_scalar(
                    f"grad_norm/image_tokenizer/model/Qformer/bert/encoder/layer/0/attention/self/value/{norm}",
                    norms_0[norm],
                    global_step=self.global_step,
                )
            norms_cross_0 = grad_norm(self.image_tokenizer.model.Qformer.bert.encoder.layer[0].crossattention.self.value, norm_type=2)
            for norm in norms_cross_0.keys():
                self.logger.experiment.add_scalar(
                    f"grad_norm/image_tokenizer/model/Qformer/bert/encoder/layer/0/crossattention/self/value/{norm}",
                    norms_cross_0[norm],
                    global_step=self.global_step,
                )
            try:
                norms_1 = grad_norm(self.image_tokenizer.model.Qformer.bert.encoder.layer[1].attention.self.value, norm_type=2)
                for norm in norms_1.keys():
                    self.logger.experiment.add_scalar(
                        f"grad_norm/image_tokenizer/model/Qformer/bert/encoder/layer/1/attention/self/value/{norm}",
                        norms_1[norm],
                        global_step=self.global_step,
                    )
                norms_7 = grad_norm(self.image_tokenizer.model.Qformer.bert.encoder.layer[7].attention.self.value, norm_type=2)
                for norm in norms_7.keys():
                    self.logger.experiment.add_scalar(
                        f"grad_norm/image_tokenizer/model/Qformer/bert/encoder/layer/7/attention/self/value/{norm}",
                        norms_7[norm],
                        global_step=self.global_step,
                    )
            except:
                pass
        elif self.cfg.experiment.stage == 2:
            codebook_norm = grad_norm(self.image_tokenizer.model.quantize.embedding, norm_type=2)
            for norm in codebook_norm.keys():
                self.logger.experiment.add_scalar(
                    f"grad_norm/image_tokenizer/model/quantize/{norm}",
                    codebook_norm[norm],
                    global_step=self.global_step,
                )

            blocks_norm = grad_norm(self.image_tokenizer.model.blocks, norm_type=2)
            for norm in blocks_norm.keys():
                self.logger.experiment.add_scalar(
                    f"grad_norm/image_tokenizer/model/blocks/{norm}",
                    blocks_norm[norm],
                    global_step=self.global_step,
                )
            
            transformer_decoder_norm = grad_norm(self.image_tokenizer.model.blocks_image, norm_type=2)
            for norm in transformer_decoder_norm.keys():
                self.logger.experiment.add_scalar(
                    f"grad_norm/image_tokenizer/model/blocks_image/{norm}",
                    transformer_decoder_norm[norm],
                    global_step=self.global_step,
                )
            
            generation_mlp_norm = grad_norm(self.image_tokenizer.model.distill_image_proj, norm_type=2)
            for norm in generation_mlp_norm.keys():
                self.logger.experiment.add_scalar(
                    f"grad_norm/image_tokenizer/model/distill_image_proj/{norm}",
                    generation_mlp_norm[norm],
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

        if self.stage == 1:
            if not self.cfg.stage1.use_diffusion_loss:
                self.save_path = f"{tb_log_dir}/histogram"
                os.makedirs(self.save_path, exist_ok=True)

                self.save_path = f"{tb_log_dir}/histogram/epoch_{self.current_epoch}"
                os.makedirs(self.save_path, exist_ok=True)

                self.check_image_text_similarity(batch, batch_idx, save_dir=save_path)
            elif self.cfg.stage1.use_diffusion_loss:
                self.save_path = f"{tb_log_dir}/reconstructed_images"
                os.makedirs(self.save_path, exist_ok=True)
                
                self.save_path = f"{tb_log_dir}/reconstructed_images/epoch_{self.current_epoch}"
                os.makedirs(self.save_path, exist_ok=True)

                img, text, image_id = batch
                # loss, slots = self.get_stage_1_diffusion_loss(batch, batch_idx)
                loss, slots = self.debug_lsd_loss(batch, batch_idx, is_validation=True)

                self.log(
                    "val/loss_diffusion",
                    loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )

                # Change validation scheduler
                cur_scheduler = self.image_tokenizer.diffusion_model.scheduler
                self.image_tokenizer.diffusion_model.scheduler = self.val_schduler

                reconstructed_images = self.image_tokenizer.diffusion_model(
                    prompt_embeds=slots,
                    height=256,
                    width=256,
                    guidance_scale=1.3,
                    # num_inference_steps=25,
                    num_inference_steps=100,
                ).images

                # Return it to original scheduler
                self.image_tokenizer.diffusion_model.scheduler = cur_scheduler

                for img, cur_id in zip(reconstructed_images, image_id):
                    # save PIL image to save_path
                    img.save(f"{self.save_path}/{cur_id}")


        elif self.stage == 2:
            image, captions, image_id = batch
            bypass_codebook = self.cfg.stage2.bypass_codebook

            with torch.no_grad():
                _, image_embeds = self.get_stage_2_loss(batch, batch_idx, is_validation=True)

                reconstructed_images = self.image_tokenizer.diffusion_model(
                    image_embeds=image_embeds,
                    negative_image_embeds=None,
                    guidance_scale=10,
                    noise_level=0,
                    latents=self.image_tokenizer.latents,
                ).images
                
            self.save_path = f"{tb_log_dir}/images/version_{self.logger.version}/epoch_{self.current_epoch}/images"
            os.makedirs(self.save_path, exist_ok=True)

            tensor_images = []

            for img, cur_id in zip(reconstructed_images, image_id):
                # save PIL image to save_path
                img.save(f"{self.save_path}/{cur_id}")

                # For tensorboard logging
                tensor_images.append(self.pil_to_tensor(img).unsqueeze(0))

            tensor_images = torch.cat(tensor_images, dim=0)

            # Check if image is already logged
            if batch_idx not in self.logged_original_image:
                self.logger.experiment.add_images(
                    f"original/image_batch_{batch_idx}",
                    image,
                )

                self.logger.experiment.add_images(
                    f"original/image_batch_{batch_idx}_seed_reconstructed",
                    tensor_images,
                )

                # logging original caption
                for caption in captions:
                    self.logger.experiment.add_text(
                        f"original/gt_text_image_batch_{batch_idx}",
                        caption,
                    )

                self.logged_original_image.add(batch_idx)
            else:
                self.logger.experiment.add_images(
                    f"images/image_batch_{batch_idx}",
                    tensor_images,
                    global_step=self.sample_image_ind,
                )
                self.sample_image_ind += 1

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
        for n, p in self.image_tokenizer.model.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
            num_parameters += p.data.nelement()
        for n, p in self.slot_attention.named_parameters():
            if not p.requires_grad:
                continue
            # if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            #     p_non_wd.append(p)
            else:
                p_wd.append(p)
            num_parameters += p.data.nelement()
        for n, p in self.unet.named_parameters():
            if not p.requires_grad:
                continue
            # if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            #     p_non_wd.append(p)
            else:
                p_wd.append(p)
            num_parameters += p.data.nelement()

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
        wrapper = SEEDTrainingWrapper.load_from_checkpoint(cfg.weight_path, cfg=cfg, strict=False).to(device)
        print("Loaded model from checkpoint")
    else:
        wrapper = SEEDTrainingWrapper(cfg).to(device)
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
