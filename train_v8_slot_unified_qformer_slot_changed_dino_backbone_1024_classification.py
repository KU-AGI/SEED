import os
from typing import Any, List
import torch
from IPython.core.completer import back_latex_name_matcher
from fontTools.misc.plistlib import start_dict
from skimage.restoration.uft import image_quad_norm
from torch.cuda.amp import autocast
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
import time
from tqdm import tqdm

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

import numpy as np

from datamodules.seed_llama_datamodule import SEEDDataModule
from models.seed_llama_tokenizer import ImageTokenizer

from datamodules.imagenet_datamodule import ImageNetDataModule

from calculate_clip_score import calculate_clip_s_for_folder, calculate_lpips_for_folder, calculate_psnr_for_folder, \
    calculate_ssim_for_folder
from utils.config import build_config

from lavis.common.dist_utils import is_dist_avail_and_initialized

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


class SlotTestWrapper(LightningModule):
    """Training wrapper for Slot

    Args:
        LightningModule (cfg, model): model should be ImageTokenizer
    """

    def __init__(self, cfg, class_names, val_length):
        super().__init__()
        self.cfg = cfg
        self.stage = cfg.experiment.stage
        self.class_names = class_names
        self.val_length = val_length
        self.acc = 0.0

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
        
        if hasattr(self.cfg.stage1, "used_sublayer") and self.cfg.stage1.used_sublayer is not None:
            self.image_tokenizer.model.Qformer.bert.encoder.config.num_hidden_layers = self.cfg.stage1.used_sublayer

        dinov2 = torch.hub.load("facebookresearch/dinov2", cfg.stage1.dino_model_name)
        self.backbone = DINOBackbone(dinov2).eval()
        self.visual_embedding_encoder = nn.Linear(1024, 1408)
        self.image_tokenizer.model.visual_encoder = None

        # self.backbone = self.image_tokenizer.model.visual_encoder
        self.out_layer_norm = nn.LayerNorm(768)
        self.out_linear_1024 = nn.Linear(768, 1024)


        # Change number of tokens
        if hasattr(self.cfg.stage1, 'slot_num') and self.cfg.stage1.slot_num is not None:
            query_tokens_sliced = self.image_tokenizer.model.query_tokens[:, :self.cfg.stage1.slot_num, :].contiguous().view(-1, self.cfg.stage1.slot_num, 768)
            self.image_tokenizer.model.query_tokens = nn.Parameter(query_tokens_sliced)
            self.image_tokenizer.model.Qformer.bert.query_length = self.cfg.stage1.slot_num

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
            self.image_feat_proj = nn.Linear(1024, 1024)
            self.text_feat_proj = nn.Linear(768, 1024)
        else:
            self.use_itc = False

        self.text_embeddings = None

    def setup(self, stage):
        # Freeze backbone
        if hasattr(self, "backbone") and self.backbone is not None:
            for param in self.backbone.parameters():
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
        is_check_time = False
        if is_check_time:
            cur_time = time.time()

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

        if is_check_time:
            self.print("backbone time: ", time.time() - cur_time)
            cur_time = time.time()

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

        if is_check_time:
            self.print("Qformer time: ", time.time() - cur_time)
            cur_time = time.time()

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
            _, image_feats = self.get_slot_embedding(batch, batch_idx)
        
        b = image_feats.size(0)

        assert image_feats.size(-1) == 1024

        image_feats = self.image_feat_proj(image_feats)  # [b, 32, 1024]
        image_feats = F.normalize(image_feats, dim=-1)
        image_feats = rearrange(image_feats[:, -1, :], "b d -> b 1 d").contiguous()

        text_feats = self.get_text_feats(batch, batch_idx) # [b, seq, 768]
        text_feats = self.text_feat_proj(text_feats)    # [b, seq, 1024]
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
        is_check_time = False
        # Debug
        if is_check_time:
            self.print("============ One batch start ============")
            start_time = time.time()
            cur_time = time.time()

        noisy_model_input, noise, timesteps, model_input = self.get_diffusion_noisy_model_input(batch)

        # Debug
        if is_check_time:
            self.print("get_diffusion_noisy_model_input time: ", time.time() - cur_time)
            cur_time = time.time()

        logging_dict = {}

        image_feats = self.get_image_feats(batch, batch_idx)

        if is_check_time:
            self.print("get_image_feats time: ", time.time() - cur_time)
            cur_time = time.time()

        slots, slots_1024 = self.get_slot_embedding(batch, batch_idx, image_feats)

        if is_check_time:
            self.print("get_slot_embedding time: ", time.time() - cur_time)
            cur_time = time.time()

        # itc loss calculation
        if self.use_itc:
            loss_itc = self.get_itc_loss_use_last_token(batch, batch_idx, is_validation, slots_1024)
            logging_dict["loss_itc"] = loss_itc

        if is_check_time:
            self.print("get_itc_loss_use_last_token time: ", time.time() - cur_time)
            cur_time = time.time()

        # Diffusion loss calculation

        # Predict the noise residual
        model_pred = self.unet(
            noisy_model_input, timesteps, slots_1024,
        ).sample

        if is_check_time:
            self.print("unet time: ", time.time() - cur_time)
            cur_time = time.time()

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

        if is_check_time:
            self.print("loss_diffusion time: ", time.time() - cur_time)
            cur_time = time.time()

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

        if is_check_time:
            self.print("loss logging time: ", time.time() - cur_time)
            self.print("one batch time: ", time.time() - start_time)

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
            image_feats = self.get_image_feats(batch, batch_idx)

            # GT slots
            slots, slots_1024 = self.get_slot_embedding(batch, batch_idx, image_feats)

        # Quantize slots
        quant, embed_ind, loss_codebook = self.quantize(slots_1024)

        if self.cfg.stage2.vq.vq_type == "residual_vq":
            loss_codebook = loss_codebook.sum()

        logging_dict["loss_codebook"] = loss_codebook

        # Flatten the quantized slots
        embed_ind = embed_ind.reshape(slots.shape[0], -1)

        # Reconstruct the slots
        slots_1024_blocks_applied = self.apply_transformer(quant, self.blocks, self.pos_embed)

        loss_recon = F.mse_loss(slots_1024, slots_1024_blocks_applied, reduction="mean")
        logging_dict["loss_recon"] = loss_recon
        logging_dict["sim_768"] = F.cosine_similarity(slots_1024, slots_1024_blocks_applied, dim=-1).mean()

        # Compute itc loss
        if self.use_itc:
            loss_itc = self.get_itc_loss_use_last_token(batch, batch_idx, is_validation, slots_1024_blocks_applied)
            logging_dict["loss_itc"] = loss_itc

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

        if len(batch) == 3:
            img, text, image_id = batch
        elif len(batch) == 2:
            img, text = batch

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

    @torch.no_grad()
    def on_test_start(self) -> None:
        self.acc = 0
        self.text_embeddings = []
        for class_name in tqdm(self.class_names):
            prompt = f"A photo of {class_name}"
            # Text embedding
            text_tokens = self.image_tokenizer.model.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128,
            )

            text_output = self.image_tokenizer.model.Qformer.bert(
                text_tokens.input_ids.to(self.device),
                attention_mask=text_tokens.attention_mask.to(self.device),
                return_dict=True,
                use_slot=True,
            )

            text_feat = text_output.last_hidden_state  # Only use [CLS] Token
            self.text_embeddings.append(text_feat)

        self.text_embeddings = torch.cat(self.text_embeddings, dim=0)   # [1000, 768]

        self.text_embeddings = self.text_feat_proj(self.text_embeddings)    # [1000, 1024]
        self.text_embeddings = F.normalize(self.text_embeddings, dim=-1)
        self.text_embeddings = self.text_embeddings[:, 0, :].contiguous()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        image_feats = self.get_image_feats(batch, batch_idx)

        slots, slots_1024 = self.get_slot_embedding(batch, batch_idx, image_feats)

        slots_1024 = F.normalize(slots_1024, dim=-1)
        slots_1024_last_token = slots_1024[:, -1, :] # [b, 1024]

        # Compare similarity with text embeddings
        sim = torch.matmul(
            self.text_embeddings,
            rearrange(slots_1024_last_token, "b d -> d b")
        ).T   # [b, 1000]

        #  Get prediction for batch
        _, pred = sim.max(dim=-1)

        # Get accuracy
        self.acc += (pred == batch[2]).float().sum().item()

        loss_itc = F.cross_entropy(sim, batch[2], label_smoothing=0.1)
        self.print(f"Loss ITC: {loss_itc}")

    def on_test_end(self) -> None:
        self.acc = self.acc / 50000
        self.print(f"Accuracy :{self.acc * 100} %")
        print(f"Accuracy :{self.acc * 100} %")

if __name__ == "__main__":
    cfg, cfg_yaml = build_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(cfg.experiment.seed, workers=True)

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    os.makedirs(cfg.result_file_path, exist_ok=True)

    datamodule = ImageNetDataModule(batch_size=64)
    datamodule.setup()

    class_names = datamodule.val_dataset.class_names

    # Debug
    # datamodule = SEEDDataModule(cfg, transform=transform, use_coco_val=cfg.dataset.val_config.use_coco_val)

    val_dataloader = datamodule.val_dataloader()

    # val_dataloader = datamodule.val_dataloader()
    val_length = len(val_dataloader)

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
        log_every_n_steps=cfg.experiment.log_every_n_steps,
        val_check_interval=cfg.experiment.val_check_interval,
        # check_val_every_n_epoch=cfg.experiment.check_val_every_n_epoch,
        enable_checkpointing=cfg.experiment.enable_checkpointing,
        num_sanity_val_steps=cfg.experiment.num_sanity_val_steps,
        precision=str(cfg.optimizer.precision),
        callbacks=[ModelSummary(max_depth=3)],
        accumulate_grad_batches=cfg.experiment.grad_accumulation,
        gradient_clip_val=cfg.experiment.grad_clip_val,
    )

    if cfg.load_weight and cfg.resume:
        raise ValueError("Only checkpoint or finetune")

    if cfg.load_weight:
        print(f"Load weight from {cfg.weight_path}")
        wrapper = SlotTestWrapper.load_from_checkpoint(cfg.weight_path, cfg=cfg, class_names=class_names, val_length=val_length, strict=False)
    else:
        wrapper = SlotTestWrapper(cfg, class_names=class_names)
        if cfg.experiment.stage == 1:
            print(f"Stage 1 init from {cfg.checkpoint_path.model_path}")
        elif cfg.experiment.stage == 2:
            print(f"Stage 2 init from {cfg.weight_path}")

    wrapper.setup("fit")

    trainer.test(wrapper, dataloaders=val_dataloader)

    trainer.strategy.barrier()
