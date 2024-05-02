import os
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
import pyrootutils

import torch.nn.functional as F
from einops import rearrange
import transformers

from pytorch_lightning import loggers as pl_loggers
from functools import partial
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.utilities import grad_norm

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt

from models.seed_llama_tokenizer import ImageTokenizer

from datamodules.c2f_datamodule import SEEDDataModule

from calculate_clip_score import calculate_clip_s_for_folder
from utils.config import build_config

from lavis.models import load_model
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
            from_pretrained=True if cfg.stage1.init == "SEED" else False,
            vit_precision=cfg.optimizer.vit_precision,
            diffusion_precision=cfg.optimizer.diffusion_precision,
        )

        self.B = None

        self.transform_224 = transforms.Resize((224, 224), antialias=True)

        # For diffusion DDP
        if self.image_tokenizer.diffusion_model is not None:
            self.feature_extractor = self.image_tokenizer.diffusion_model.feature_extractor
            self.image_encoder = self.image_tokenizer.diffusion_model.image_encoder
            self.image_normalizer = self.image_tokenizer.diffusion_model.image_normalizer
            self.image_noising_scheduler = self.image_tokenizer.diffusion_model.image_noising_scheduler
            self.tokenizer = self.image_tokenizer.diffusion_model.tokenizer
            self.text_encoder = self.image_tokenizer.diffusion_model.text_encoder
            self.unet = self.image_tokenizer.diffusion_model.unet
            self.scheduler = self.image_tokenizer.diffusion_model.scheduler
            self.vae = self.image_tokenizer.diffusion_model.vae

            # For SDS
            t_range = [0.2, 0.6]
            # t_range = [0.02, 0.98]
            self.num_train_timesteps = 1000
            self.min_step = int(self.num_train_timesteps * t_range[0])
            self.max_step = int(self.num_train_timesteps * t_range[1])
            self.alphas = self.image_noising_scheduler.alphas_cumprod  # for convenience

        # For logging
        self.pil_to_tensor = transforms.ToTensor()
        self.sample_image_ind = 0
        self.logged_original_image = set()
        
        self.stage = cfg.experiment.stage
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        # For C2F
        c2f_schedule = torch.linspace(1, self.cfg.experiment.min_pos_weight, 32)
        self.c2f_schedule = c2f_schedule.unsqueeze(0)
        self.text_tokenizer = self.image_tokenizer.model.tokenizer


    def setup(self, stage):
        # Setup training parameter
        self.image_tokenizer.model.train()
        for param in self.image_tokenizer.model.parameters():
            param.requires_grad = True

        # Freeze ViT Encoder
        for param in self.image_tokenizer.model.visual_encoder.parameters():
            param.requires_grad = False
        for param in self.image_tokenizer.model.ln_vision.parameters():
            param.requires_grad = False

        # Diffusion frozen
        if self.image_tokenizer.diffusion_model is not None:
            for param in self.image_tokenizer.diffusion_model.image_encoder.parameters():
                param.requires_grad = False
            for param in self.image_tokenizer.diffusion_model.image_normalizer.parameters():
                param.requires_grad = False
            for param in self.image_tokenizer.diffusion_model.text_encoder.parameters():
                param.requires_grad = False
            # In this case, unet is frozen
            for param in self.image_tokenizer.diffusion_model.unet.parameters():
                param.requires_grad = False
            for param in self.image_tokenizer.diffusion_model.vae.parameters():
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
            if not self.cfg.resume:
                self.random_initialize_stage2_model_weights()

        ## make dump folder
        os.makedirs(self.cfg.result_file_path, exist_ok=True)

    def make_unet_trainable_for_img_embeds(self):
        for p in self.image_tokenizer.diffusion_model.unet.parameters():
            p.requires_grad = False

        for p in self.image_tokenizer.diffusion_model.unet.class_embedding.parameters():
            p.requires_grad = True

        for block in self.image_tokenizer.diffusion_model.unet.down_blocks:
            try:
                for resnet in block.resnets:
                    for p in resnet.time_emb_proj.parameters():
                        p.requires_grad = True
            except Exception as e:
                print(e)
                continue

        for block in self.image_tokenizer.diffusion_model.unet.up_blocks:
            try:
                for resnet in block.resnets:
                    for p in resnet.time_emb_proj.parameters():
                        p.requires_grad = True
            except Exception as e:
                print(e)
                continue

        for resnet in self.image_tokenizer.diffusion_model.unet.mid_block.resnets:
            for p in resnet.time_emb_proj.parameters():
                p.requires_grad = True

    def random_initialize_stage2_model_weights(self):
        """Random initialize stage 2 model weights
        """
        print("Random initialize stage 2 model weights")
        for param in self.image_tokenizer.model.blocks.parameters():
            param.requires_grad = False
        self.image_tokenizer.model.blocks.to("cpu")
        # unFreeze stage 2 model and initialize with random weights
        for param in self.image_tokenizer.model.encode_task_layer.parameters():
            # nn.init.xavier_uniform_(param)
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
            for idx in range(self.B):
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
        return self.image_tokenizer.model.get_causal_embeddings(image)

    def forward_stage_2(self, batch, batch_idx: int, bypass_codebook=False):
        """_summary_
        Original forward function for stage 2        
        Just to see how the forward function works

        Args:
            batch (_type_): _description_
            batch_idx (int): _description_

        Returns:
            _type_: _description_
        """        

        # Causal embedding is trained in stage 1.
        # [b, 32, 768]
        if len(batch) == 3:
            image, text, image_id = batch
        elif len(batch) == 2:
            image, text = batch

        with torch.no_grad():
            causal_embeddings = self.get_causal_embeddings(image)

        # [b, 32, 768] = > [b, 32, 32]
        query_output_down = self.image_tokenizer.model.encode_task_layer(causal_embeddings)

        # quant [b, 32, 32], loss_embed [b, 32, 768], embed_ind [b, 32]
        if bypass_codebook:
            # Bypass codebook
            print("Bypass codebook")
            quant = query_output_down
            loss_embed = None
            embed_ind = None
        else:
            # Quantize
            print("Quantize")
            quant, loss_embed, embed_ind = self.image_tokenizer.model.quantize(query_output_down)
            embed_ind = embed_ind.reshape(quant.shape[0], -1)

        if bypass_codebook:
            # # [b, 32, 32] => [b, 32, 768]
            query_output_up = self.image_tokenizer.model.decode_task_layer(quant)
        else:
            quant_embedding = self.image_tokenizer.model.quantize.get_codebook_entry(embed_ind)
            # # [b, 32, 32] => [b, 32, 768]
            query_output_up = self.image_tokenizer.model.decode_task_layer(quant_embedding)

        # [b, 32, 768] => [b, 32, 768]
        query_output_up = self.image_tokenizer.model.get_transformer_decoded_embedding(query_output_up)

        # [b, 32, 768] => [b, 32, 32] => [b, 1024]
        reverse_output_proj = self.image_tokenizer.model.get_mlp_decoded_embedding(query_output_up)

        return reverse_output_proj

    def logging_train_stage2(self, clip_cosine_similarity, loss_dict):
        self.log(
            "train/clip_cosine_similarity",
            clip_cosine_similarity,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        for loss_name, loss_value in loss_dict.items():
            self.log(
                f'train/{loss_name}',
                loss_value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
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

    def get_qformer_tokens(self, image):
        with torch.no_grad():
            image_embeds = self.image_tokenizer.model.ln_vision(
                self.image_tokenizer.model.visual_encoder(image)
            )
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        query_tokens = self.image_tokenizer.model.query_tokens.expand(image_embeds.shape[0], -1, -1)

        # Assume image_embeds.shape[0] is the batch size (b) and you have 32 tokens (n)
        b, n, _ = query_tokens.shape

        query_output = self.image_tokenizer.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        return query_output

    def get_stage1_loss_c2f(self, batch, batch_idx: int):
        if len(batch) == 4:
            image, pos_ids, text_tokens, text_attention_masks = batch
        else:
            image, pos_ids, text_tokens, text_attention_masks, captions = batch
        b = image.shape[0]

        # [b, 32, 768]
        query_output = self.get_qformer_tokens(image)
        # Choose the token with the same index as pos_ids
        # [b, 2, 768]
        image_feats = torch.stack([query_output.last_hidden_state[i].index_select(0, pos_ids[i]) for i in range(b)])
        image_feats = F.normalize(image_feats, dim=-1)

        batch_size, num_positive_samples, token_lens = text_tokens.shape

        text_output = self.image_tokenizer.model.Qformer.bert(
            text_tokens.reshape(batch_size * num_positive_samples, -1),
            attention_mask=text_attention_masks.reshape(batch_size * num_positive_samples, -1),
            return_dict=True,
        )

        # CLS token
        # [batch_size * num_positive_samples, embed_dim]
        text_feats = F.normalize(text_output.last_hidden_state[:, 0, :], dim=-1)
        # [batch_size, num_positive_samples, embed_dim]
        text_feats = rearrange(text_feats, "(bs n) d -> bs n d", bs=b).contiguous()

        ###============== Image-text Contrastive ===================###
        # [batch_size*num_gpu, num_positive_samples, embed_dim]
        image_feats_all = self.all_gather_with_grad(image_feats)

        # [batch_size*num_gpu, num_positive_samples, embed_dim]
        text_feats_all = self.all_gather_with_grad(text_feats)

        mat_image = rearrange(image_feats, "bs n d -> bs 1 1 n d")
        mat_text_all = rearrange(text_feats_all, "(bs ngpus) n d -> (bs ngpus) n d 1", bs=b)
        sim_i2t = torch.matmul(mat_image, mat_text_all).squeeze(-1) / self.temp
        sim_i2t = rearrange(sim_i2t, "bs1 bs2 n1 n2 -> bs1 (bs2 n1) n2", bs1=b)

        mat_text = rearrange(text_feats, "bs n d -> bs 1 1 n d")
        mat_image_all = rearrange(image_feats_all, "(bs ngpus) n d -> (bs ngpus) n d 1", bs=b)
        sim_t2i = torch.matmul(mat_text, mat_image_all).squeeze(-1) / self.temp
        sim_t2i = rearrange(sim_t2i, "bs1 bs2 n1 n2 -> bs1 (bs2 n1) n2", bs1=b)

        # c2f_schedule is [1, 0.97, ..., 0.129, 0.1]
        c2f_schedule = self.c2f_schedule.repeat(b, 1).to(self.device)
        with torch.no_grad():
            # Set for batch, where is the positive sample
            # 1 means positive, 0 means negative
            # [b, num_positive_samples, b]
            pos_or_neg = torch.eye(b, device=self.device).unsqueeze(1).repeat(1, num_positive_samples, 1)
            labels = []
            weights = []
            for i in range(num_positive_samples):
                # Positivie sample in 
                target_pos_idx = torch.LongTensor([[i]] * b).to(self.device)
                # Get the ids of the positive samples (Random choice between 0 ~ 31)
                # pos_ids : [b, num_positive_samples]
                # target_pos_ids : [b, 1] (Choose ith of pos_ids)
                target_pos_ids = pos_ids.gather(1, target_pos_idx)
                # difference between pos_ids and target_pos_ids
                abs_diff = torch.abs(pos_ids - target_pos_ids)
                # Get the schedule for the difference
                schedule = c2f_schedule.gather(1, abs_diff)
                # Make the schedule to be normalized
                # [b, num_positive_samples]
                schedule = schedule / schedule.sum(-1, keepdim=True)
                # weight : [b, num_positive_samples, b] 
                # Hadamard product between pos_or_neg and schedule
                # pos_or_neg is diagonal matrix, so each component of schedule is multiplied by the corresponding diagonal element
                weight = pos_or_neg * schedule.unsqueeze(-1)
                # weight.split(1) : list of [num_positive_sample, b] (len : b)
                # torch.cat(weight.split(1), dim=1) : [num_positive_samples * b, b]
                # weight : [b, num_positive_samples * b]
                weight = torch.cat(weight.split(1), dim=1).squeeze().T
                weights.append(weight)
                label = torch.cat(pos_or_neg.split(1), dim=1).squeeze().T
                labels.append(label)
            # [b, num_positive_sample * b, num_positive_sample]
            local_labels = torch.stack(labels, dim=-1)
            local_weigths = torch.stack(weights, dim=-1)
            all_labels = []
            all_weights = []

            local_rank = dist.get_rank()
            world_size = torch.distributed.get_world_size()
            for rank in range(world_size):
                if rank == local_rank:
                    all_labels.append(local_labels)
                    all_weights.append(local_weigths)
                else:
                    all_labels.append(torch.zeros_like(local_labels))
                    all_weights.append(torch.zeros_like(local_weigths))
            pos_labels = torch.cat(all_labels, dim=1)
            neg_labels = 1 - pos_labels
            weights = torch.cat(all_weights, dim=1)

            label_smoothing = self.cfg.experiment.label_smoothing
            smoothed_pos_labels = torch.clip(pos_labels - label_smoothing, min=0)

            if label_smoothing > 0:
                soft_label = label_smoothing / ((b * num_positive_samples * world_size) - num_positive_samples)
                smoothed_neg_labels = neg_labels * soft_label
            else:
                smoothed_neg_labels = neg_labels

        sim_i2t = F.softmax(sim_i2t, dim=1)
        sim_t2i = F.softmax(sim_t2i, dim=1)

        prob = (sim_i2t + sim_t2i)/2
        pos_loss = (smoothed_pos_labels * weights * -torch.log(prob)).sum(1).mean(0)
        neg_loss = (smoothed_neg_labels * -torch.log(1 - prob)).sum(1).mean(0)

        for i in range(num_positive_samples):
            self.log(
                f"train/loss_pos_itc_{i+1}",
                pos_loss[i],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

            self.log(
                f"train/loss_neg_itc_{i+1}",
                neg_loss[i],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        pos_loss = pos_loss.mean()
        neg_loss = neg_loss.mean()

        self.log(
                "train/loss_pos_itc",
                pos_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        self.log(
                "train/loss_neg_itc",
                neg_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        # selected_sim_i2t = torch.stack([sim_i2t[i].index_select(1, target_pos_idx[i]) for i in range(self.B)])
        # selected_sim_i2t = selected_sim_i2t.squeeze(-1)
        # selected_sim_t2i = torch.stack([sim_t2i[i].index_select(1, target_pos_idx[i]) for i in range(self.B)])
        # selected_sim_t2i = selected_sim_t2i.squeeze(-1)

        # loss_c2f_itc_pos = (
        #     F.cross_entropy(sim_i2t, all_labels, label_smoothing=self.cfg.experiment.label_smoothing, reduction='none')
        #     + F.cross_entropy(sim_t2i, all_labels, label_smoothing=self.cfg.experiment.label_smoothing, reduction='none')
        # ) / 2

        loss = pos_loss + neg_loss
        self.log(
                "train/loss_itc",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return loss

    @torch.no_grad()
    def check_image_text_similarity(self, batch, batch_idx: int, save_dir="image_text_similarity"):
        if len(batch) == 3:
            image, text, image_id = batch
        elif len(batch) == 2:
            image, text = batch
        rank = dist.get_rank()

        with torch.no_grad():
            image_embeds = self.image_tokenizer.model.ln_vision(
                self.image_tokenizer.model.visual_encoder(image)
            )
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        query_tokens = self.image_tokenizer.model.query_tokens.expand(image_embeds.shape[0], -1, -1)

        # Assume image_embeds.shape[0] is the batch size (b) and you have 32 tokens (n)
        b, n, _ = query_tokens.shape

        query_output = self.image_tokenizer.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
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

    def sds_loss(
            self,
            image_embeds,
            clean_image,
            guidance_scale=100,
            grad_scale=1,
            prompt=None,
            prompt_embeds=None,
    ):
        """Score distillation sampling"""
        if prompt is None and prompt_embeds is None:
            # prompt = len(image) * [""] if isinstance(image, list) else ""
            # Changed because we get image_embeds as input
            prompt = image_embeds.shape[0] * [""] if isinstance(image_embeds, torch.Tensor) else ""

        # 2. Define call parameters
        batch_size = image_embeds.shape[0]

        device = image_embeds.device

        # Convert images to latent space
        # latents = self.vae.encode(clean_image).latent_dist.sample()
        # latents = latents * self.vae.config.scaling_factor

        # 3. Encode input prompt

        # [b, 77, 1024]
        # Now img2img, prompt_embeds is None
        prompt_embeds = self.image_tokenizer.diffusion_model._encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=None,
            lora_scale=None,
        )

        image_embeds = self.image_tokenizer.diffusion_model._encode_image(
            image=None,
            device=device,
            batch_size=batch_size,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            noise_level=0,
            generator=None,
            image_embeds=image_embeds,
            negative_image_embeds=None,
        )
        do_classifier_free_guidance = True

        latents = self.vae.encode(clean_image).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step, self.max_step + 1, [1], dtype=torch.long, device=device
        )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)

        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            class_labels=image_embeds,
        ).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
        )

        # w(t), sigma_t^2
        self.alphas = self.alphas.to(device)
        w = 1 - self.alphas[t]
        grad = grad_scale * w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        targets = (latents - grad)
        # loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]
        # Why not mean?
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='mean')

        return loss

    def clip_loss(self, image_embeds, gt_img_clip_embeddings):
        similarity_target = torch.ones(image_embeds.shape[0], device=image_embeds.device)
        loss_clip = torch.nn.functional.cosine_embedding_loss(image_embeds, gt_img_clip_embeddings, similarity_target)
        return loss_clip

    def get_stage_2_loss_bypass_codebook(self, batch, batch_idx: int):
        """_summary_

        Args:
            batch (_type_): _description_
            batch_idx (int): _description_
        """        
        #------------------------
        # Stage 2 Training
        #------------------------
        img, text = batch

        #------------------------
        # Stage 2 - 1 : Codebook Training
        #------------------------
        with torch.no_grad():
            causal_embeddings = self.get_causal_embeddings(img)
            gt_img_clip_embeddings = self.get_clip_img_embedding(img)

        # TODO: query_output should be trained to be similar with text embedding
        # Image embedding is cross attentioned.
        # Notice: query_output_down is match to clip embedding?
        # [b, 32, 32]
        query_output_down = self.image_tokenizer.model.encode_task_layer(causal_embeddings)

        # quant [b, 32, 32], loss_embed [b, 32, 768], embed_ind [b, 32]
        # bypass code book 
        quant = query_output_down

        #------------------------
        # Stage 2 - 2 : Reconstruction Caual Embedding
        #------------------------

        # quant embedding dimension is [b, 32, 32]
        # decoder_task_layer upscale it to [b, 32, 768]
        # [b, 32, 32] => [b, 32, 768]
        query_output_up = self.image_tokenizer.model.decode_task_layer(quant)

        # Transformer decoder
        query_output_up = self.image_tokenizer.model.get_transformer_decoded_embedding(query_output_up)
        
        # query_output_up_pos_image should be similar to original causal_embeddings
        # Maximize cosine similarity between query_output_up_pos_image and causal_embeddings

        #loss_recon = F.cosine_similarity(query_output_up, causal_embeddings).mean()
        loss_recon = F.mse_loss(query_output_up, causal_embeddings)
        loss_dict = {
            "loss_recon": loss_recon,
        }
        loss_total = self.cfg.experiment.recon_loss_weight * loss_recon

        #------------------------
        # Stage 2 - 3 : Reconstruction Generation Embedding
        #------------------------

        # MLP
        # query_output_up = causal_embeddings
        image_embeds = self.image_tokenizer.model.get_mlp_decoded_embedding(query_output_up)
        gt_img_clip_embeddings.requires_grad = False

        sds_loss_weight = self.cfg.experiment.sds_loss_weight * self.sds_loss_weights[self.global_step]
        if self.cfg.experiment.clip_loss_weight > 0:
            loss_clip = F.mse_loss(image_embeds, gt_img_clip_embeddings)
            # loss_clip = self.clip_loss(image_embeds, gt_img_clip_embeddings)
            loss_dict['clip_loss'] = loss_clip
            _loss_clip = self.cfg.experiment.clip_loss_weight * loss_clip
            if self.cfg.experiment.cross_annealing:
                _loss_clip *= (1 - sds_loss_weight)
            loss_total += _loss_clip

        if sds_loss_weight > 0:
            loss_sds = self.sds_loss(
                image_embeds=image_embeds,
                clean_image=img,
                guidance_scale=10,
                grad_scale=1,
            )

            loss_dict['loss_sds'] = loss_sds
            loss_dict['sds_weight'] = sds_loss_weight
            loss_total += sds_loss_weight * loss_sds

        loss_dict['loss'] = loss_total
        #------------------------
        # Logging
        #------------------------
        with torch.no_grad():
            clip_cosine_similarity = F.cosine_similarity(image_embeds, gt_img_clip_embeddings).mean()

        self.logging_train_stage2(clip_cosine_similarity, loss_dict)

        return loss_total

    def get_stage_2_loss(self, batch, batch_idx: int):
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
        quant, loss_embed, embed_ind = self.image_tokenizer.model.quantize(query_output_down)
        
        #------------------------
        # Stage 2 - 2 : Reconstruction Caual Embedding
        #------------------------

        # quant embedding dimension is [b, 32, 32]
        # decoder_task_layer upscale it to [b, 32, 768]
        # [b, 32, 32] => [b, 32, 768]
        query_output_up = self.image_tokenizer.model.decode_task_layer(quant)

        # Transformer decoder
        query_output_up = self.image_tokenizer.model.get_transformer_decoded_embedding(query_output_up)
        
        # query_output_up_pos_image should be similar to original causal_embeddings
        # Maximize cosine similarity between query_output_up_pos_image and causal_embeddings

        #loss_recon = F.cosine_similarity(query_output_up, causal_embeddings).mean()
        loss_recon = F.mse_loss(query_output_up, causal_embeddings)
        
        #------------------------
        # Stage 2 - 3 : Reconstruction Generation Embedding
        #------------------------

        # MLP
        # query_output_up = causal_embeddings
        reverse_output_proj = self.image_tokenizer.model.get_mlp_decoded_embedding(query_output_up)

        gt_img_clip_embeddings = self.get_clip_img_embedding(img)
    
        loss_generation_embed = F.mse_loss(reverse_output_proj, gt_img_clip_embeddings)

        loss_total = loss_embed + loss_recon + loss_generation_embed
        loss_total = loss_total.mean()

        # loss_dict = {"loss_embed": loss_embed, "loss_recon": loss_recon,
        #         "loss_generation_embed": loss_generation_embed,
        #         "loss": loss_total}
        
        loss_dict = {"loss_generation_embed": loss_generation_embed,
                     "loss_embed": loss_embed,
                     "loss_recon": loss_recon,
                     "loss": loss_total}

        #------------------------
        # Logging
        #------------------------
        generation_embedding_cosine_similarity = F.cosine_similarity(reverse_output_proj, gt_img_clip_embeddings).mean()

        self.logging_train_stage2(generation_embedding_cosine_similarity, loss_dict)

        return loss_total

    def on_train_start(self):
        print(f"\n====Traing Stage {self.stage}====")
        if self.stage == 2 and self.cfg.stage2.bypass_codebook:
            print("\n====Bypass codebook====")

        print("Save config")
        self.save_config()

    def training_step(self, batch, batch_idx: int):
        self.B = batch[0].shape[0]

        if self.stage == 1:
            loss = self.get_stage1_loss_c2f(batch, batch_idx)
        elif self.stage == 2:
            if self.cfg.stage2.bypass_codebook:
                loss = self.get_stage_2_loss_bypass_codebook(batch, batch_idx)
            else:
                loss = self.get_stage_2_loss(batch, batch_idx)
        
        return loss

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        # {'grad_2.0_norm/weight': 0.0003, 'grad_2.0_norm/bias': 0.0, 'grad_2.0_norm_total': 0.0003}
        if self.cfg.experiment.stage == 1:
            norms_0 = grad_norm(self.image_tokenizer.model.Qformer.bert.encoder.layer[0].attention.self.value,
                                norm_type=2)
            for norm in norms_0.keys():
                self.logger.experiment.add_scalar(
                    f"grad_norm/image_tokenizer/model/Qformer/bert/encoder/layer/0/attention/self/value/{norm}",
                    norms_0[norm],
                    global_step=self.global_step,
                )
            norms_1 = grad_norm(self.image_tokenizer.model.Qformer.bert.encoder.layer[1].attention.self.value,
                                norm_type=2)
            for norm in norms_1.keys():
                self.logger.experiment.add_scalar(
                    f"grad_norm/image_tokenizer/model/Qformer/bert/encoder/layer/1/attention/self/value/{norm}",
                    norms_1[norm],
                    global_step=self.global_step,
                )
            norms_7 = grad_norm(self.image_tokenizer.model.Qformer.bert.encoder.layer[7].attention.self.value,
                                norm_type=2)
            for norm in norms_7.keys():
                self.logger.experiment.add_scalar(
                    f"grad_norm/image_tokenizer/model/Qformer/bert/encoder/layer/7/attention/self/value/{norm}",
                    norms_7[norm],
                    global_step=self.global_step,
                )
        elif self.cfg.experiment.stage == 2:
            codebook_norm = grad_norm(self.image_tokenizer.model.quantize.embedding, norm_type=2)
            for norm in codebook_norm.keys():
                self.logger.experiment.add_scalar(
                    f"grad_norm/image_tokenizer/model/quantize/{norm}",
                    codebook_norm[norm],
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
        os.makedirs(f"{self.cfg.result_file_path}/{self.current_epoch}", exist_ok=True)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int, save_path=None):
        if self.logger is not None and isinstance(self.logger, pl.loggers.TensorBoardLogger):
            tb_log_dir = self.logger.log_dir
        else:
            tb_log_dir = self.cfg.result_file_path  # Fallback directory if logger is not set

        if self.stage == 1:
            save_path = f"{tb_log_dir}/histogram"
            os.makedirs(save_path, exist_ok=True)

            save_path = f"{tb_log_dir}/histogram/epoch_{self.current_epoch}"
            os.makedirs(save_path, exist_ok=True)

            self.check_image_text_similarity(batch, batch_idx, save_dir=save_path)
        elif self.stage == 2:
            image, captions, image_id = batch
            bypass_codebook = self.cfg.stage2.bypass_codebook

            with torch.no_grad():
                image_embeds = self.forward_stage_2(batch, batch_idx, bypass_codebook)
                reconstructed_images = self.image_tokenizer.diffusion_model(
                    image_embeds=image_embeds,
                    negative_image_embeds=None,
                    guidance_scale=10,
                    noise_level=0,
                    latents=self.image_tokenizer.latents,
                ).images
                
            save_path = f"{tb_log_dir}/images/version_{self.logger.version}/epoch_{self.current_epoch}/images"
            os.makedirs(save_path, exist_ok=True)

            tensor_images = []

            for img, cur_id in zip(reconstructed_images, image_id):
                # save PIL image to save_path
                img.save(f"{save_path}/{cur_id}")

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
        generated_image_dir = f"{tb_log_dir}/images/version_{self.logger.version}/epoch_{self.current_epoch}/images"
        clip_score = calculate_clip_s_for_folder(original_image_dir, generated_image_dir)

        print(f"clip score: {clip_score}")
        self.log_dict({
            'clip_score': clip_score,
        },on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.log(
            "clip_score_coco_karpathy",
            clip_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def configure_optimizer(self):
        # TODO make optimizer class and configurations
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
        print(f"number of parameters: {num_parameters}")

        lr = self.cfg.optimizer.max_lr
        betas = (self.cfg.hyperparameters.beta_1, self.cfg.hyperparameters.beta_2)
        weight_decay = self.cfg.hyperparameters.weight_decay
        optim_params = [
            {
                "params": p_wd,
                "weight_decay": float(weight_decay),
            },
            {"params": p_non_wd, "weight_decay": 0},
        ]


        optimizer = torch.optim.AdamW(
            optim_params,
            lr=float(lr),
            weight_decay=float(weight_decay),
            betas=betas,
        )

        return optimizer

    def configure_optimizers(self):
        optimizer = self.configure_optimizer()
        #scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=100, num_training_steps=5000)
        num_training_steps = self.cfg.experiment.total_training_steps
        num_warmup_steps = self.cfg.experiment.num_warmup_steps
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        if self.cfg.experiment.sds_loss_weight > 0 and self.cfg.experiment.use_sds_loss_schedule:
            _num_training_steps = num_training_steps // 8
            def f(current_step: int):
                return 1 - max(0.0, float(_num_training_steps - current_step) / float(_num_training_steps))
        else:
            def f(current_step: int):
                return 1

        x = np.arange(0, num_training_steps)
        self.sds_loss_weights = np.array(list(map(f, x)))

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

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.result_file_path)
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval="step")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        # monitor="clip_score_coco_karpathy" if cfg.experiment.stage == 2 else "val/loss_itc",
        # mode="max" if cfg.experiment.stage == 2 else "min",
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
        # val_check_interval=cfg.experiment.val_check_interval,
        check_val_every_n_epoch=cfg.experiment.check_val_every_n_epoch,
        enable_checkpointing=cfg.experiment.enable_checkpointing,
        num_sanity_val_steps=cfg.experiment.num_sanity_val_steps,
        precision=str(cfg.optimizer.precision),
        callbacks=[ModelSummary(max_depth=3), lr_logger] + [
            checkpoint_callback] if cfg.experiment.enable_checkpointing else [],
        accumulate_grad_batches=cfg.experiment.grad_accumulation,
        gradient_clip_val=cfg.experiment.grad_clip_val,
    )

    if cfg.load_weight and cfg.resume:
        raise ValueError("Only checkpoint or finetune")

    wrapper = SEEDTrainingWrapper(cfg).to(device)

    datamodule = SEEDDataModule(cfg, tokenizer=wrapper.text_tokenizer, transform=transform)
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    cfg.experiment.total_training_steps = datamodule.total_training_steps

    if cfg.load_weight:
        wrapper.load_from_checkpoint(cfg.weight_path, cfg=cfg)
        print("Loaded model from checkpoint")

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
