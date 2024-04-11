import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist

import json
from typing import List, Tuple, Dict, Any, Union, Optional, Callable

import hydra
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, seed_everything
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import pyrootutils

import torch.nn.functional as F
from einops import rearrange
import transformers

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.utilities import grad_norm

import numpy as np
import matplotlib.pyplot as plt

from models.seed_qformer.vit import Block
from models.seed_llama_tokenizer import ImageTokenizer

from datamodules.seed_llama_datamodule import SEEDDataModule

from calculate_clip_score import calculate_clip_s_for_folder
from utils.config import build_config

from lavis.models import load_model
from lavis.common.dist_utils import is_dist_avail_and_initialized
from functools import partial

from transformers import CLIPVisionModelWithProjection

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
            # self.image_encoder = self.image_tokenizer.diffusion_model.image_encoder
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
            # self.image_normalizer = self.image_tokenizer.diffusion_model.image_normalizer
            # self.image_noising_scheduler = self.image_tokenizer.diffusion_model.image_noising_scheduler
            self.tokenizer = self.image_tokenizer.diffusion_model.tokenizer
            self.text_encoder = self.image_tokenizer.diffusion_model.text_encoder
            self.unet = self.image_tokenizer.diffusion_model.unet
            self.scheduler = self.image_tokenizer.diffusion_model.scheduler
            self.vae = self.image_tokenizer.diffusion_model.vae
            self.image_processor = self.image_tokenizer.diffusion_model.image_processor

        # For logging
        self.pil_to_tensor = transforms.ToTensor()
        self.sample_image_ind = 0
        self.logged_original_image = set()
        
        self.stage = cfg.experiment.stage
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.recon_s = True

        # For SDS
        self.projection = nn.Linear(768, 1024)

        t_range = [0.2, 0.6]
        # t_range = [0.02, 0.98]
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        # self.alphas = self.image_noising_scheduler.alphas_cumprod  # for convenience

    def setup(self, stage):
        # Setup training parameter
        self.image_tokenizer.model.train()
        for param in self.image_tokenizer.model.parameters():
            param.requires_grad = True

        # Freeze ViT Encoder
        for param in self.image_tokenizer.model.visual_encoder.parameters():
            param.requires_grad = False

        # Diffusion frozen
        if self.image_tokenizer.diffusion_model is not None:
            # Check image_encoder is in diffusion model then freeze it
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            if hasattr(self.image_tokenizer.diffusion_model, "image_normalizer"):
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
            if not self.cfg.load_weight:
                self.random_initialize_stage2_model_weights()
            if self.cfg.stage2.train_unet:
                print("Make unet trainable for image embeds")
                self.make_unet_trainable_for_img_embeds()

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

    def logging_train_stage2(self, loss_dict, is_val=False):
        stage = "val" if is_val else "train"
        for loss_name, loss_value in loss_dict.items():
            self.log(
                f'{stage}/{loss_name}',
                loss_value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
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
        # TODO: Use 'final' causal embedding? Does it mean to use last token embedding?
        # Debug
        image_feats = rearrange(query_output.last_hidden_state[:, -1, :], "b d -> b 1 d").contiguous()
        image_feats = F.normalize(image_feats, dim=-1)

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
            max_length=128,
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

    def _compute_snr(self, timesteps):
        """
        Computes SNR as per
        https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def sds_loss(
            self,
            encoder_hidden_state,
            clean_image,
    ):
        """Score distillation sampling"""
        latents = self.vae.encode(clean_image).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        bsz = latents.shape[0]

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        # timesteps = torch.randint(
        #     self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device
        # )
        timesteps = torch.randint(0, self.num_train_timesteps, (bsz,), dtype=torch.long, device=self.device)

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        if self.scheduler.config.prediction_type == "epsilon":
            targets = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            targets = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type: {self.scheduler.config.prediction_type}")

        # Predict the noise residual and compute loss
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_state).sample

        loss = F.mse_loss(model_pred, targets, reduction='mean')

        return loss

    def clip_loss(self, image_embeds, gt_img_clip_embeddings):
        similarity_target = torch.ones(image_embeds.shape[0], device=image_embeds.device)
        loss_clip = torch.nn.functional.cosine_embedding_loss(image_embeds, gt_img_clip_embeddings, similarity_target)
        return loss_clip

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
        if len(batch) == 3:
            img, text, image_id = batch
        elif len(batch) == 2:
            img, text = batch

        #------------------------
        # Stage 2 - 1 : Codebook Training
        #------------------------
        with torch.no_grad():
            causal_embeddings = self.get_causal_embeddings(img)

        bypass_codebook = self.cfg.stage2.bypass_codebook

        # TODO: query_output should be trained to be similar with text embedding
        # Image embedding is cross attentioned.
        # [b, 32, 32]
        if bypass_codebook:
            query_output_down = causal_embeddings
        else:
            query_output_down = self.image_tokenizer.model.encode_task_layer(causal_embeddings)

        # quant [b, 32, 32], loss_embed [b, 32, 768], embed_ind [b, 32]

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
        if bypass_codebook:
            query_output_up = quant
        else:
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
        # reverse_output_proj = self.image_tokenizer.model.get_mlp_decoded_embedding(query_output_up_pos_image)
        # [b, 32, 768] => [b, 32, 1024]
        encoder_hidden_state = self.projection(query_output_up_pos_image)

        # For debug
        gt_prompt_embeds = self.image_tokenizer.diffusion_model._encode_prompt(
            prompt=text,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        ) 

        gt_loss_sds = self.sds_loss(
            encoder_hidden_state=gt_prompt_embeds,
            clean_image=img,
        )

        loss_sds = self.sds_loss(
            encoder_hidden_state=encoder_hidden_state,
            clean_image=img,
        )

        # dict for logging
        loss_dict = {}

        # logging unweighted loss
        loss_dict["unweighted_loss_sds"] = loss_sds
        loss_dict["unweighted_loss_recon"] = loss_recon
        if not bypass_codebook:
            loss_dict["unweighted_loss_embed"] = loss_embed

        loss_dict["unweighted_gt_loss_sds"] = gt_loss_sds

        # Loss balance
        loss_total = 0.0
        weighted_loss_sds = loss_sds * self.cfg.stage2.loss_weight.loss_sds
        weighted_loss_recon = loss_recon * self.cfg.stage2.loss_weight.loss_recon
        weighted_loss_embed = loss_embed * self.cfg.stage2.loss_weight.loss_codebook

        loss_total += weighted_loss_recon + weighted_loss_sds
        # For NSVQ, codebook loss (loss_embed) should be used only for logging
        if self.cfg.stage2.vq.type != 'nsvq':
            loss_total += weighted_loss_embed

        loss_dict["loss_sds"] = weighted_loss_sds
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

        self.logging_train_stage2(loss_dict, is_val=is_validation)

        return loss_total, (encoder_hidden_state, gt_prompt_embeds)

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
            loss = self.get_stage_1_loss_use_last_token(batch, batch_idx)
        elif self.stage == 2:
            loss, _ = self.get_stage_2_loss(batch, batch_idx)
        
        return loss

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        # {'grad_2.0_norm/weight': 0.0003, 'grad_2.0_norm/bias': 0.0, 'grad_2.0_norm_total': 0.0003}
        if self.cfg.experiment.stage == 1:
            norms_0 = grad_norm(self.image_tokenizer.model.Qformer.bert.encoder.layer[0].attention.self.value, norm_type=2)
            for norm in norms_0.keys():
                self.logger.experiment.add_scalar(
                    f"grad_norm/image_tokenizer/model/Qformer/bert/encoder/layer/0/attention/self/value/{norm}",
                    norms_0[norm],
                    global_step=self.global_step,
                )
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
        return

    @torch.no_grad()
    def pipeline(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.image_tokenizer.diffusion_model.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        # We already projection prompt_embedding
        prompt_embeds = self.image_tokenizer.diffusion_model._encode_prompt(
            prompt=None,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.image_tokenizer.diffusion_model.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.image_tokenizer.diffusion_model.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.image_tokenizer.diffusion_model.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        do_denormalize = [True] * image.shape[0]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        return image

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
            if len(batch) == 3:
                image, captions, image_id = batch
            else:
                image, captions = batch
                image_id = [str(i) for i in range(image.size(0))]

            with torch.no_grad():
                _, encoder_hidden_state = self.get_stage_2_loss(
                    batch, batch_idx, is_validation=True
                )

                # reconstructed_images = self.image_tokenizer.diffusion_model(
                #     prompt_embeds=encoder_hidden_state,
                #     guidance_scale=0.0,
                #     # latents=self.image_tokenizer.latents,
                # ).images

                reconstructed_images = self.pipeline(
                    prompt_embeds=encoder_hidden_state[0],
                    guidance_scale=2.0,
                    num_inference_steps=150,
                    generator=torch.Generator().manual_seed(self.cfg.experiment.seed),
                )

                gt_images = self.pipeline(
                    prompt_embeds=encoder_hidden_state[1],
                    guidance_scale=2.0,
                    num_inference_steps=150,
                    generator=torch.Generator().manual_seed(self.cfg.experiment.seed),
                )
                
            save_path = f"{tb_log_dir}/images/version_{self.logger.version}/epoch_{self.current_epoch}/images"
            os.makedirs(save_path, exist_ok=True)

            tensor_images = []

            for img, cur_id in zip(reconstructed_images, image_id):
                # save PIL image to save_path
                img.save(f"{save_path}/{cur_id}")

                # For tensorboard logging
                tensor_images.append(self.pil_to_tensor(img).unsqueeze(0))

            save_path = f"{tb_log_dir}/images/version_{self.logger.version}/epoch_{self.current_epoch}/gt_images"
            os.makedirs(save_path, exist_ok=True)
            for img, cur_id in zip(gt_images, image_id):
                img.save(f"{save_path}/{cur_id}")

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

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-8)
        lr = self.cfg.optimizer.max_lr
        betas = (self.cfg.hyperparameters.beta_1, self.cfg.hyperparameters.beta_2)
        weight_decay = self.cfg.hyperparameters.weight_decay
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

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

    datamodule = SEEDDataModule(cfg, transform=transform)
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    # cfg.experiment.total_training_steps = datamodule.total_training_steps
    cfg.experiment.total_training_steps = 4 * cfg.experiment.max_epochs
    cfg.experiment.num_warmup_steps = 0.03 * cfg.experiment.total_training_steps

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.result_file_path)
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval="step")

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
        # overfit_batches=cfg.experiment.overfit_batches,
        check_val_every_n_epoch=cfg.experiment.check_val_every_n_epoch,
        enable_checkpointing=cfg.experiment.enable_checkpointing,
        num_sanity_val_steps=cfg.experiment.num_sanity_val_steps,
        precision=str(cfg.optimizer.precision),
        callbacks=[ModelSummary(max_depth=3), lr_logger],
        accumulate_grad_batches=cfg.experiment.grad_accumulation,
        gradient_clip_val=cfg.experiment.grad_clip_val,
    )

    if cfg.load_weight and cfg.resume:
        raise ValueError("Only checkpoint or finetune")

    wrapper = SEEDTrainingWrapper(cfg).to(device)

    if cfg.load_weight:
        wrapper.load_from_checkpoint(cfg.weight_path, cfg=cfg)
        print("Loaded model from checkpoint")
    else:
        if cfg.experiment.stage == 1:
            print("Stage 1 init from BLIP-2")
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
            # wrapper, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
            wrapper, train_dataloaders=val_dataloader, val_dataloaders=val_dataloader
        )
    
    trainer.strategy.barrier()
