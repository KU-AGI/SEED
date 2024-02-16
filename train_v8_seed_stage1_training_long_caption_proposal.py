import os
from typing import Any, List
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
import torch.distributed as dist

from tqdm import tqdm
import time
import json

import argparse
import time

import hydra
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, seed_everything
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import PIL
from torchvision.transforms.functional import to_pil_image
from utils.config import build_config
import pyrootutils
from datamodules import build_datamodule
from datamodules.tokenizers import TokenizerUtils

from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
import torch.nn.functional as F
from pytorch_lightning.strategies import DDPStrategy, DDPFullyShardedStrategy
import open_clip
from models.seed_llama_tokenizer import ImageTokenizer
import transformers

from pytorch_lightning import loggers as pl_loggers
from functools import partial
from models.seed_qformer.vit import Block
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.utilities import grad_norm

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from einops import rearrange, repeat
import pdb
from calculate_clip_score import calculate_clip_s_for_folder
from coco_dataloader import CocoDataset
from torch.utils.data import DataLoader
from lavis.common.dist_utils import is_dist_avail_and_initialized

from datamodules.compression_datamodule import CompressionDataModule
import matplotlib.pyplot as plt

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

BOI_TOKEN = "<img>"
EOI_TOKEN = "</img>"
IMG_TOKEN = "<img_{:05d}>"

IMG_FLAG = "<image>"
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 8192
IMAGE_ID_SHIFT = 32000

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
        self.image_tokenizer = ImageTokenizer(model_path=cfg.checkpoint_path.model_path,
                            fp16=cfg.optimizer.fp16,
                            from_pretrained=True,
                            diffusion_model_path=cfg.checkpoint_path.diffusion_model_path,
                            load_diffusion=True,
                            is_train_stage_1=False,
                            )

        # My code
        # For make clip embedding directly from [b, 32, 32] to [b, 1024]
        self.depth = 4
        self.embedding_block = nn.ModuleList([
            Block(dim=32,
                    num_heads=16,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6)) for i in range(self.depth)
        ])
        self.embedding_proj = nn.Linear(32 * 32, 1024).to(self.device)

        self.transform_224 = transforms.Resize((224, 224), antialias=True)

        # diffusions
        self.feature_extractor = self.image_tokenizer.diffusion_model.feature_extractor
        self.image_encoder = self.image_tokenizer.diffusion_model.image_encoder
        self.image_normalizer = self.image_tokenizer.diffusion_model.image_normalizer
        self.image_noising_scheduler = self.image_tokenizer.diffusion_model.image_noising_scheduler
        self.tokenizer = self.image_tokenizer.diffusion_model.tokenizer
        self.text_encoder = self.image_tokenizer.diffusion_model.text_encoder
        self.unet = self.image_tokenizer.diffusion_model.unet
        self.scheduler = self.image_tokenizer.diffusion_model.scheduler
        self.vae = self.image_tokenizer.diffusion_model.vae

        # For logging
        self.sample_embed_ind = None
        self.pil_to_tensor = transforms.ToTensor()
        self.sample_image_ind = 0
        self.logged_original_image = set()
        
        self.stage = 1
        self.temp = nn.Parameter(0.07 * torch.ones([]))
    
    def random_initialize_stage2_model_weights(self):
        """Random initialize stage 2 model weights
        """        
        # Random initialize stage 2 model weights
        for param in self.image_tokenizer.model.parameters():
            param.requires_grad = False

        # For fp16
        if self.cfg.optimizer.fp16:
            self.image_tokenizer = self.image_tokenizer.half()
            self.image_encoder = self.image_encoder.half()
            self.embedding_proj = self.embedding_proj.half()
            for blk in self.embedding_block:
                blk = blk.half()

    def setup(self, stage):
        # Setup training parameter
        self.image_tokenizer.model.train()
        for param in self.image_tokenizer.model.parameters():
            param.requires_grad = True

        # Freeze ViT Encoder
        for param in self.image_tokenizer.model.visual_encoder.parameters():
            param.requires_grad = False

        # Diffusion frozen
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
            
        if self.stage == 2:
            for param in self.image_tokenizer.model.parameters():
                param.requires_grad = False
                
            # unFreeze stage 2 model and initialize with random weights
            for param in self.image_tokenizer.model.encode_task_layer.parameters():
                # nn.init.xavier_uniform_(param) 
                nn.init.normal_(param, mean=0.0, std=0.02)              
                param.requires_grad = True 
            for param in self.image_tokenizer.model.quantize.parameters():
                nn.init.normal_(param, mean=0.0, std=0.02)
                param.requires_grad = True
            for param in self.image_tokenizer.model.decode_task_layer.parameters():
                nn.init.normal_(param, mean=0.0, std=0.02)
                param.requires_grad = True
            for param in self.image_tokenizer.model.blocks_image.parameters():
                nn.init.normal_(param, mean=0.0, std=0.02)
                param.requires_grad = True
            for param in self.image_tokenizer.model.image_down.parameters():
                nn.init.normal_(param, mean=0.0, std=0.02)
                param.requires_grad = True
            for param in self.image_tokenizer.model.distill_image_proj.parameters():
                nn.init.normal_(param, mean=0.0, std=0.02)
                param.requires_grad = True
            
        # For fp16
        if self.cfg.optimizer.fp16:
            self.image_tokenizer = self.image_tokenizer.half()
            self.image_encoder = self.image_encoder.half()
            self.embedding_proj = self.embedding_proj.half()
            for blk in self.embedding_block:
                blk = blk.half()

    def save_config(self):
        config_save_path = os.path.join(self.logger.log_dir, "config.yaml")
        with open(config_save_path, "w") as f:
            json.dump(self.cfg, f, indent=4)
        
        # For test training
        # self.image_tokenizer.model.distill_image_proj = nn.Linear(32 * 32, 1024).to(self.device)

    def set_stage2_learnable(self):
        for param in self.image_tokenizer.model.parameters():
            param.requires_grad = False
            
        # unFreeze stage 2 model and initialize with random weights
        for param in self.image_tokenizer.model.encode_task_layer.parameters():
            param.requires_grad = True 
        for param in self.image_tokenizer.model.quantize.parameters():
            param.requires_grad = True
        for param in self.image_tokenizer.model.decode_task_layer.parameters():
            param.requires_grad = True
        for param in self.image_tokenizer.model.blocks_image.parameters():
            param.requires_grad = True
        for param in self.image_tokenizer.model.image_down.parameters():
            param.requires_grad = True
        for param in self.image_tokenizer.model.distill_image_proj.parameters():
            param.requires_grad = True
        

    def load_sample_images(self, dataloader):
        self.sample_images = []
        for idx, batch in enumerate(dataloader):
            if idx > 1:
                break
            else:
                self.sample_images.append(batch.img.to(self.device))
        self.sample_images = torch.cat(self.sample_images, dim=0)

    def get_clip_text_embedding(self, batch_text):
        """CLIP text embedding

        Args:
            batch_text (List): List contains text. [b, 32]

        Returns:
            float: clip text embedding [b, 1024]
        """        
        gt_text_clip_embeddings = []
        b = len(batch_text)
        with torch.no_grad():
            for idx in range(b):
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

    def get_stage2_quant(self, img):
        # Causal embedding is trained in stage 1.
        causal_embeddings = self.get_causal_embeddings(img)
        query_output_up = causal_embeddings

        # [b, 32, 32]
        # query_output_down = self.image_tokenizer.model.encode_task_layer(causal_embeddings)

        # For debug
        loss_embed = None
        embed_ind = None

        query_output_up = self.image_tokenizer.model.get_transformer_decoded_embedding(query_output_up)

        quant = self.image_tokenizer.model.get_mlp_decoded_embedding(query_output_up)


        return quant, loss_embed, embed_ind

    def get_original_stage2_quant(self, img):
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

        #------------------------
        # Stage 2 - 3 : Reconstruction Generation Embedding
        #------------------------
        # MLP
        reverse_output_proj = self.image_tokenizer.model.get_mlp_decoded_embedding(query_output_up)

        return reverse_output_proj, loss_embed, embed_ind
    
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

    @torch.no_grad()
    def check_image_text_similarity(self, batch, batch_idx: int, save_dir="image_text_similarity"):
        device = self.device
        image, text, image_id = batch
        with torch.no_grad():
            image_embeds = self.image_tokenizer.model.ln_vision(
                self.image_tokenizer.model.visual_encoder(image)
            )
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.image_tokenizer.model.query_tokens.expand(image_embeds.shape[0], -1, -1)


        # Assume image_embeds.shape[0] is the batch size (b) and you have 32 tokens (n)
        b, n, _ = query_tokens.shape

        # Step 1: Create a causal mask
        causal_mask = torch.triu(torch.ones((n, n), device=device) * float('-inf'), diagonal=1)
        
        # Step 2: Apply causal mask in attention
        # Add a new dimension to the mask for the batch size and expand it to match the batch size
        causal_mask = causal_mask.unsqueeze(0).expand(b, -1, -1)  # shape: [b, n, n]
        
        query_output = self.image_tokenizer.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
            attention_mask=causal_mask,  # Apply causal mask here
        )

        # Use last hidden state
        # We have 32 tokens, and use last token as image embedding
        # [b, 32, 768]
        image_feats = F.normalize(query_output.last_hidden_state, dim=1)

        text_tokens = self.image_tokenizer.model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        text_output = self.image_tokenizer.model.Qformer.bert(
            text_tokens.input_ids.to(device),
            attention_mask=text_tokens.attention_mask.to(device),
            return_dict=True,
        )

        # CLS token
        # [b, 768]
        text_feat = F.normalize(text_output.last_hidden_state[:, 0, :], dim=1)

        ###============== Image-text Contrastive ===================###

        # Original BLIP-2 loss
        # Compute for each query token
        image_feats_all = self.concat_all_gather(
            image_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = self.concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

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

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

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
        plt.savefig(f"{save_dir}/token_histogram_image_text_batch{batch_idx}.png")
        # plt.show()
        ############################################################

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            rearrange(text_feat, "bs d -> bs 1 1 d"), rearrange(image_feats_all, "bs_X_ngpus n d -> bs_X_ngpus d n")
            # text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

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

        plt.savefig(f"{save_dir}/token_histogram_text_image_batch{batch_idx}.png")

    def get_stage_1_loss_compression_contrastive(self, batch, batch_idx: int, is_validation=False):
        """
            Contrastive loss using last token of the query_output
        Args:
            batch (_type_): _description_
            batch_idx (int): _description_
            is_validation (bool, optional): _description_. Defaults to False.
        """
        # image = self.transform_224(batch.img)
        device = self.device
        image = batch[0].to(device)
        # caption_list is [32, batch_size]
        caption_list = batch[1]
        text = []
        # Flatten caption list (32, batch_size) to (32 * batch_size, 1)
        for caption in caption_list:
            text.extend(list(caption))

        with torch.no_grad():
            image_embeds = self.image_tokenizer.model.ln_vision(
                self.image_tokenizer.model.visual_encoder(image)
            )
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.image_tokenizer.model.query_tokens.expand(image_embeds.shape[0], -1, -1)

        # Assume image_embeds.shape[0] is the batch size (b) and you have 32 tokens (n)
        b, n, _ = query_tokens.shape

        # Step 1: Create a causal mask
        causal_mask = torch.triu(torch.ones((n, n), device=device) * float('-inf'), diagonal=1)
        
        # Step 2: Apply causal mask in attention
        # Add a new dimension to the mask for the batch size and expand it to match the batch size
        causal_mask = causal_mask.unsqueeze(0).expand(b, -1, -1)  # shape: [b, n, n]
        
        query_output = self.image_tokenizer.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
            attention_mask=causal_mask,  # Apply causal mask here
        )

        # Use last hidden state
        # We have 32 tokens, and use last token as image embedding
        # [b, 32, 768]
        # TODO: Use 'final' causal embedding? Does it mean to use last token embedding?
        image_feats = F.normalize(query_output.last_hidden_state, dim=1)

        # text_token shape is [32 * b, max_length]
        # Max length is 1138, but BERT model can handle max_length 512
        # Token longer than 256 is about 1% of the total data
        text_tokens = self.image_tokenizer.model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        # text_output shape is [32 * b, max_length, 768]
        text_output = self.image_tokenizer.model.Qformer.bert(
            text_tokens.input_ids.to(device),
            attention_mask=text_tokens.attention_mask.to(device),
            return_dict=True,
        )

        # CLS token
        # [32 * b, 768]
        text_feat = F.normalize(text_output.last_hidden_state[:, 0, :], dim=1)

        text_feat = rearrange(text_feat, "(n b) d -> b n d", n=32, b=b).contiguous()

        ###============== Image-text Contrastive ===================###
        # Compute for each query token
        image_feats_all = self.concat_all_gather(
            image_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = self.concat_all_gather(text_feat)  # [batch_size*num_gpu, num_query_tokens, embed_dim]

        # image_feats.unsqueeze(1) : [batch_size, 1, num_query_tokens, embed_dim]
        # text_feat_all.unsqueeze(-1) : [batch_size*num_gpu, embed_dim, 1] => broadcast to [batch_size, batch_size*num_gpu, embed_dim, 1]
        # Last two dimensions are broadcasted to all other dimensions
        # [j, 1, n, m] x [k, m, p] => [j, k, n, p]
        # https://pytorch.org/docs/stable/generated/torch.matmul.html
        # sim_q2t : [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_q2t = torch.matmul(
            rearrange(image_feats, "bs n d -> bs 1 n d"), rearrange(text_feat_all, "(bs ngpus) n d -> (bs ngpus) d n", bs=b)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        # sim_i2t_each_token : [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_i2t_each_token = torch.diagonal(sim_q2t, dim1=-2, dim2=-1)
        sim_i2t_each_token = sim_i2t_each_token / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            rearrange(text_feat, "bs n d -> bs 1 n d"), rearrange(image_feats_all, "(bs ngpus) n d -> (bs ngpus) d n", bs=b)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens, num_query_tokens]

        # First text compare to first image token
        # Second text compare to second image token
        sim_t2i_each_token = torch.diagonal(sim_t2q, dim1=-2, dim2=-1)
        sim_t2i_each_token = sim_t2i_each_token / self.temp  # [batch_size, batch_size*num_gpu]

        rank = dist.get_rank()
        bs = image.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            image.device
        )

        targets = repeat(targets, "bs -> bs n", n=32)

        loss_itc = (
            F.cross_entropy(sim_i2t_each_token, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i_each_token, targets, label_smoothing=0.1)
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

    def forward_stage_2(self,batch, batch_idx: int):
        """_summary_
        Original forward function for stage 2        

        Args:
            batch (_type_): _description_
            batch_idx (int): _description_

        Returns:
            _type_: _description_
        """        

        # Causal embedding is trained in stage 1.
        # [b, 32, 768]
        causal_embeddings = self.get_causal_embeddings(batch.img)

        # [b, 32, 768] = > [b, 32, 32]
        query_output_down = self.image_tokenizer.model.encode_task_layer(causal_embeddings)

        # quant [b, 32, 32], loss_embed [b, 32, 768], embed_ind [b, 32]
        quant, loss_embed, embed_ind = self.image_tokenizer.model.quantize(query_output_down)

        embed_ind = embed_ind.reshape(quant.shape[0], -1)

        # [b, 32, 32] => [b, 32, 768]
        query_output_up = self.image_tokenizer.model.decode_task_layer(quant)

        quant_embedding = self.image_tokenizer.model.quantize.get_codebook_entry(embed_ind)

        # [b, 32, 32] => [b, 32, 768]
        query_output_up = self.image_tokenizer.model.decode_task_layer(quant_embedding)

        # [b, 32, 768] => [b, 32, 768]
        query_output_up = self.image_tokenizer.model.get_transformer_decoded_embedding(query_output_up)

        # [b, 32, 768] => [b, 32, 32] => [b, 1024]
        reverse_output_proj = self.image_tokenizer.model.get_mlp_decoded_embedding(query_output_up)

        return reverse_output_proj

    def get_stage_2_loss(self, batch, batch_idx: int, is_validation=False):
        """_summary_

        Args:
            batch (_type_): _description_
            batch_idx (int): _description_
        """        
        #------------------------
        # Stage 2 Training
        #------------------------
        img = self.transform_224(batch.img)

        #------------------------
        # Stage 2 - 1 : Codebook Training
        #------------------------
        causal_embeddings = self.get_causal_embeddings(img)

        # TODO: query_output should be trained to be similar with text embedding
        # Image embedding is cross attentioned.
        # Notice: query_output_down is match to clip embedding?
        '''
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
        '''
        

        # bypass the quantization
        query_output_up = causal_embeddings

        # Transformer decoder
        query_output_up = self.image_tokenizer.model.get_transformer_decoded_embedding(query_output_up)
        
        

        # query_output_up_pos_image should be similar to original causal_embeddings
        # Maximize cosine similarity between query_output_up_pos_image and causal_embeddings

        # loss_recon = F.normalize(F.cosine_similarity(query_output_up, causal_embeddings)).mean()
        loss_recon = F.mse_loss(query_output_up, causal_embeddings)
        
        #------------------------
        # Stage 2 - 3 : Reconstruction Generation Embedding
        #------------------------

        # MLP
        # query_output_up = causal_embeddings
        reverse_output_proj = self.image_tokenizer.model.get_mlp_decoded_embedding(query_output_up)

        gt_img_clip_embeddings = self.get_clip_img_embedding(img)
    
        loss_generation_embed = F.mse_loss(reverse_output_proj, gt_img_clip_embeddings)

        # loss_total = loss_embed - loss_recon + loss_generation_embed
        # loss_total = loss_embed + loss_recon + loss_generation_embed
        loss_total = loss_recon + loss_generation_embed
        loss_total = loss_total.mean()

        loss_dict = {"loss_recon": loss_recon,
                    "loss_generation_embed": loss_generation_embed,
                    "loss": loss_total}
        
        # loss_dict = {"loss_generation_embed": loss_generation_embed,
        #              "loss": loss_total}

        #------------------------
        # Logging
        #------------------------
        generation_embedding_cosine_similarity = F.cosine_similarity(reverse_output_proj, gt_img_clip_embeddings).mean()

        self.logging_train(generation_embedding_cosine_similarity, loss_dict)

        norms_0 = grad_norm(self.image_tokenizer.model.decode_task_layer, norm_type=2)
        for norm in norms_0.keys():
            self.logger.experiment.add_scalar(
                f"grad_norm/image_tokenizer/model/decode_task_layer/{norm}",
                norms_0[norm],
                global_step=self.global_step,
            )

        return loss_total

    def logging_train(self, generation_embedding_cosine_similarity, loss_dict):
        self.log(
            "train/generation_embedding_cosine_similarity",
            generation_embedding_cosine_similarity,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        '''
        self.log(
            "train/codebook_loss_embed",
            loss_dict["loss_embed"].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        '''

        self.log(
            "train/reconstruction_loss",
            loss_dict["loss_recon"].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train/generation_embed_loss",
            loss_dict["loss_generation_embed"].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
    
    def logging_val(self, quant, loss_embed, gt_img_clip_embeddings, loss):
        self.log(
            "val/generation_embed_loss",
            loss,
            sync_dist=True,
        )

        self.log(
            "val/codebook_loss_embed",
            loss_embed.mean(),
            sync_dist=True,
        )
        # Cosine similarity logging
        self.log(
            "val/stage_2_codebook_cosine_similarity",
            F.cosine_similarity(quant, gt_img_clip_embeddings).mean(),
            sync_dist=True,
        )
    
    def training_step(self, batch, batch_idx: int):
        # gt_text is a list of string
        # Encoding text in list to ascii
        #batch.gt_txt = [[text[0].encode("ascii", "ignore").decode()] for text in batch.gt_txt]

        # image, captions = batch
        # self.check_image_text_similarity((image, captions[-1], None), batch_idx, save_dir="check_dci_similarity")
        # return

        stage_1_loss = self.get_stage_1_loss_compression_contrastive(batch, batch_idx)
        return stage_1_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int):
        if self.logger is not None and isinstance(self.logger, pl.loggers.TensorBoardLogger):
            tb_log_dir = self.logger.log_dir
        else:
            tb_log_dir = self.cfg.result_file_path  # Fallback directory if logger is not set

        if batch_idx < 10:
            save_path = f"{tb_log_dir}/histogram"
            os.makedirs(save_path, exist_ok=True)
            save_path = f"{tb_log_dir}/histogram/epoch_{self.current_epoch}"
            os.makedirs(save_path, exist_ok=True)
            self.check_image_text_similarity(batch, batch_idx, save_dir=save_path)

        image, captions, image_id = batch
        image_embeds, _, _ = self.get_original_stage2_quant(image)

        # with torch.no_grad():
        #     reconstructed_images = self.image_tokenizer.diffusion_model(
        #         image_embeds=image_embeds,
        #         negative_image_embeds=None,
        #         guidance_scale=10,
        #         noise_level=0,
        #         latents=self.image_tokenizer.latents,
        #     ).images

        # Assuming image_embeds is your tensor of embeddings
        batch_size, _ = image_embeds.shape
        small_batch_size = 1  # Divide the batch into N smaller batches

        reconstructed_images = []
        for i in tqdm(range(0, batch_size, small_batch_size)):
            small_batch_embeds = image_embeds[i:i+small_batch_size]
            with torch.no_grad():
                small_batch_reconstructed_images = self.image_tokenizer.diffusion_model(
                    image_embeds=small_batch_embeds,
                    negative_image_embeds=None,
                    guidance_scale=10,
                    noise_level=0,
                    latents=self.image_tokenizer.latents,
                ).images
            reconstructed_images.extend(small_batch_reconstructed_images)

        # Construct the save directory path
        save_path = f"{tb_log_dir}/images/epoch_{self.current_epoch}/images"
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

        # logging weight distribution to check if weight is updated (gradient is flowing)
        self.logger.experiment.add_histogram(
            "weight_distribution/image_tokenizer/model/Qformer/bert/encoder/layer/0/attention/self/value",
            next(self.image_tokenizer.model.Qformer.bert.encoder.layer[0].attention.self.value.parameters()),
            global_step=self.global_step,
        )

        self.logger.experiment.add_histogram(
            "weight_distribution/image_tokenizer/model/Qformer/bert/encoder/layer/1/attention/self/value",
            next(self.image_tokenizer.model.Qformer.bert.encoder.layer[1].attention.self.value.parameters()),
            global_step=self.global_step,
        )

        self.logger.experiment.add_histogram(
            "weight_distribution/image_tokenizer/model/Qformer/bert/encoder/layer/7/attention/self/value",
            next(self.image_tokenizer.model.Qformer.bert.encoder.layer[7].attention.self.value.parameters()),
            global_step=self.global_step,
        )
        
        self.logger.experiment.add_histogram(
            "weight_distribution/embedding_proj",
            next(self.embedding_proj.parameters()),
            global_step=self.global_step,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.optimizer.max_lr,
            betas=(0.9, 0.999),
            weight_decay=1e-8)

        total_training_steps = self.cfg.experiment.total_training_steps
        # scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=total_training_steps * 0.03,
            num_training_steps=total_training_steps,
            )
            # num_cycles=2)

        lr_scheduler_config = {
            "scheduler": scheduler,
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config,}
        
    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        # {'grad_2.0_norm/weight': 0.0003, 'grad_2.0_norm/bias': 0.0, 'grad_2.0_norm_total': 0.0003}
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
        norms_proj = grad_norm(self.embedding_proj, norm_type=2)
        for norm in norms_proj.keys():
            self.logger.experiment.add_scalar(
                f"grad_norm/embedding_proj/{norm}",
                norms_proj[norm],
                global_step=self.global_step,
            )

    def on_validation_epoch_start(self):
        self.save_config()

    def on_validation_epoch_end(self):
        if self.logger is not None and isinstance(self.logger, pl.loggers.TensorBoardLogger):
            tb_log_dir = self.logger.log_dir
        else:
            tb_log_dir = self.cfg.result_file_path  # Fallback directory if logger is not set

        # Construct the save directory path
        save_path = f"{tb_log_dir}/images/epoch_{self.current_epoch}/images"

        original_image_dir = self.cfg.root_dir
        generated_image_dir = save_path
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
        )

def get_val_dataloader(cfg):
    karpathy_file = cfg.karpathy_file_path
    root_dir = cfg.root_dir
    start_index = 0
    end_index = 256
    val_dataset = CocoDataset(root_dir, karpathy_file, tokenizer=None, start_index=start_index, end_index=end_index)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.experiment.val_batch_size, collate_fn=val_dataset.collate_fn, num_workers=cfg.dataset.num_workers)
    return val_dataloader

if __name__ == "__main__":
    cfg, cfg_yaml = build_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(cfg.experiment.seed, workers=True)

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    os.makedirs(cfg.result_file_path, exist_ok=True)

    # Setup dataloader
    datamodule = CompressionDataModule(
        batch_size=cfg.experiment.local_batch_size,
        num_workers=cfg.dataset.num_workers,
        transform=transform,
        compression_level=-1)
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = get_val_dataloader(cfg)

    # Setup callback
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.result_file_path)
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        accelerator=device,
        num_nodes=cfg.dist.n_nodes,
        devices=cfg.dist.n_gpus,
        strategy="ddp",
        max_epochs=cfg.experiment.max_epochs,
        deterministic=False,
        logger=tb_logger,
        log_every_n_steps=cfg.experiment.log_every_n_steps,
        # val_check_interval=cfg.experiment.val_check_interval,
        check_val_every_n_epoch=cfg.experiment.check_val_every_n_epoch,
        enable_checkpointing=cfg.experiment.enable_checkpointing,
        num_sanity_val_steps=cfg.experiment.num_sanity_val_steps,
        precision=str(cfg.optimizer.precision),
        callbacks=[ModelSummary(max_depth=3), lr_logger],
        accumulate_grad_batches=cfg.experiment.grad_accumulation,
        # gradient_clip_val=cfg.optimizer.gradient_clip_val,
    )

    wrapper = SEEDTrainingWrapper(cfg).to(device)
    wrapper.setup("fit")

    cfg.experiment.total_training_steps = int(((len(train_dataloader) / cfg.dist.n_gpus) / cfg.experiment.grad_accumulation) * cfg.experiment.max_epochs)

    trainer.fit(
        wrapper, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
        ckpt_path="/home/zheedong/Projects/SEED/logs/seed_stage_1_long_caption_training_debug/lightning_logs/version_48_epoch_1/checkpoints/epoch=0-step=313.ckpt"
    )
    trainer.strategy.barrier()
