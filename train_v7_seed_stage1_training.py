import os
from typing import Any, List
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

from tqdm import tqdm
import time
import json

import argparse

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
from info_nce import InfoNCE, info_nce
from models.seed_llama_tokenizer import ImageTokenizer
import transformers

from pytorch_lightning import loggers as pl_loggers
from functools import partial
from models.seed_qformer.vit import Block
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.utilities import grad_norm

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from einops import rearrange
import pdb


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
                            is_train_stage_1=True,
                            )

        self.B = None
        
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
                #nn.init.xavier_uniform_(param) 
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
        
        # For test training
        # self.image_tokenizer.model.distill_image_proj = nn.Linear(32 * 32, 1024).to(self.device)

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

    def get_stage2_quant(self, img):
        # Causal embedding is trained in stage 1.
        causal_embeddings = self.get_causal_embeddings(img)

        # [b, 32, 32]
        query_output_down = self.image_tokenizer.model.encode_task_layer(causal_embeddings)

        # For debug
        quant = query_output_down
        loss_embed = None
        embed_ind = None

        # Simple flatten and linear projection
        # Debug : Try to use transformer
        # for blk in self.embedding_block:
        #     quant = blk(quant)
        quant = quant.view(quant.shape[0], -1)
        quant = self.embedding_proj(quant)

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

    def get_stage_1_loss(self, batch, batch_idx: int, is_validation=False):
        """_summary_

        Args:
            batch (_type_): _description_
            batch_idx (int): _description_
            is_validation (bool, optional): _description_. Defaults to False.
        """
        # image = self.transform_224(batch.img)
        device = self.device
        image = batch.img.to(device)
        text = [text[0].encode("ascii", "ignore").decode() for text in batch.gt_txt]
        with torch.no_grad():
            image_embeds = self.image_tokenizer.model.ln_vision(
                self.image_tokenizer.model.visual_encoder(image)
            )
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.image_tokenizer.model.query_tokens.expand(image_embeds.shape[0], -1, -1)

        # Bidirectional cross attention
        '''
        query_output = self.image_tokenizer.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )
        '''

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
        image_feats = F.normalize(query_output.last_hidden_state[:, :, :], dim=1)

        text_tokens = self.image_tokenizer.model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        text_output = self.image_tokenizer.model.Qformer.bert(
            text_tokens.input_ids.to(device),
            attention_mask=text_tokens.attention_mask.to(device),
            return_dict=True,
        )

        # CLS token
        # [b, 768]
        text_feats = F.normalize(text_output.last_hidden_state[:, 0, :], dim=1)

        ###============== Image-text Contrastive ===================###
        # image_feats : [b, 32, 768] => [32, b, 768]
        # text_feat : [b, 768] => [768, b]
        # sim_i2t : [32, b, 768] * [768, b] => [32, b, b]
        image_feats = rearrange(image_feats, "b n d -> n b d")
        text_feats = rearrange(text_feats, "b d -> d b")

        sim_i2t = torch.matmul(
            image_feats, text_feats
        ).squeeze()

        bs = image.size(0)
        # targets : [32, b]
        targets = torch.arange(bs, dtype=torch.long).repeat(32, 1).to(device)

        loss_itc = F.cross_entropy(sim_i2t, targets)

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

        ''' bypass
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
        '''        

        query_output_up = causal_embeddings
        # Transformer decoder
        query_output_up = self.image_tokenizer.model.get_transformer_decoded_embedding(query_output_up)
        
        

        # query_output_up_pos_image should be similar to original causal_embeddings
        # Maximize cosine similarity between query_output_up_pos_image and causal_embeddings

        loss_recon = F.cosine_similarity(query_output_up, causal_embeddings).mean()
        
        
        
        #------------------------
        # Stage 2 - 3 : Reconstruction Generation Embedding
        #------------------------

        # MLP
        # query_output_up = causal_embeddings
        reverse_output_proj = self.image_tokenizer.model.get_mlp_decoded_embedding(query_output_up)

        gt_img_clip_embeddings = self.get_clip_img_embedding(img)
    
        loss_generation_embed = F.mse_loss(reverse_output_proj, gt_img_clip_embeddings)

        #loss_total = loss_embed - loss_recon + loss_generation_embed
        loss_total = loss_generation_embed
        loss_total = loss_total.mean()

        # loss_dict = {"loss_embed": loss_embed, "loss_recon": loss_recon,
        #         "loss_generation_embed": loss_generation_embed,
        #         "loss": loss_total}
        
        loss_dict = {"loss_generation_embed": loss_generation_embed,
                     "loss": loss_total}

        #------------------------
        # Logging
        #------------------------
        #generation_embedding_cosine_similarity = F.cosine_similarity(reverse_output_proj, gt_img_clip_embeddings).mean()

        #self.logging_train(generation_embedding_cosine_similarity, loss_dict)
        self.logging_train(None, loss_dict)

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

        self.log(
            "train/codebook_loss_embed",
            loss_dict["loss_embed"].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

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
        self.B = batch.img.shape[0]
        # gt_text is a list of string
        # Encoding text in list to ascii
        #batch.gt_txt = [[text[0].encode("ascii", "ignore").decode()] for text in batch.gt_txt]

        if self.current_epoch <= 5:
            stage_1_loss = self.get_stage_1_loss(batch, batch_idx)
            return stage_1_loss
        else:
            # Freeze stage 1 model
            for param in self.image_tokenizer.model.Qformer.parameters():
                param.requires_grad = False
            stage_2_loss = self.get_stage_2_loss(batch, batch_idx)
            return stage_2_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int):
        image = self.transform_224(batch.img)
        image_embeds, _, _ = self.get_stage2_quant(image)

        with torch.no_grad():
            reconstructed_images = self.image_tokenizer.diffusion_model(
                image_embeds=image_embeds,
                negative_image_embeds=None,
                guidance_scale=10,
                noise_level=0,
                num_inference_steps=100,
                latents=self.image_tokenizer.latents,
            ).images

        tensor_images = []
        for img in reconstructed_images:
            tensor_images.append(self.pil_to_tensor(img).unsqueeze(0))
        tensor_images = torch.cat(tensor_images, dim=0)

        # Check if image is already logged
        if batch_idx not in self.logged_original_image:
            self.logger.experiment.add_images(
                f"original/image_batch_{batch_idx}",
                batch.img,
            )

            self.logger.experiment.add_images(
                f"original/image_batch_{batch_idx}_seed_reconstructed",
                tensor_images,
            )

            # logging original caption
            self.logger.experiment.add_text(
                f"original/gt_text_image_batch_{batch_idx}",
                batch.gt_txt[0][0],
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-8)
        total_trainig_steps = 782 * self.cfg.experiment.max_epochs
        scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=total_trainig_steps * 0.03,
            num_training_steps=2 * total_trainig_steps,
            num_cycles=2)

        lr_scheduler_config = {
            "scheduler": scheduler,
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config,
                "gradient_clip_val": self.cfg.optimizer.grad_clip_val,}
        
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

if __name__ == "__main__":
    cfg, cfg_yaml = build_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # cfg.tokenizer_cfg_path = "configs/tokenizer/seed_llama_tokenizer.yaml"
    seed_everything(cfg.experiment.seed, workers=True)

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    os.makedirs(cfg.result_file_path, exist_ok=True)

    datamodule = build_datamodule(
        cfg=cfg,
        train_transform=transform,
        val_transform=transform,
        pin_memory=False,
        epoch=cfg.experiment.max_epochs,
        total_gpus=cfg.dist.n_gpus,
    )

    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.result_file_path)
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        accelerator=device,
        num_nodes=cfg.dist.n_nodes,
        devices=cfg.dist.n_gpus,
        # strategy=DDPStrategy(
        #     find_unused_parameters=True,
        #     ddp_comm_hook=default_hooks.fp16_compress_hook
        #     if cfg.optimizer.fp16_grad_comp
        #     else None,
        # ),
        strategy="ddp",
        max_epochs=cfg.experiment.max_epochs,
        deterministic=False,
        logger=tb_logger,
        log_every_n_steps=1,
        # val_check_interval=cfg.experiment.val_check_interval,
        check_val_every_n_epoch=cfg.experiment.check_val_every_n_epoch,
        # enable_checkpointing=cfg.experiment.enable_checkpointing,
        enable_checkpointing=True,
        # Debug
        num_sanity_val_steps=2,
        precision='bf16',
        # overfit_batches=cfg.experiment.overfit_batches,
        callbacks=[ModelSummary(max_depth=3), lr_logger],
        accumulate_grad_batches=cfg.experiment.grad_accumulation
    )

    wrapper = SEEDTrainingWrapper(cfg).to(device)
    wrapper.setup("fit")

    trainer.fit(
        wrapper, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
    )
    trainer.strategy.barrier()
