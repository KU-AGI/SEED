import os
from typing import Any, List
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from torch.nn import functional as F

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
from torchvision.transforms.functional import to_pil_image
from utils.config import build_config
import pyrootutils
from datamodules import build_datamodule
from datamodules.tokenizers import TokenizerUtils

from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
import torch.nn.functional as F
from pytorch_lightning.strategies import DDPStrategy
import open_clip
from info_nce import InfoNCE, info_nce
from models.seed_llama_tokenizer import ImageTokenizer
import transformers

from pytorch_lightning import loggers as pl_loggers

import numpy as np
from PIL import Image
import pdb


pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

IMG_FLAG = '<image>'
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 8192
image_id_shift = 32000

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class Stage1Training(LightningModule):
    def __init__(self, cfg, model, tokenizer):
        super().__init__()
        self.cfg = cfg
        # ImageTokenizer model
        self.image_tokenizer = model
        self.tokenizer = tokenizer

        # Frozen Text Encoder For Contrastive Learning
        self.model_clip, self.preprocess_train_clip, self.preprocess_val_clip = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
        for param in self.model_clip.parameters():
            param.requires_grad = False
        self.model_clip.eval()
        self.tokenizer_clip = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

        self.generation_config = {
            'num_beams': 5,
            'max_new_tokens': 32,
            'do_sample': False
        }

        self.s_token = "USER:"
        self.e_token = "ASSISTANT:"
        self.sep = "\n"

        # For test
        # is projection between query_output.last_hidden_state and gt_text_clip_embeddings is similar?
        self.text_down = nn.Linear(768, 32, bias=False).to(self.device)
        self.distill_text_proj = nn.Linear(32 * 32, 1024).to(self.device)

    def training_step(self, batch, batch_idx: int):

        # print(batch)
        B = batch.img.shape[0]
        # CLIP embedding for computing similarity between embeedings
        # Final query_output contrastive learning to text embedding.
        # pdb.set_trace()
        # gt_img_clip_embeddings = []
        gt_text_clip_embeddings = []
        with torch.no_grad():
            for idx in range(B):
                gt_text_clip_embeddings.append(self.tokenizer_clip(batch.gt_txt[idx]).squeeze().to(self.device))
            gt_text_clip_embeddings = torch.stack(gt_text_clip_embeddings, dim=0)

            # gt_img_clip_embeddings = self.model_clip.encode_image(batch.img.to(self.device))
            gt_text_clip_embeddings = self.model_clip.encode_text(gt_text_clip_embeddings.to(self.device))
        
        #pdb.set_trace()
        # For test cosine similarity
        # print(F.cosine_similarity(gt_text_clip_embeddings, gt_img_clip_embeddings))

        #--------------------------
        
        # pdb.set_trace()
        # Image into fp16
        img = batch.img.to(self.device)
        if self.image_tokenizer.fp16:
            img = img.half()

        # This is full process to get [b, 32] tokens (embed_ind is tokens)
        # embed_ind, query_output_up = self.image_tokenizer.model.get_codebook_indices(img)


        #--------------------------
        # pdb.set_trace()
        # Now step by step process to compute loss
        with self.image_tokenizer.model.maybe_autocast():
            # [b, 257, 1408]
            image_embeds = self.image_tokenizer.model.ln_vision(self.image_tokenizer.model.visual_encoder(img))
            # [b, 256]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(img.device)
                # [b, 32, 768]
        # Original query_tokens shape is [1, 32, 768]
        # Match to batch size
        query_tokens = self.image_tokenizer.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        # query_output : [b, 32, 768]
        query_output = self.image_tokenizer.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # TODO: query_output should be trained to be similar with text embedding
        # Image embedding is cross attentioned.

        # Notice: query_output_down is match to clip embedding?
        # CLIP ViT-H/14 text embeeding is [b, 1024]. Then it matches to [b, 32, 32]?
        # [b, 32, 32]
        query_output_down = self.image_tokenizer.model.encode_task_layer(query_output.last_hidden_state)

        # quant [b, 32, 32], loss_embed [b, 32, 768], embed_ind [b, 32]
        quant, loss_embed, embed_ind = self.image_tokenizer.model.quantize(query_output_down)

        embed_ind = embed_ind.reshape(quant.shape[0], -1)
        # quant embedding dimension is [b, 32, 32]
        # decoder_task_layer upscale it to [b, 32, 768]
        # [b, 32, 32] => [b, 32, 768]
        query_output_up = self.image_tokenizer.model.decode_task_layer(quant)

        '''
        text_query_output_down_flatten_cos_sim = F.cosine_similarity(query_output_down_flatten, gt_text_clip_embeddings)
        img_query_output_down_flatten_cos_sim = F.cosine_similarity(query_output_down_flatten, gt_img_clip_embeddings)
        print(f"Cosine similarity between gt text and query_output_down_flatten: {text_query_output_down_flatten_cos_sim}")            
        print(f"Cosine similarity between gt img and query_output_down_flatten: {img_query_output_down_flatten_cos_sim}")
        '''

        #------------------------
        # TODO: Contrastive loss?
        text_embedding = self.text_down(query_output.last_hidden_state.float())
        text_embedding = text_embedding.reshape(text_embedding.shape[0], -1)
        text_embedding_proj = self.distill_text_proj(text_embedding)

        info_nce_loss = info_nce(text_embedding_proj, gt_text_clip_embeddings)
        self.log('info_nce_loss', info_nce_loss, batch_size=B, \
            on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "lr",
            self.lr_schedulers().get_last_lr()[0],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        return info_nce_loss

        F.cosine_similarity(text_embedding_proj, gt_text_clip_embeddings)

        '''
        #------------------------
        # Stage 2 Training
        #------------------------

        pos_embed_image = self.image_tokenizer.model.pos_embed_image.repeat(B, 1, 1)
        query_output_up_pos_image = query_output.last_hidden_state + pos_embed_image

        # Transformers block
        # multilayer transformer decoder
        for blk in self.image_tokenizer.model.blocks_image:
            query_output_up_pos_image = blk(query_output_up_pos_image)
        pdb.set_trace()

        # MLP
        # Still [b, 32, 768]
        query_output_up = query_output_up_pos_image
        # 2 layer mlp to 768 -> 32, [b, 32, 32]
        reverse_output = self.image_tokenizer.model.image_down(query_output_up)
        # [b, 32, 32] => [b, 32 * 32]
        reverse_output = reverse_output.reshape(B, -1)
        # [b, 1024] => [b, 1024]
        reverse_output_proj = self.image_tokenizer.model.distill_image_proj(reverse_output)
        
        text_reverse_output_proj_cos_sim = F.cosine_similarity(reverse_output_proj, gt_text_clip_embeddings)
        print(text_reverse_output_proj_cos_sim)
        img_reverse_output_proj_cos_sim = F.cosine_similarity(reverse_output_proj, gt_img_clip_embeddings)
        print(img_reverse_output_proj_cos_sim)

        pdb.set_trace()
        '''
            
        return
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=10)

        lr_scheduler_config = {
            "scheduler": scheduler,
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


if __name__ == "__main__":
    cfg, cfg_yaml = build_config()
    device = "cuda"

    cfg.dataset.name = 'cc3m_coco'
    cfg.dataset.type = 'webdataset'

    cfg.tokenizer_cfg_path = 'configs/tokenizer/seed_llama_tokenizer.yaml'
    cfg.dataset.gt_text = True
    seed_everything(cfg.experiment.seed, workers=True)

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    tokenizer_cfg = OmegaConf.load(cfg.tokenizer_cfg_path)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg, device=device, load_diffusion=True)
    model = tokenizer._image_tokenizer.train()

    cfg.experiment.local_batch_size = 4096
    cfg.dist.n_gpus = 2

    datamodule = build_datamodule(
        cfg=cfg,
        train_transform=transform,
        val_transform=transform,
        pin_memory=False,
        epoch=100,
        total_gpus=cfg.dist.n_gpus,
    )

    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="./stage1_training/")

    trainer = pl.Trainer(
        accelerator="gpu",
        num_nodes=cfg.dist.n_nodes,
        devices=cfg.dist.n_gpus,
        strategy=DDPStrategy(
            find_unused_parameters=True,
            ddp_comm_hook=default_hooks.fp16_compress_hook
            if cfg.optimizer.fp16_grad_comp
            else None,
        ),
        max_epochs=10,
        enable_checkpointing=False,
        limit_test_batches=1.0,
        deterministic=True,
        logger=tb_logger,
        log_every_n_steps=10
    )

    # Setup training parameters
    model.model.train()
    for param in model.parameters():
        param.required_grad = True

    # Freeze ViT Encoder
    for param in model.model.visual_encoder.parameters():
        param.required_grad = False

    wrapper = Stage1Training(cfg, model, tokenizer)

    trainer.fit(wrapper, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.strategy.barrier()
