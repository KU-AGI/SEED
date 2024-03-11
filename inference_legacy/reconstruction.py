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
from functools import partial
from models.seed_qformer.vit import Block

import numpy as np
from PIL import Image
import pdb


pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

BOI_TOKEN = "<img>"
EOI_TOKEN = "</img>"
IMG_TOKEN = "<img_{:05d}>"

IMG_FLAG = "<image>"
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 8192
image_id_shift = 32000

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class SEEDTrainingWrapper(LightningModule):
    """Training wrapper for SEED

    Args:
        LightningModule (cfg, model): model should be ImageTokenizer
    """    
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        # ImageTokenizer model
        # Target model to train
        self.image_tokenizer = model

        # Frozen Text Encoder For Contrastive Learning
        (
            self.model_clip,
            self.preprocess_train_clip,
            self.preprocess_val_clip,
        ) = open_clip.create_model_and_transforms(
            "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
        for param in self.model_clip.parameters():
            param.requires_grad = False
        self.model_clip.eval()
        self.tokenizer_clip = open_clip.get_tokenizer(
            "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
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

        # For logging
        self.sample_embed_ind = None

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
                    self.tokenizer_clip(batch_text[idx]).squeeze().to(self.device)
                )
            gt_text_clip_embeddings = torch.stack(gt_text_clip_embeddings, dim=0)

            # gt_img_clip_embeddings = self.model_clip.encode_image(batch.img.to(self.device))
            gt_text_clip_embeddings = self.model_clip.encode_text(
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
        return self.model_clip.encode_image(batch_img.to(self.device))

    def get_causal_embeddings(self, image):
        return self.image_tokenizer.model.get_causal_embeddings(image)

    def test_step(self, batch, batch_idx: int):
        """_summary_

        Args:
            batch (_type_): _description_
            batch_idx (int): _description_
        """        
        #------------------------
        # Stage 2 Training
        #------------------------

        #------------------------
        # Stage 2 - 1 : Codebook Training
        #------------------------

        # Causal embedding is trained in stage 1.
        causal_embeddings = self.get_causal_embeddings(batch.img)

        # [b, 32, 32]
        query_output_down = self.image_tokenizer.model.encode_task_layer(causal_embeddings)

        # quant [b, 32, 32], loss_embed [b, 32, 768], embed_ind [b, 32]
        quant, loss_embed, embed_ind = self.image_tokenizer.model.quantize(query_output_down)

        for blk in self.embedding_block:
            quant = blk(quant)
        quant = quant.view(quant.shape[0], -1)
        generation_embedding = self.embedding_proj(quant)

        gt_img_clip_embeddings = self.get_clip_img_embedding(batch.img)

        print(F.cosine_similarity(generation_embedding, gt_img_clip_embeddings, dim=1))
    
        pdb.set_trace()

        image = self.image_tokenizer.diffusion_model(
            image_embeds=generation_embedding.type(torch.float16),
            negative_image_embeds=None,
            guidance_scale=10,
            noise_level=0,
            num_inference_steps=20,
            latents=self.image_tokenizer.latents,
        ).images

        pdb.set_trace()
        
        return


if __name__ == "__main__":
    cfg, cfg_yaml = build_config(cfg_path="configs/seed_llama_tokenizer.yaml")
    device = "cuda"

    # cfg.tokenizer_cfg_path = "configs/tokenizer/seed_llama_tokenizer.yaml"
    seed_everything(cfg.experiment.seed, workers=True)

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    # Empty Image tokenizer model
    # BLIP2QformerQuantizer is loaded inside ImageTokenizer
    image_tokenzier = ImageTokenizer(model_path='pretrained/seed_tokenizer/seed_quantizer.pt',
                           fp16=False,
                           from_pretrained=True,
                           load_diffusion=True,
                           diffusion_model_path='stabilityai/stable-diffusion-2-1-unclip'
                           )

    # Debugging
    cfg.experiment.local_batch_size = 2
    cfg.dist.n_gpus = 1

    datamodule = build_datamodule(
        cfg=cfg,
        train_transform=transform,
        val_transform=transform,
        pin_memory=False,
        epoch=cfg.experiment.max_epochs,
        total_gpus=cfg.dist.n_gpus,
    )

    datamodule.setup()
    val_dataloader = datamodule.val_dataloader()
    val_dataloader.dataset.set_custom_length(30000)

    trainer = pl.Trainer(
        accelerator="gpu",
        num_nodes=cfg.dist.n_nodes,
        devices=cfg.dist.n_gpus,
        strategy=DDPStrategy(
            find_unused_parameters=False,
        ),
        max_epochs=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        deterministic=True,
    )

    # Setup training parameters
    image_tokenzier.model.train()
    for param in image_tokenzier.parameters():
        param.requires_grad = True

    # Freeze ViT Encoder
    for param in image_tokenzier.model.visual_encoder.parameters():
        param.requires_grad = False

    wrapper = SEEDTrainingWrapper.load_from_checkpoint("my_generation_embedding/lightning_logs/version_0/checkpoints/epoch=0-step=667.ckpt", cfg=cfg, model=image_tokenzier)
    wrapper.eval()

    trainer.test(
        wrapper, dataloaders=val_dataloader
    )
    trainer.strategy.barrier()
