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

        # For test in stage 1 hypothesis
        # is projection between query_output.last_hidden_state and gt_text_clip_embeddings is similar?
        self.text_down = nn.Linear(768, 32, bias=False).to(self.device)
        self.distill_text_proj = nn.Linear(32 * 32, 1024).to(self.device)

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

    def get_stage_1_loss(self, batch, batch_idx: int):
        """Final query_output contrastive learning to text embedding.
        Args:
            batch (Item): Image and text batch. Image is [b, 3, 224, 224], text is [b, 32]
            batch_idx (int): batch index

        Returns:
            float: loss for contrastive learning
        """        
        # print(batch)

        # CLIP embedding for computing similarity between embeedings
        gt_text_clip_embeddings = self.get_clip_text_embedding(batch.gt_txt)

        # pdb.set_trace()
        # For test cosine similarity
        # print(F.cosine_similarity(gt_text_clip_embeddings, gt_img_clip_embeddings))

        # --------------------------
        # causal_embeddings is [b, 32, 768]
        causal_embeddings = self.get_causal_embeddings(batch.img.half().to(self.device))

        # ------------------------
        # TODO: Contrastive loss?
        # CLIP ViT-H/14 text embeeding is [b, 1024]. Then it matches to [b, 32, 32]?
        text_embedding = self.text_down(causal_embeddings.float())
        text_embedding = text_embedding.reshape(text_embedding.shape[0], -1)
        text_embedding_proj = self.distill_text_proj(text_embedding)

        F.cosine_similarity(text_embedding_proj, gt_text_clip_embeddings)

        info_nce_loss = info_nce(text_embedding_proj, gt_text_clip_embeddings)

        self.log(
            "info_nce_loss",
            info_nce_loss,
            batch_size=self.B,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "lr",
            self.lr_schedulers().get_last_lr()[0],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        return info_nce_loss

    def get_stage_2_loss(self, batch, batch_idx: int):
        """_summary_

        Args:
            batch (_type_): _description_
            batch_idx (int): _description_
        """        
        #------------------------
        # Stage 2 Training
        #------------------------
        pdb.set_trace()

        #------------------------
        # Stage 2 - 1 : Codebook Training
        #------------------------

        # Causal embedding is trained in stage 1.
        causal_embeddings = self.get_causal_embeddings(batch.img)

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

        pdb.set_trace()
        # quant embedding dimension is [b, 32, 32]
        # decoder_task_layer upscale it to [b, 32, 768]
        # [b, 32, 32] => [b, 32, 768]
        query_output_up = self.image_tokenizer.model.decode_task_layer(quant)

        print(F.cosine_similarity(query_output_up, causal_embeddings))

        # Transformer decoder
        query_output_up = self.image_tokenizer.model.get_transformer_decoded_embedding(query_output_up)

        # query_output_up_pos_image should be similar to original causal_embeddings
        # Maximize cosine similarity between query_output_up_pos_image and causal_embeddings
        loss_reconstruction_causal_embedding = F.cosine_similarity(query_output_up, causal_embeddings).mean()

        print(F.cosine_similarity(query_output_up, causal_embeddings))

        #------------------------
        # Stage 2 - 3 : Reconstruction Generation Embedding
        #------------------------

        pdb.set_trace()
        # MLP
        reverse_output_proj = self.image_tokenizer.model.get_mlp_decoded_embedding(query_output_up)

        # reverse_output_proj should be similar to gt_img_clip_embeddings
        # MSE Loss between reverse_output_proj and gt_img_clip_embeddings
        gt_img_clip_embeddings = self.get_clip_img_embedding(batch.img.to(self.device))
        loss_reconstruction_generation_embedding = F.mse_loss(reverse_output_proj, gt_img_clip_embeddings)

        #------------------------
        # Debugging
        # Text Clip Embedding
        gt_text_clip_embeddings = self.get_clip_text_embedding(batch.gt_txt)

        # Compare between gt text clip embedding and query_output_up
        print(F.cosine_similarity(gt_text_clip_embeddings, gt_img_clip_embeddings))

        # Compare between gt text clip embedding and reverse_output_proj
        print(F.cosine_similarity(gt_text_clip_embeddings, reverse_output_proj))

        # Compare between gt image clip embedding and reverse_output_proj
        print(F.cosine_similarity(gt_img_clip_embeddings, reverse_output_proj))
        #------------------------

        #------------------------
        # Debugging
        # Is possible to reconstruct image from causal embedding?
        debug_reverse_output_proj = self.image_tokenizer.model.get_mlp_decoded_embedding(causal_embeddings)

        # reverse_output_proj should be similar to gt_img_clip_embeddings
        # MSE Loss between reverse_output_proj and gt_img_clip_embeddings
        print(F.cosine_similarity(debug_reverse_output_proj, reverse_output_proj))
        print(F.cosine_similarity(debug_reverse_output_proj, gt_img_clip_embeddings))
        print(F.cosine_similarity(debug_reverse_output_proj, gt_text_clip_embeddings))

        # If after transformer decoder, it is possible to reconstruct image from causal embedding?
        debug_query_output_up = self.image_tokenizer.model.get_transformer_decoded_embedding(causal_embeddings)
        debug_reverse_output_proj = self.image_tokenizer.model.get_mlp_decoded_embedding(debug_query_output_up)

        print(F.cosine_similarity(debug_reverse_output_proj, reverse_output_proj))
        print(F.cosine_similarity(debug_reverse_output_proj, gt_img_clip_embeddings))
        print(F.cosine_similarity(debug_reverse_output_proj, gt_text_clip_embeddings))

        #------------------------

        pdb.set_trace()

        return loss_embed - loss_reconstruction_causal_embedding + loss_reconstruction_generation_embedding

    def training_step(self, batch, batch_idx: int):
        self.B = batch.img.shape[0]

        #stage_1_loss = self.get_stage_1_loss(batch, batch_idx)
        stage_2_loss = self.get_stage_2_loss(batch, batch_idx)
        return stage_2_loss
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=100
        )

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

    cfg.dataset.name = "cc3m_coco"
    cfg.dataset.type = "webdataset"

    # cfg.tokenizer_cfg_path = "configs/tokenizer/seed_llama_tokenizer.yaml"
    cfg.dataset.gt_text = True
    seed_everything(cfg.experiment.seed, workers=True)

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    # Empty Image tokenizer model
    # BLIP2QformerQuantizer is loaded inside ImageTokenizer
    image_tokenzier = ImageTokenizer(model_path='pretrained/seed_tokenizer/seed_quantizer.pt',
                           fp16=False,
                           is_train=False)

    # Debugging
    cfg.experiment.local_batch_size = 1
    cfg.dist.n_gpus = 1

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

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="./stage2_debugging/")

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
        log_every_n_steps=10,
    )

    # Setup training parameters
    image_tokenzier.model.train()
    for param in image_tokenzier.parameters():
        param.required_grad = True

    # Freeze ViT Encoder
    for param in image_tokenzier.model.visual_encoder.parameters():
        param.required_grad = False

    wrapper = SEEDTrainingWrapper(cfg, image_tokenzier)

    trainer.fit(
        wrapper, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    trainer.strategy.barrier()
