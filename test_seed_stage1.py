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
from torchvision.utils import save_image

from typing import Any, Callable, Dict, List, Optional, Union

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

def drop_randomly(x, drop_prob):
    """_summary_
        Get Tensor x, and randomly drop some of them
        according to drop_prob
    Args:
        x (torch.Tensor): [b, 32] shape Tensor
        drop_prob (float): Probability of dropping
    
    Returns:
        torch.Tensor: [b, 32] shape Tensor
    """    
    keep = torch.rand(x.shape[0], x.shape[1], device=x.device) > drop_prob
    return x * keep

def change_randomly(x, change_prob):
    """_summary_
        Get Tensor x, and randomly change some of them
        image codes are 0 to 8191, so randomly change
        according to change_prob
    Args:
        x (torch.Tensor): [b, 32] shape Tensor
        change_prob (float): Probability of changing 

    Returns:
        torch.Tensor: [b, 32] shape Tensor
    """    

    # Create a mask with the same shape as x, with True values where changes should happen
    change_mask = torch.rand_like(x, dtype=torch.float32) < change_prob

    # Generate random values for each element in the tensor, in the range [0, 8191]
    random_values = torch.randint(0, 8192, x.shape, device=x.device)

    # Use the mask to replace elements in x with the random values
    x = torch.where(change_mask, random_values, x)

    return x

class SEEDTestWrapper(LightningModule):
    """Test wrapper for SEED

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

    def setup(self, stage):
        # Setup training parameter
        self.image_tokenizer.model.train()
        for param in self.image_tokenizer.model.parameters():
            param.requires_grad = False

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

        # For fp16
        if self.cfg.optimizer.fp16:
            self.image_tokenizer = self.image_tokenizer.half()
            self.image_encoder = self.image_encoder.half()
            self.embedding_proj = self.embedding_proj.half()
            for blk in self.embedding_block:
                blk = blk.half()

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

    def get_stage_2_loss(self, batch, batch_idx: int, is_validation=False):
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
        quant, loss_embed, embed_ind = self.get_stage2_quant(batch)

        gt_img_clip_embeddings = self.get_clip_img_embedding(batch.img)
        
        loss = F.mse_loss(quant, gt_img_clip_embeddings)

        #------------------------
        # Logging

        if not is_validation:
            self.logging_train(quant, loss_embed, gt_img_clip_embeddings, loss)
        else:
            self.logging_val(quant, loss_embed, gt_img_clip_embeddings, loss)
            self.sample_embed_ind = embed_ind.reshape(self.B, -1)

        return loss + loss_embed.mean()
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx: int):
        """_summary_

        Args:
            batch (_type_): _description_
            batch_idx (int): _description_
        """
        mode = "change"
        save_root = "causal_embedding_test_images"
        image_save_root = f"{save_root}/{mode}_test_1"
        os.makedirs(image_save_root, exist_ok=True)

        image = batch.img.to(self.device)

        # Save Original Image
        save_image(image, f"{image_save_root}/original_image.png")

        # Get Image Tokens, Save Reconstructed Image
        indicies = self.image_tokenizer.encode(
            image_torch=image
        )

        reconstructed_image = self.image_tokenizer.decode(
            indicies
        )
        reconstructed_image[0].save(
            f"{image_save_root}/reconstructed_image.png"
        )

        json_path = f"{image_save_root}/image_tokens.json"
        indicies_dict = {
            "original_image_tokens": indicies.tolist()
        }

        # Accoding to drop_prob, drop some of tokens
        for prob in [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
            save_path = f"{mode}_{prob}"
            os.makedirs(f"{image_save_root}/{save_path}", exist_ok=True)

            # Save 10 images for each drop_prob
            indicies_dict[f"{mode}_{prob}"] = []
            for index in range(10):
                if mode == "drop":
                    new_indicies = drop_randomly(indicies, prob)
                elif mode == "change":
                    new_indicies = change_randomly(indicies, prob)
                else:
                    raise NotImplementedError(f"{mode} is not implemented")
                    exit()

                image = self.image_tokenizer.decode(
                    new_indicies
                )
                image[0].save(
                    f"{image_save_root}/{save_path}/image_{index}.png"
                )
                indicies_dict[f"{mode}_{prob}"].append(new_indicies.tolist())
            
        
        with open(json_path, "w") as f:
            json.dump(indicies_dict, f)
        exit()
        return

if __name__ == "__main__":
    cfg, cfg_yaml = build_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # cfg.tokenizer_cfg_path = "configs/tokenizer/seed_llama_tokenizer.yaml"
    seed_everything(cfg.experiment.seed, workers=True)

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    '''
    transform = transforms.Compose(
        [
            transforms.Resize(224, antialias=True),
            transforms.ToTensor(),
        ]
    )
    '''

    # Set up datamodule
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

    trainer = pl.Trainer(
        accelerator=device,
        num_nodes=cfg.dist.n_nodes,
        devices=cfg.dist.n_gpus,
        strategy="ddp",
        max_epochs=cfg.experiment.max_epochs,
        enable_checkpointing=cfg.experiment.enable_checkpointing,
        enable_model_summary=False,
        deterministic=True,
    )

    wrapper = SEEDTestWrapper(cfg).to(device)
    wrapper.setup("fit")

    trainer.test(
        wrapper, dataloaders=val_dataloader,
    )
    trainer.strategy.barrier()
