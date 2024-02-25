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

from pytorch_lightning import loggers as pl_loggers
from functools import partial
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.utilities import grad_norm

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt

from models.seed_qformer.vit import Block
from models.seed_llama_tokenizer import ImageTokenizer

from coco_dataloader import CocoDataset

from datamodules.seed_llama_datamodule import SEEDDataModule

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

        # For logging
        self.pil_to_tensor = transforms.ToTensor()
        self.sample_image_ind = 0
        self.logged_original_image = set()
        
        self.stage = cfg.experiment.stage
        self.temp = nn.Parameter(0.07 * torch.ones([]))

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

        # [b, 32, 32] => [b, 32, 768]
        query_output_up = self.image_tokenizer.model.decode_task_layer(quant)

        # [b, 32, 768] => [b, 32, 768]
        query_output_up = self.image_tokenizer.model.get_transformer_decoded_embedding(query_output_up)

        # [b, 32, 768] => [b, 32, 32] => [b, 1024]
        reverse_output_proj = self.image_tokenizer.model.get_mlp_decoded_embedding(query_output_up)

        return reverse_output_proj

    @torch.no_grad()
    def test_step(self, batch, batch_idx: int):
        if self.logger is not None and isinstance(self.logger, pl.loggers.TensorBoardLogger):
            tb_log_dir = self.logger.log_dir
        else:
            tb_log_dir = self.cfg.result_file_path  # Fallback directory if logger is not set

        _, _, image_name = batch
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

        save_path = f"{tb_log_dir}/images/version_{self.logger.version}/COCOVal30000"
        os.makedirs(save_path, exist_ok=True)

        for img, cur_name in zip(reconstructed_images, image_name):
            # save PIL image to save_path
            img.save(f"{save_path}/{cur_name}")
        
        return

        
if __name__ == "__main__":
    cfg, cfg_yaml = build_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    seed_everything(cfg.experiment.seed, workers=True)

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    os.makedirs(cfg.result_file_path, exist_ok=True)

    from datamodules.datasets.coco_val import COCOValDataSet
    from torch.utils.data import DataLoader

    mode = 5000  # 30000 or 5000

    print(f"Load {mode} images from COCO Validation.")
    if mode == 30000:
        test_dataset = COCOValDataSet(transform=transform)
    else:
        test_dataset = CocoDataset(
            root_dir=cfg.dataset.val_config.root_dir,
            karpathy_file=cfg.dataset.val_config.karpathy_file_path,
            tokenizer=None,
            start_index=0,
            end_index=5000,
        )
    
    print(f"Test dataset length: {len(test_dataset)}")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.experiment.val_batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.result_file_path)

    trainer = pl.Trainer(
        accelerator=device,
        num_nodes=cfg.dist.n_nodes,
        devices=cfg.dist.n_gpus,
        strategy="ddp",
        max_epochs=1,
        deterministic=True,
        precision=str(cfg.optimizer.precision),
        callbacks=[ModelSummary(max_depth=3)],
        logger=tb_logger,
    )

    wrapper = SEEDTrainingWrapper(cfg).to(device)
    # wrapper = SEEDTrainingWrapper.load_from_checkpoint(cfg.weight_path, cfg=cfg, strict=False).to(device)

    trainer.test(wrapper, dataloaders=test_dataloader)

    

