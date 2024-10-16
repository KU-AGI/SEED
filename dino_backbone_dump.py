import os
import torch
import torch.nn as nn

import hydra
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, seed_everything
from omegaconf import OmegaConf
import pyrootutils

from pytorch_lightning.strategies import DDPStrategy, DDPFullyShardedStrategy
from einops import rearrange

import numpy as np

from datamodules.seed_llama_datamodule import SEEDDataModule

from utils.config import build_config
from tqdm import tqdm

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

BOI_TOKEN = "<img>"
EOI_TOKEN = "</img>"
IMG_TOKEN = "<img_{:05d}>"

IMG_FLAG = "<image>"
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 8192
IMAGE_ID_SHIFT = 32000

class DINOBackbone(nn.Module):
    def __init__(self, dinov2):
        super().__init__()
        self.dinov2 = dinov2

    def forward(self, x):
        enc_out = self.dinov2.forward_features(x)
        return rearrange(
            enc_out["x_norm_patchtokens"], 
            "b (h w) c -> b c h w",
            h=int(np.sqrt(enc_out["x_norm_patchtokens"].shape[-2]))
        )

class DINOBackboneWrapper(LightningModule):
    """Training wrapper for Slot

    Args:
        LightningModule (cfg, model): model should be ImageTokenizer
    """    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.stage = cfg.experiment.stage

        # Backbone is from ImageTokenizer
        dinov2 = torch.hub.load("facebookresearch/dinov2", cfg.stage1.dino_model_name)
        self.backbone = DINOBackbone(dinov2).eval()

        self.save_path = "/ssd0/data/slot_train_dino_dump"

    def setup(self, stage):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def test_step(self, batch, batch_idx):
        if len(batch) == 2:
            images, captions = batch
        else:
            images, captions, _ = batch

        image_embeds = self.backbone(images)

        for idx, (image_embed, caption) in enumerate(zip(image_embeds, tqdm(captions))):
            feature_path = f"{self.save_path}/{torch.cuda.current_device()}_{self.global_step}_{idx}.pt"
            torch.save(image_embed, feature_path)
            with open(f"{self.save_path}/{torch.cuda.current_device()}_{self.global_step}_{idx}.txt", "w") as f:
                f.write(caption)

if __name__ == "__main__":
    cfg, cfg_yaml = build_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    seed_everything(cfg.experiment.seed, workers=True)

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    os.makedirs(cfg.result_file_path, exist_ok=True)

    datamodule = SEEDDataModule(cfg, transform=transform, use_coco_val=cfg.dataset.val_config.use_coco_val)
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()

    cfg.experiment.total_training_steps = datamodule.total_training_steps

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
        accumulate_grad_batches=cfg.experiment.grad_accumulation,
        gradient_clip_val=cfg.experiment.grad_clip_val,
    )

    if cfg.load_weight and cfg.resume:
        raise ValueError("Only checkpoint or finetune")

    wrapper = DINOBackboneWrapper(cfg)
    wrapper.setup("fit")
    
    trainer.test(
        wrapper, dataloaders=train_dataloader
    )
    
    trainer.strategy.barrier()
