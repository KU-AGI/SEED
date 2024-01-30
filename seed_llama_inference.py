import hydra
from omegaconf import OmegaConf
from PIL import Image
import pyrootutils
import os
import glob
import hydra

import pyrootutils
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch

from omegaconf import OmegaConf
import json
from typing import Any, Optional
import transformers
from PIL import Image
import torchvision
from torchvision.transforms.functional import InterpolationMode

from torch.utils.data import Dataset, ConcatDataset, DataLoader
from pycocotools.coco import COCO
from tqdm import tqdm
import argparse

from diffusers import DiffusionPipeline
from diffusers.utils import load_image
from pytorch_lightning import LightningModule, seed_everything, Trainer
from models.pipeline_stable_unclip_img2img import StableUnCLIPImg2ImgPipeline
from models.seed_llama_tokenizer import ImageTokenizer

from utils.config import build_config

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

class SEEDi2iPipeline(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.result_file_path = cfg.result_file_path

        self.image_tokenizer = ImageTokenizer(model_path=cfg.checkpoint_path.model_path,
                            fp16=True,
                            from_pretrained=True,
                            diffusion_model_path=cfg.checkpoint_path.diffusion_model_path,
                            load_diffusion=True,
                            is_train=False,
                            )

        # For DDP
        self.feature_extractor = self.image_tokenizer.diffusion_model.feature_extractor
        self.image_encoder = self.image_tokenizer.diffusion_model.image_encoder
        self.image_normalizer = self.image_tokenizer.diffusion_model.image_normalizer
        self.image_noising_scheduler = self.image_tokenizer.diffusion_model.image_noising_scheduler
        self.tokenizer = self.image_tokenizer.diffusion_model.tokenizer
        self.text_encoder = self.image_tokenizer.diffusion_model.text_encoder
        self.unet = self.image_tokenizer.diffusion_model.unet
        self.scheduler = self.image_tokenizer.diffusion_model.scheduler
        self.vae = self.image_tokenizer.diffusion_model.vae
    
    def drop_randomly(self, x, drop_prob):
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

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        image, image_path = batch
        with torch.no_grad():
            image_indices = self.image_tokenizer.encode(image_torch=image)
            DROP_PROB = 0.7
            if DROP_PROB > 0:
                image_indices = self.drop_randomly(image_indices, drop_prob=DROP_PROB)
            image_variations = self.image_tokenizer.decode(indices=image_indices)

        for image_name, image_variation in zip(tqdm(image_path, desc=f"CUDA:{self.global_rank} Save"), image_variations):
            image_name = os.path.basename(image_name)
            save_path = os.path.join(self.result_file_path, image_name)

            # Image variation is PIL image
            image_variation.save(save_path)

        return


class COCODataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image_path

def create_data_loader(image_paths, transform, batch_size=8):
    dataset = COCODataset(image_paths, transform)
    return DataLoader(dataset, batch_size=batch_size, num_workers=32, shuffle=False)

if __name__ == "__main__":
    cfg, cfg_yaml = build_config()

    with open("coco/annotations/karpathy/dataset_coco_test.json", "r") as f:
        coco_test = json.load(f)

    coco_test_imgs = sorted([
        '/ssd0/data/coco/images/val2014/' + img["file_name"]
        for img in coco_test["images"]
    ])

    # cfg.tokenizer_cfg_path = "configs/tokenizer/seed_llama_tokenizer.yaml"
    seed_everything(cfg.experiment.seed, workers=True)

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    cfg.result_file_path = "seed_tokenizer_img2img_variation_coco_karpathy_test_drop_70"
    os.makedirs(cfg.result_file_path, exist_ok=True)

    data_loader = create_data_loader(coco_test_imgs, transform, batch_size=16)

    test_wrapper = SEEDi2iPipeline(cfg)

    trainer = Trainer(
        devices=4,
        accelerator="cuda",
        strategy="ddp",
        max_epochs=1,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.test(test_wrapper, dataloaders=data_loader)