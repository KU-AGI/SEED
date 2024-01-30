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

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

IMG_FLAG = '<image>'
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 8192
image_id_shift = 32000

class StableDiffusionPipeline(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.output_path = "stable_diffusion_img2img_variation_coco_karpathy_test"
        os.makedirs(self.output_path, exist_ok=True)

        self.model = StableUnCLIPImg2ImgPipeline.from_pretrained(cfg.checkpoint_path.diffusion_model_path, torch_dtype=torch.float16)

        # For DDP
        self.feature_extractor = self.model.feature_extractor
        self.image_encoder = self.model.image_encoder
        self.image_normalizer = self.model.image_normalizer
        self.image_noising_scheduler = self.model.image_noising_scheduler
        self.tokenizer = self.model.tokenizer
        self.text_encoder = self.model.text_encoder
        self.unet = self.model.unet
        self.scheduler = self.model.scheduler
        self.vae = self.model.vae

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        image, image_path = batch
        with torch.no_grad():
            image_variations = self.model(image).images

        for image_name, image_variation in zip(tqdm(image_path, desc=f"CUDA:{self.global_rank} Save"), image_variations):
            image_name = os.path.basename(image_name)
            save_path = os.path.join(self.output_path, image_name)

            # Image variation is PIL image
            image_variation.save(save_path)

        return
    
    @torch.no_grad()
    def test_step_seed(self, batch, batch_idx):
        image, image_path = batch
        with torch.no_grad():
            image_variations = self.model(image).images

        for image_name, image_variation in zip(tqdm(image_path, desc=f"CUDA:{self.global_rank} Save"), image_variations):
            image_name = os.path.basename(image_name)
            save_path = os.path.join(self.output_path, image_name)

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

def create_data_loader(image_paths, transform, batch_size=16):
    dataset = COCODataset(image_paths, transform)
    return DataLoader(dataset, batch_size=batch_size, num_workers=32, shuffle=False)

if __name__ == "__main__":
    with open("coco/annotations/karpathy/dataset_coco_test.json", "r") as f:
        coco_test = json.load(f)

    coco_test_imgs = sorted([
        '/ssd0/data/coco/images/val2014/' + img["file_name"]
        for img in coco_test["images"]
    ])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    data_loader = create_data_loader(coco_test_imgs, transform, batch_size=16)

    test_wrapper = StableDiffusionPipeline(OmegaConf.load("configs/generation_config.yaml"))

    trainer = Trainer(
        devices=4,
        accelerator="cuda",
        strategy="ddp",
        max_epochs=1,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.test(test_wrapper, dataloaders=data_loader)