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
from diffusers import StableDiffusionPipeline

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

IMG_FLAG = '<image>'
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 8192
image_id_shift = 32000

class SDGenerationWrapper(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.output_path = "stable_diffusion_text2img_generation_DCI_8000"
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "short_caption_generated"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "long_caption_generated"), exist_ok=True)

        model_id = "stabilityai/stable-diffusion-2-1"
        self.model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

        # For DDP
        self.vae = self.model.vae
        self.text_encoder = self.model.text_encoder
        self.tokenizer = self.model.tokenizer
        self.unet = self.model.unet
        self.scheduler = self.model.scheduler
        self.feature_extractor = self.model.feature_extractor

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        image_name_list, short_caption, long_caption = batch
        with torch.no_grad():
            short_caption_generated_images = self.model(list(short_caption)).images
            long_caption_generated_images = self.model(list(long_caption)).images

        for image_name, generated_image in zip(tqdm(image_name_list, desc=f"CUDA:{self.global_rank} Short Caption Save"), short_caption_generated_images):
            save_path = os.path.join(self.output_path, "short_caption_generated", image_name)

            # Image variation is PIL image
            generated_image.save(save_path)

        for image_name, generated_image in zip(tqdm(image_name_list, desc=f"CUDA:{self.global_rank} Long Caption Save"), long_caption_generated_images):
            save_path = os.path.join(self.output_path, "long_caption_generated", image_name)

            # Image variation is PIL image
            generated_image.save(save_path)

        return

class DCIDataset(Dataset):
    """_summary_
        Define a custom dataset for COCO images
    Args:
        Dataset (_type_): _description_
    """    
    def __init__(self):
        # self.data_path = "/home/byeongguk/projects/magvlt2/MLLM_Evaluations/selected_captions_with_short_caption.json"
        self.data_path = "/home/zheedong/Projects/SEED/dci_random_captions_choice.json"
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.data = sorted(self.data, key=lambda x: x["image"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data[idx]["image"]
        short_caption = self.data[idx]["short_caption"]
        long_caption = self.data[idx]["long_caption"]

        return image_name, short_caption, long_caption

if __name__ == "__main__":
    dataset = DCIDataset()
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=16)

    test_wrapper = SDGenerationWrapper(OmegaConf.load("configs/generation_config.yaml"))

    trainer = Trainer(
        devices=1,
        accelerator="cuda",
        strategy="ddp",
        max_epochs=1,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.test(test_wrapper, dataloaders=data_loader)