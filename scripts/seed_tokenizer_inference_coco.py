import hydra
from omegaconf import OmegaConf
from PIL import Image
import pyrootutils
import os
import glob
import hydra

import pyrootutils
import os
import torch

from omegaconf import OmegaConf
import json
from typing import Optional
import transformers
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


from torch.utils.data import Dataset, ConcatDataset, DataLoader
from pycocotools.coco import COCO
from tqdm import tqdm
import json
import argparse

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

tokenizer_cfg_path = 'configs/tokenizer/seed_llama_tokenizer_hf.yaml'
transform_cfg_path = 'configs/transform/clip_transform.yaml'

image_path = 'images/cat.jpg'
save_dir = './tokenizer_inference_coco'
save_path = os.path.join(save_dir, os.path.basename(image_path))

os.makedirs(save_dir, exist_ok=True)
already_exists_imgs = set(os.listdir(save_dir))

device = 'cuda'

tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
tokenizer = hydra.utils.instantiate(tokenizer_cfg, device=device, load_diffusion=True)

transform_cfg = OmegaConf.load(transform_cfg_path)
transform = hydra.utils.instantiate(transform_cfg)

image = Image.open(image_path).convert('RGB')

image_tensor = transform(image).to(device)
image_ids = tokenizer.encode_image(image_torch=image_tensor)

images = tokenizer.decode_image(image_ids)

# images[0].save(save_path)

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
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

with open("coco/annotations/karpathy/dataset_coco_test.json", "r") as f:
    coco_test = json.load(f)

coco_test_imgs = [
    '/ssd0/data/coco/images/val2014/' + img["file_name"]
    for img in coco_test["images"]
]

data_loader = create_data_loader(coco_test_imgs, transform, batch_size=1)

for batch in tqdm(data_loader):
    image, image_path = batch
    if os.path.basename(image_path[0]) in already_exists_imgs:
        print(f"Skipping {image_path[0]}")
        continue
    image = image.to(device)
    image_ids = tokenizer.encode_image(image_torch=image)
    images = tokenizer.decode_image(image_ids)
    images[0].save(os.path.join(save_dir, os.path.basename(image_path[0])))