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
from tqdm import tqdm
import json
import argparse

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

ANNOTATION_ROOT = "/home/zheedong/Projects/DCI/data/densely_captioned_images/complete"
ORIGINAL_IMAGE_ROOT = "/home/zheedong/Projects/DCI/data/densely_captioned_images/photos"
LORA_MODEL_PATH = "/home/zheedong/Projects/DCI/models/dci_pick1"

def get_image_name(image_path):
    return image_path.split("/")[-1].split(".")[0]

def get_image_name_annotation_dict(annotation_root):
    image_name_annotation_dict = {}
    annotations = os.listdir(annotation_root)
    for annotation in tqdm(annotations):
        with open(os.path.join(annotation_root, annotation), encoding="utf-8") as f:
            data = json.load(f)
        image_name_annotation_dict[get_image_name(data["image"])] = annotation
    return image_name_annotation_dict

def get_annotation_path(image_name, image_name_annotation_dict):
    return f"{ANNOTATION_ROOT}/{image_name_annotation_dict[get_image_name(image_name)]}"

def get_longset_caption(data):
    longest_caption = ""
    for i in data['summaries']['base']:
        if len(i) > len(longest_caption):
            longest_caption = i
    return longest_caption

class DCIDataset(Dataset):
    def __init__(self, image_root, transforms=None):
        self.image_root = image_root
        self.image_name_annotation_dict = get_image_name_annotation_dict(ANNOTATION_ROOT)
        self.image_path_list = sorted(os.listdir(image_root))

        # Filter out images that have no annotation
        self.valid_image_path_list = [img for img in self.image_path_list if get_image_name(img) in self.image_name_annotation_dict]

        print(f"Total {len(self.image_path_list)} images, {len(self.valid_image_path_list)} images have annotations")

        self.transforms = transforms

    def __len__(self):
        return len(self.valid_image_path_list)

    def __getitem__(self, idx):
        image_name = self.valid_image_path_list[idx]

        try:
            with open(get_annotation_path(image_name, self.image_name_annotation_dict), encoding="utf-8") as f:
                data = json.load(f)

            image_path = os.path.join(self.image_root, image_name)
            image = Image.open(image_path).convert("RGB")
            if self.transforms:
                image = self.transforms(image)

            # short_caption = data["short_caption"]
            # summarized_long = get_longset_caption(data)
            # extra_caption = data["extra_caption"]
            
            return image, image_path#, short_caption, summarized_long, extra_caption
        except:
            print(f"Error: {image_name}")
            return None, None, None

if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = "tokenizer_inference_DCI"

    # Check if the save path exists
    os.makedirs(save_path, exist_ok=True)
    already_exists_imgs = os.listdir(save_path)

    tokenizer_cfg_path = 'configs/tokenizer/seed_llama_tokenizer_hf.yaml'
    transform_cfg_path = 'configs/transform/clip_transform.yaml'

    tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg, device=device, load_diffusion=True)

    transform_cfg = OmegaConf.load(transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    dataset = DCIDataset(ORIGINAL_IMAGE_ROOT, transforms=transform)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False)

    for batch in tqdm(data_loader):
        image, image_path = batch
        if os.path.basename(image_path[0]) in already_exists_imgs:
            print(f"Skipping {image_path[0]}")
            continue
        image = image.to(device)
        image_ids = tokenizer.encode_image(image_torch=image)
        images = tokenizer.decode_image(image_ids)
        for idx, image in enumerate(images):
            image.save(os.path.join(save_path, os.path.basename(image_path[idx])))