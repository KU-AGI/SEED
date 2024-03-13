import os
from utils.fid.fid import FrechetInceptionDistance
from tqdm import tqdm
import math
from PIL import Image
import torch
import numpy as np
from einops import rearrange
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import pdb
import glob
import pickle

#created_img_path = 'coco_images_resize_fid_27_0'
created_img_path = 'i2i_reconstruction'

original_img_list = os.listdir(created_img_path)
img_list = []

for idx, img_name in enumerate(original_img_list):
    if img_name[-4:] != '.jpg':
        continue
    else:
        img_list.append(img_name)
    
    
real_samples = [Image.open(f"coco/images/val2014/{img}") for img in img_list]
fake_samples = [Image.open(f"{created_img_path}/{img}") for img in img_list]

print(f"Generated image number: {len(img_list)}")
fid = FrechetInceptionDistance().cuda()
fid_batch_size = 2000

def transform_PIL_to_pixels(pil_img: Image):
    pil_img = pil_img.convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize(299, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(299),
            transforms.PILToTensor(),
        ]
    )
    return preprocess(pil_img)

# FID Score

n_batches = min(math.ceil(len(fake_samples) / fid_batch_size), math.ceil(len(real_samples) / fid_batch_size))

for i in tqdm(range(n_batches), total=n_batches):
    sp = i * fid_batch_size
    ep = (i + 1) * fid_batch_size

    real_samples_ = real_samples[sp:ep]
    fake_samples_ = fake_samples[sp:ep]

    # PIL convert to 3 channel RGB
    real_samples_ = torch.stack([transform_PIL_to_pixels(img) for img in real_samples_], dim=0)
    fake_samples_ = torch.stack([transform_PIL_to_pixels(img) for img in fake_samples_], dim=0)

    fid.update(real_samples_.cuda(), real=True)
    fid.update(fake_samples_.cuda(), real=False)

fid_score = fid.compute()

print(f"FID: {fid_score}")

