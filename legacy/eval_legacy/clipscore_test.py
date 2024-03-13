import torch
from PIL import Image
import clip
import pdb
from torchvision.transforms import ToTensor
from torchvision import transforms
from einops import rearrange
import glob
import os
from tqdm import tqdm
import pickle
import numpy as np

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


def clip_score(compare_img_set: tuple, clip_set: tuple):
    pil_1, pil_2 = compare_img_set
    model_clip, preprocess_clip = clip_set
    model_clip.eval()

    pixel_1 = transform_PIL_to_pixels(pil_1)
    pixel_2 = transform_PIL_to_pixels(pil_2)

    processed_1 = rearrange(preprocess_clip(pil_1), 'c h w -> 1 c h w').to(device)
    processed_2 = rearrange(preprocess_clip(pil_2), 'c h w -> 1 c h w').to(device)

    # model_clip input must have 4 dimension
    # b c h w, and naive pixels
    with torch.no_grad():
        image_1_feature = model_clip.encode_image(processed_1)
        image_2_feature = model_clip.encode_image(processed_2)

    scores = torch.nn.functional.cosine_similarity(image_1_feature, image_2_feature, dim=1)
    return scores

device = "cuda"
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)

total_clip_score = 0
cnt = 0
testset_path = 'i2i_reconstruction'
for img_name in tqdm(os.listdir(testset_path)):
    if img_name.split(".")[-1] != 'jpg':
        continue
    img_1 = Image.open(f'{testset_path}/{img_name}')
    img_2 = Image.open(f'coco/images/val2014/{img_name}')
    cnt += 1
    cur_clip_score = clip_score((img_1, img_2), (model_clip, preprocess_clip))
    total_clip_score += cur_clip_score
    print(f"{img_name} CLIP sim: {cur_clip_score}")

print(f"Average CLIP similarity: {total_clip_score / cnt}")