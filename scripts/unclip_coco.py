import torch
from diffusers import UnCLIPScheduler, DDPMScheduler, StableUnCLIPPipeline
from diffusers.models import PriorTransformer
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from tqdm import tqdm
import json
import os
from omegaconf import OmegaConf
import hydra
import argparse
from torchvision import transforms

prior_model_id = "kakaobrain/karlo-v1-alpha"
data_type = torch.float16
prior = PriorTransformer.from_pretrained(prior_model_id, subfolder="prior", torch_dtype=data_type)

prior_text_model_id = "openai/clip-vit-large-patch14"
prior_tokenizer = CLIPTokenizer.from_pretrained(prior_text_model_id)
prior_text_model = CLIPTextModelWithProjection.from_pretrained(prior_text_model_id, torch_dtype=data_type)
prior_scheduler = UnCLIPScheduler.from_pretrained(prior_model_id, subfolder="prior_scheduler")
prior_scheduler = DDPMScheduler.from_config(prior_scheduler.config)

stable_unclip_model_id = "stabilityai/stable-diffusion-2-1-unclip-small"

pipe = StableUnCLIPPipeline.from_pretrained(
    stable_unclip_model_id,
    torch_dtype=data_type,
    variant="fp16",
    prior_tokenizer=prior_tokenizer,
    prior_text_encoder=prior_text_model,
    prior=prior,
    prior_scheduler=prior_scheduler,
)

pipe = pipe.to("cuda")
'''
wave_prompt = "dramatic wave, the Oceans roar, Strong wave spiral across the oceans as the waves unfurl into roaring crests; perfect wave form; perfect wave shape; dramatic wave shape; wave shape unbelievable; wave; wave shape spectacular"

images = pipe(prompt=wave_prompt).images
images[0].save("waves.png")
'''
def get_transform(type='clip', keep_ratio=True, image_size=224):
    if type == 'clip':
        transform = []
        if keep_ratio:
            transform.extend([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
            ])
        else:
            transform.append(transforms.Resize((image_size, image_size)))
        transform.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        return transforms.Compose(transform)

'''
transform_cfg_path = 'configs/transform/clip_transform.yaml'
transform_cfg = OmegaConf.load(transform_cfg_path)
transform = hydra.utils.instantiate(transform_cfg)
'''
transform = get_transform(type='clip', keep_ratio=True, image_size=224)

with open("coco/annotations/karpathy/dataset_coco_test.json", "r") as f:
    coco_test = json.load(f)

coco_img_id_caption = {
    img["image_id"]: img["caption"]
    for img in coco_test["annotations"]
}

coco_file_name_id = {
    img["file_name"]: img["id"]
    for img in coco_test["images"]
}

def get_file_name(path):
    return path.split("/")[-1]

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
device = 'cuda'

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
        print(f"Created save path {args.save_path}")
    
    already_exists_imgs = set(os.listdir(args.save_path))

    for batch in tqdm(data_loader):
        image, image_path = batch
        if os.path.basename(image_path[0]) in already_exists_imgs:
            print(f"Skipping {image_path[0]}")
            continue
        prompt = coco_img_id_caption[coco_file_name_id[get_file_name(image_path[0])]]
        images = pipe(prompt=prompt).images
        images[0].save(os.path.join(args.save_path, os.path.basename(image_path[0])))