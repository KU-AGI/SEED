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

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

IMG_FLAG = '<image>'
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 8192
IMAGE_ID_SHIFT = 32000

s_token = "USER:"
e_token = "ASSISTANT:"
sep = "\n"

def generate(tokenizer, input_tokens, generation_config, model):

    input_ids = tokenizer(input_tokens, add_special_tokens=False, return_tensors='pt').input_ids
    input_ids = input_ids.to("cuda")

    generate_ids = model.generate(
        input_ids=input_ids,
        **generation_config
    )
    generate_ids = generate_ids[0][input_ids.shape[1]:]
    
    return generate_ids

def decode_image_text(generate_ids, tokenizer, save_path=None):

    boi_list = torch.where(generate_ids == tokenizer(BOI_TOKEN, add_special_tokens=False).input_ids[0])[0]
    eoi_list = torch.where(generate_ids == tokenizer(EOI_TOKEN, add_special_tokens=False).input_ids[0])[0]

    if len(boi_list) == 0 and len(eoi_list) == 0:
        text_ids = generate_ids
        texts = tokenizer.decode(text_ids, skip_special_tokens=True)
        return texts

    else:
        boi_index = boi_list[0]
        eoi_index = eoi_list[0]

        text_ids = generate_ids[:boi_index]
        if len(text_ids) != 0:
            texts = tokenizer.decode(text_ids, skip_special_tokens=True)
            print(texts)
            
        image_ids = (generate_ids[boi_index+1:eoi_index] - IMAGE_ID_SHIFT).reshape(1,-1)

        images = tokenizer.decode_image(image_ids)

        images[0].save(save_path)

def make_t2i_input_tokens(prompt, tokenizer, is_instruction=True):
    if is_instruction:
        return f"{tokenizer.bos_token}USER: {prompt} Please generation an image.\nASSISTANT: "
    else:
        return f"{tokenizer.bos_token} {prompt} {tokenizer.eos_token} {BOI_TOKEN}"

class DCIDataset(Dataset):
    """_summary_
        Define a custom dataset for COCO images
    Args:
        Dataset (_type_): _description_
    """    
    def __init__(self):
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

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_save_path", type=str, default=None)
    parser.add_argument("--use_short_or_long", type=str, default=None)
    args = parser.parse_args()

    if args.use_short_or_long is None:
        print("use_short_or_long is not specified")
        exit()
    elif args.use_short_or_long != "short" and args.use_short_or_long != "long":
        print("use_short_or_long is be either short or long")
        exit()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    tokenizer_cfg_path = 'configs/tokenizer/seed_llama_tokenizer_hf.yaml'
    tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg, device=device, load_diffusion=True)

    transform_cfg_path = 'configs/transform/clip_transform.yaml'
    transform_cfg = OmegaConf.load(transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    model_cfg = OmegaConf.load('configs/llm/seed_llama_8b.yaml')
    model = hydra.utils.instantiate(model_cfg, torch_dtype=torch.float16)
    model = model.eval().to(device)

    generation_config = {
            'temperature': 1.0,
            'num_beams': 5,
            'max_new_tokens': 512,
            'do_sample': False
        }

    if args.image_save_path is None:
        print("image_path is not specified")
        exit()
    else:
        os.makedirs(args.image_save_path, exist_ok=True)

    results = []
    processed_set = set()

    # Create a data loader with your specified batch size
    dataset = DCIDataset()
    # Now batch size 1
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in tqdm(data_loader):
        while True:
            try:
                image_name, short_caption, long_caption = batch

                if args.use_short_or_long == "short":
                    prompt = short_caption[0]
                elif args.use_short_or_long == "long":
                    prompt = long_caption[0]
                else:
                    print("use_short_or_long is be either short or long")
                    exit()

                # Make input tokens
                input_tokens = make_t2i_input_tokens(prompt, tokenizer)
                generate_ids = generate(tokenizer, input_tokens, generation_config, model)

                # Save generated image
                save_path = f"{args.image_save_path}/{image_name[0]}"
                decode_image_text(generate_ids, tokenizer, save_path)
                break
            except Exception as e:
                print(f"Exception : {e}")
                continue