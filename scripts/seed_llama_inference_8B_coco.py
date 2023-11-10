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

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

IMG_FLAG = '<image>'
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 8192
image_id_shift = 32000

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
            
        image_ids = (generate_ids[boi_index+1:eoi_index] - image_id_shift).reshape(1,-1)

        images = tokenizer.decode_image(image_ids)

        images[0].save(save_path)


device = "cuda"

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
        'num_beams': 1,
        'max_new_tokens': 512,
        'top_p': 0.5,
        'do_sample': True
    }

s_token = "USER:"
e_token = "ASSISTANT:"
sep = "\n"

with open("coco/annotations/karpathy/dataset_coco_test.json", "r") as f:
    coco_test = json.load(f)

coco_test_imgs = [
    '/ssd0/data/coco/images/val2014/' + img["file_name"]
    for img in coco_test["images"]
]

coco_file_name_id = {
    img["file_name"]: img["id"]
    for img in coco_test["images"]
}

coco_img_id_caption = {
    img["image_id"]: img["caption"]
    for img in coco_test["annotations"]
}

def make_t2i_input_tokens(prompt, tokenizer):
    return tokenizer.bos_token + s_token + " " + prompt + sep + e_token

# Define a custom dataset for COCO images
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

# Define a DataLoader for batch processing
def create_data_loader(image_paths, transform, batch_size=16):
    dataset = COCODataset(image_paths, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def img_ids_to_img_tokens(img_ids):
    return BOI_TOKEN + ''.join([IMG_TOKEN.format(item) for item in img_ids]) + EOI_TOKEN

def img_tokens_to_input_tokens(img_tokens, prompt):
    return tokenizer.bos_token + s_token + " " + img_tokens + prompt + sep + e_token

def get_file_name(path):
    return path.split("/")[-1]

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--mode", type=str, default="i2t")
    args = parser.parse_args()

    if args.mode == "i2t":
        if args.result_file is None:
            print("result_file is not specified")
            exit()
    if args.mode == "t2i":
        if args.image_path is None:
            print("image_path is not specified")
            exit()
        if not os.path.exists(args.image_path):
            os.makedirs(args.image_path, exist_ok=True)

    results = []
    processed_set = set()

    # Create a data loader with your specified batch size
    data_loader = create_data_loader(coco_test_imgs, transform, batch_size=1)

    if args.mode == "i2t":
        # continue from the last processed file
        if os.path.exists(args.result_file):
            with open(args.result_file, "r") as f:
                results = json.load(f)
                for result in results:
                    processed_set.add(result["image_id"])

        for batch in tqdm(data_loader):
            images, paths = batch

            # Check if the file is already processed
            if coco_file_name_id[get_file_name(paths[0])] in processed_set:
                print(f"Skip file name : {get_file_name(paths[0])}, id : {coco_file_name_id[get_file_name(paths[0])]}")
                continue

            # Now, iterate over the data loader
            # images is a tensor of shape (batch_size, 3, 256, 256)
            image_tensor = images.to(device)
            # img_ids is a tensor of shape (batch_size, 32)
            batch_img_ids = tokenizer.encode_image(image_torch=image_tensor)
            # img_tokens is a tensor of shape (batch_size, 32)
            batch_input_tokens = []
            question = "Caption:"

            for img_ids in batch_img_ids:
                img_ids = img_ids.view(-1).cpu().numpy()
                img_tokens = img_ids_to_img_tokens(img_ids)
                batch_input_tokens.append(img_tokens_to_input_tokens(img_tokens, question))
            
            input_ids = None
            for input_tokens in batch_input_tokens:
                if input_ids is None:
                    # Shape [1, 48]
                    input_ids = tokenizer(input_tokens, add_special_tokens=False, return_tensors='pt').input_ids
                else:
                    # input_ids = torch.cat([input_ids, tokenizer(input_tokens, add_special_tokens=False, return_tensors='pt').input_ids], dim=0)
                    print("multiple images in a batch is not supported yet")
                    exit()

            input_ids = input_ids.to("cuda")
            generate_ids = model.generate(
                input_ids=input_ids,
                **generation_config
            )

            generate_ids = generate_ids[0][input_ids.shape[1]:]
            prediction = decode_image_text(generate_ids, tokenizer)
            results.append({"image_id": coco_file_name_id[get_file_name(paths[0])], "caption": prediction})

            with open(args.result_file, "w") as f:
                json.dump(results, f, indent='\t')
    elif args.mode == "t2i":
        for batch in tqdm(data_loader):
            while True:
                try:
                    images, paths = batch
                    prompt = coco_img_id_caption[coco_file_name_id[get_file_name(paths[0])]]
                    input_tokens = tokenizer.bos_token  + s_token + " " + prompt + sep + e_token
                    generate_ids = generate(tokenizer, input_tokens, generation_config, model)
                    save_path = args.image_path + "/" + get_file_name(paths[0])
                    decode_image_text(generate_ids, tokenizer, save_path)
                    break
                except Exception as e:
                    print(f"Exception : {e}")
                    continue
    else:
        print("invalid mode")
        exit()
