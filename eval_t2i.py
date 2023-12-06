#from torchmetrics.image.fid import FrechetInceptionDistance
from utils.fid.fid import FrechetInceptionDistance
from utils.fid.inception_score import InceptionScore

import os
from typing import Any, List
import torch
from torch.cuda.amp import autocast
from torch.nn import functional as F

from tqdm import tqdm
import time
import json

import argparse

import argparse
import time

import hydra
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, seed_everything
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from utils.config import build_config
import pyrootutils
from datamodules import build_datamodule
from datamodules.tokenizers import TokenizerUtils
from datamodules.datasets.dataclass import Items
from datamodules.utils import convert_image_to_rgb
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import InterpolationMode
to_tensor = ToTensor()

import numpy as np
from PIL import Image
import pdb
import pickle
import glob
import math

from einops import rearrange
import clip

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

IMG_FLAG = '<image>'
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 8192
image_id_shift = 32000

GENERATION_NUM = 30000

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def save_pickle(fname, data):
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)

def load_fake_samples(cfg):
    fnames = glob.glob(os.path.join(cfg.result_file_path, '*.pkl'))
    pixelss = []
    for fname in tqdm(fnames, total=len(fnames)):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        data_ = np.concatenate(data['pixelss'], axis=0)
        data_ = torch.from_numpy(data_).to(dtype=torch.uint8)
        pixelss.append(data_)
    pixelss = torch.cat(pixelss, dim=0)

    return pixelss

def load_real_samples(cfg):
    fid_transform = transforms.Compose([
        transforms.Resize(
                768, interpolation=InterpolationMode.BICUBIC, antialias=True
            ),
        transforms.CenterCrop(768),
        convert_image_to_rgb,
        transforms.PILToTensor(),
    ])

    datamodule = build_datamodule(
            cfg=cfg,
            train_transform=None,
            val_transform=fid_transform,
            pin_memory=False,
            epoch=0,
            total_gpus=1,
        )

    datamodule.setup()
    test_dataloader = datamodule.val_dataloader()
    test_dataloader.dataset.set_custom_length(GENERATION_NUM)

    imgss = []
    for it, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        imgs = batch.img.type(torch.uint8)
        imgss.append(imgs)
    imgss = torch.cat(imgss, dim=0)

    real_samples = imgss
    return real_samples

def clip_score(compare_img_set: tuple, clip_set: tuple, device):
    pixel_1, pixel_2 = compare_img_set
    model_clip, preprocess_clip = clip_set

    images_1 = [preprocess_clip(Image.fromarray(np.array(rearrange(pixel, 'c b h -> b h c')))) for pixel in pixel_1]
    images_1 = torch.stack(images_1, dim=0).to(device)
    images_2 = [preprocess_clip(Image.fromarray(np.array(rearrange(pixel, 'c b h -> b h c')))) for pixel in pixel_2]
    images_2 = torch.stack(images_2, dim=0).to(device)

    image_1_feature = model_clip.encode_image(images_1)
    image_2_feature = model_clip.encode_image(images_2)

    scores = torch.nn.functional.cosine_similarity(image_1_feature, image_2_feature)
    return scores


class T2IEvalWrapper(LightningModule):
    def __init__(self, cfg, model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

        self.generation_config = {
            'num_beams': 5,
            'max_new_tokens': 512,
            'do_sample': False
        }

        self.s_token = "USER:"
        self.e_token = "ASSISTANT:"
        self.sep = "\n"

        self.task = 'caption'
        if self.cfg.dataset.name.startswith('vqa'):
            self.task = 'vqa'
        self.outputss = []
        self.txtss = []

    def generate(self, tokenizer, input_tokens, generation_config, model):

        input_ids = tokenizer(input_tokens, add_special_tokens=False, return_tensors='pt').input_ids
        input_ids = input_ids.to("cuda")

        torch.use_deterministic_algorithms(False)
        generate_ids = model.generate(
            input_ids=input_ids,
            **generation_config
        )
        torch.use_deterministic_algorithms(True)
        generate_ids = generate_ids[0][input_ids.shape[1]:]
        
        return generate_ids

    def decode_image_text(self, generate_ids, tokenizer):
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

            # image is PIL
            images = tokenizer.decode_image(image_ids)[0]

            return images

    def test_step(self, batch: Items, batch_idx: int):
        caption = ' '.join(batch.gt_txt[0])
        if batch.imgpath[0] in os.listdir(f"{self.cfg.result_file_path}"):
            print(f"Already generated {batch.imgpath[0]}")
            generated_image = Image.open(f"{self.cfg.result_file_path}/{batch.imgpath[0]}")
        else:
            # USER: {caption} Please generation an image.\nASSISTANT: {image}
            prompt = "Please generation an image."
            input_tokens = f"{self.tokenizer.bos_token} {self.s_token} {caption} {prompt} {self.sep} {self.e_token}"
            try:
                generate_ids = self.generate(self.tokenizer, input_tokens, self.generation_config, self.model)
                generated_image = self.decode_image_text(generate_ids, self.tokenizer)
                generated_image.save(f"{self.cfg.result_file_path}/{batch.imgpath[0]}")
            except:
                print(f"Error occured at {batch.imgpath[0]}")
                with open(f"{self.cfg.result_file_path}/error.json", 'a') as f:
                    json.dump({'imgpath': batch.imgpath[0], 'caption': caption}, f, indent='\t')
                return
        
        rgb_image = generated_image.convert('RGB')
        rgb_image = rearrange(np.array(rgb_image), 'h w c -> 1 c h w')
        self.outputss.append(rgb_image)

        self.txtss += [caption]

        if (batch_idx + 1) % 100 == 0:
            save_pickle(os.path.join(self.cfg.result_file_path, f'samples_iter_{batch_idx + 1}.pkl'),
                        {'pixelss': self.outputss,
                         'txtss': self.txtss})
            self.outputss = []
            print(f'done file dump at {self.cfg.result_file_path}')

        return

    def on_test_epoch_end(self):
        if len(self.outputss) > 0:
            save_pickle(os.path.join(self.cfg.result_file_path, f'samples_rank_{self.global_rank}_iter_last.pkl'),
                        {'pixelss': self.outputss,
                         'txtss': self.txtss})
        print(f'done file dump at {self.cfg.result_file_path}')



if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    cfg, cfg_yaml = build_config()
    device = "cuda"
    os.makedirs(cfg.result_file_path, exist_ok=True)

    cfg.dataset.eval_center_crop = True

    cfg.dataset.name = 'coco2014'
    cfg.dataset.type = 'mapstyle'
    cfg.dataset.gt_text = True

    total_gpus = cfg.dist.n_gpus * cfg.dist.n_nodes

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    tokenizer_cfg = OmegaConf.load(cfg.tokenizer_cfg_path)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg, device=device, load_diffusion=True)

    model_cfg = OmegaConf.load(cfg.model_cfg_path)
    # Pass model load for debugging
    model = hydra.utils.instantiate(model_cfg, torch_dtype=torch.float16)
    model = model.eval()

    seed_everything(cfg.experiment.seed, workers=True)

    datamodule = build_datamodule(
        cfg=cfg,
        train_transform=None,
        val_transform=transform,
        pin_memory=False,
        epoch=0,
        total_gpus=total_gpus,
    )

    datamodule.setup()
    rank = int(os.environ["RANK"]) if "RANK" in os.environ.keys() else 0


    test_dataloader = datamodule.val_dataloader()
    test_dataloader.dataset.set_custom_length(GENERATION_NUM)

    from pytorch_lightning.strategies import DDPStrategy
    trainer = pl.Trainer(
        accelerator="gpu",
        num_nodes=cfg.dist.n_nodes,
        devices=cfg.dist.n_gpus,
        strategy=DDPStrategy(
            find_unused_parameters=False,
        ),
        max_epochs=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        deterministic=True,
    )

    # Pass trainer step for debugging
    wrapper = T2IEvalWrapper(cfg, model, tokenizer)
    wrapper.eval()

    trainer.test(wrapper, dataloaders=test_dataloader)
    trainer.strategy.barrier()

    torch.distributed.destroy_process_group()

    if trainer.global_rank == 0:
        # Road fake samples
        fake_samples = load_fake_samples(cfg)

        # Road real samples
        # If image is not normalized, 
        real_samples = load_real_samples(cfg)

        r_idx = np.random.permutation(
            min(fake_samples.shape[0], real_samples.shape[0])
        )[:30000]

        real_samples = real_samples[r_idx]
        fake_samples = fake_samples[r_idx]

        fid = FrechetInceptionDistance().cuda()
        fid_batch_size = 10

        # Debug
        # FID Score

        n_batches = min(math.ceil(fake_samples.shape[0] / fid_batch_size), math.ceil(real_samples.shape[0] / fid_batch_size))
        for i in tqdm(range(n_batches), total=n_batches):
            sp = i * fid_batch_size
            ep = (i + 1) * fid_batch_size

            real_samples_ = real_samples[sp:ep]
            fake_samples_ = fake_samples[sp:ep]

            fid.update(real_samples_.cuda(), real=True)
            fid.update(fake_samples_.cuda(), real=False)

        fid_score = fid.compute()

        print(cfg.result_file_path)
        print(f"FID: {fid_score}")

        # IS Score

        incs = InceptionScore().cuda()
        incs_batch_size = 100
        n_fake_batches = math.ceil(fake_samples.shape[0] / incs_batch_size)
        for i in tqdm(range(n_fake_batches), total=n_fake_batches):
            sp = i * incs_batch_size
            ep = (i + 1) * incs_batch_size
            fake_samples_ = fake_samples[sp:ep]
            incs.update(fake_samples_.cuda())

        inception_score = incs.compute()

        print(f"No CLIP Re-ranking, IS mean: {inception_score[0]:.4f}, IS std: {inception_score[1]:.4f}")

        # CLIP image-to-image similarity
        model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
        clip_batch_size = 100
        n_fake_batches = math.ceil(fake_samples.shape[0] / clip_batch_size)
        pdb.set_trace()
        total_clip_scores = 0
        for i in tqdm(range(n_fake_batches), total=n_fake_batches):
            sp = i * clip_batch_size
            ep = (i + 1) * clip_batch_size
            fake_samples_ = fake_samples[sp:ep]
            real_samples_ = real_samples[sp:ep]
            clip_scores = clip_score((fake_samples_, real_samples_), (model_clip, preprocess_clip), device=device)
            print(f"Current batch clip scores : {clip_scores}")
            total_clip_scores += clip_scores.sum().item()
        total_clip_scores /= n_fake_batches * clip_batch_size
        print(f"Clip score: {total_clip_scores}")
        


    print('done!')