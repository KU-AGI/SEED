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

import numpy as np
from PIL import Image
import pdb

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

IMG_FLAG = '<image>'
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 8192
image_id_shift = 32000

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class I2TEvalWrapper(LightningModule):
    def __init__(self, cfg, model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

        self.generation_config = {
            'num_beams': 5,
            'max_new_tokens': 32,
            'do_sample': False
        }

        self.s_token = "USER:"
        self.e_token = "ASSISTANT:"
        self.sep = "\n"

        self.task = 'caption'
        if self.cfg.dataset.name.startswith('vqa'):
            self.task = 'vqa'
        self.result = []

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
        elif len(boi_list) == 0 or len(eoi_list) == 0:
            print("Image caption generation failed")
            return ""
        else:
            boi_index = boi_list[0]
            eoi_index = eoi_list[0]

            text_ids = generate_ids[:boi_index]
            if len(text_ids) != 0:
                texts = tokenizer.decode(text_ids, skip_special_tokens=True)
                print(texts)
                
            image_ids = (generate_ids[boi_index+1:eoi_index] - image_id_shift).reshape(1,-1)

            images = tokenizer.decode_image(image_ids)

            return images[0]


    def test_step(self, batch, batch_idx: int):
        '''
        if batch_idx > 10:
            print(f"Skipping {batch_idx}...")
            return
        '''
        if self.task == 'caption':
            identifier = batch.imgpath

        result = []
        image_tensor = batch.img.cuda()
        img_ids = self.tokenizer.encode_image(image_torch=image_tensor)
        img_ids = img_ids.view(-1).cpu().numpy()
        img_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(item) for item in img_ids]) + EOI_TOKEN

        is_instruction_tuned = True
        if is_instruction_tuned:
            caption_prompt = "Please provide an accurate and concise description of the given image."
            input_tokens = self.tokenizer.bos_token + self.s_token + " " + img_tokens + caption_prompt + self.sep + self.e_token
        else:
            input_tokens = img_tokens

        while True:
            try:
                generate_ids = self.generate(self.tokenizer, input_tokens, self.generation_config, self.model)
                texts = self.decode_image_text(generate_ids, self.tokenizer)
                break
            except:
                print(f"Error in generating text for {identifier}")
                pdb.set_trace()

        B = len(image_tensor)

        if self.task == 'caption':
            captions = batch.gt_txt
            for i in range(B):
                captions_i = captions[i]
                captions_i = [TokenizerUtils.pre_caption(caption) for caption in captions_i]
                dict_ = {
                    "split": "test",
                    "image_name": identifier[i],
                    "captions": captions_i,
                    "prediction": texts
                }
                if self.cfg.dataset.name == 'nocaps':
                    dict_["domain"] = batch.domain[i]
                result.append(dict_)
                self.result.append(dict_)

        return result

    def on_test_epoch_end(self):
        outputs = self.result 
        if self.cfg.dataset.name.startswith('coco'):
            '''
            outputs_ = []
            for output in outputs:
                outputs_.extend(output)
            '''
            # print(outputs_[0])
            fname, ext = os.path.splitext(self.cfg.result_file_path)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            result_file_path = f'{fname}_{self.global_rank}{ext}'
            with open(result_file_path, 'w') as outfile:
                # json.dump(outputs_, outfile)
                json.dump(outputs, outfile)
            print(f'done json file dump at {result_file_path}')

        elif cfg.dataset.name == 'nocaps':
            outputs_all = []
            outputs_in = []
            outputs_near = []
            outputs_out = []
            for output in outputs:
                for out in output:
                    outputs_all.append(out)
                    if out["domain"] == 'in-domain':
                        outputs_in.append(out)
                    elif out["domain"] == 'near-domain':
                        outputs_near.append(out)
                    elif out["domain"] == 'out-domain':
                        outputs_out.append(out)
                    else:
                        raise NotImplementedError
            fname, ext = os.path.splitext(self.cfg.result_file_path)
            os.makedirs(os.path.dirname(fname), exist_ok=True)

            result_file_path = f'{fname}_all_{self.global_rank}{ext}'
            with open(result_file_path, 'w') as outfile:
                json.dump(outputs_all, outfile)
            print(f'done json file dump at {result_file_path}')

            result_file_path = f'{fname}_in_{self.global_rank}{ext}'
            with open(result_file_path, 'w') as outfile:
                json.dump(outputs_in, outfile)
            print(f'done json file dump at {result_file_path}')

            result_file_path = f'{fname}_near_{self.global_rank}{ext}'
            with open(result_file_path, 'w') as outfile:
                json.dump(outputs_near, outfile)
            print(f'done json file dump at {result_file_path}')

            result_file_path = f'{fname}_out_{self.global_rank}{ext}'
            with open(result_file_path, 'w') as outfile:
                json.dump(outputs_out, outfile)
            print(f'done json file dump at {result_file_path}')



if __name__ == "__main__":
    cfg, cfg_yaml = build_config()
    device = "cuda"

    cfg.dataset.name = 'coco2014'
    cfg.dataset.type = 'mapstyle'
    cfg.dataset.gt_text = True
    seed_everything(cfg.experiment.seed, workers=True)

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    tokenizer_cfg = OmegaConf.load(cfg.tokenizer_cfg_path)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg, device=device, load_diffusion=True)

    model_cfg = OmegaConf.load(cfg.model_cfg_path)
    model = hydra.utils.instantiate(model_cfg, torch_dtype=torch.float16)
    model = model.eval()

    datamodule = build_datamodule(
        cfg=cfg,
        train_transform=None,
        val_transform=transform,
        pin_memory=False,
        epoch=0,
        total_gpus=cfg.dist.n_gpus,
    )

    datamodule.setup()

    if cfg.dataset.name.startswith("coco"):
        test_dataloader = datamodule.test_dataloader()
    else:
        raise NotImplementedError

    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
    from pytorch_lightning.strategies import DDPStrategy

    trainer = pl.Trainer(
        accelerator="gpu",
        num_nodes=cfg.dist.n_nodes,
        devices=cfg.dist.n_gpus,
        strategy=DDPStrategy(
            find_unused_parameters=False,
            ddp_comm_hook=default_hooks.fp16_compress_hook
            if cfg.optimizer.fp16_grad_comp
            else None,
        ),
        max_epochs=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        limit_test_batches=1.0,
        deterministic=True
    )


    wrapper = I2TEvalWrapper(cfg, model, tokenizer)
    wrapper.eval()

    trainer.test(wrapper, dataloaders=test_dataloader)
    trainer.strategy.barrier()

    if trainer.global_rank == 0:
        from glob import glob
        head, tail = os.path.splitext(cfg.result_file_path)
        n_files = cfg.dist.n_gpus * cfg.dist.n_nodes

        if cfg.dataset.name.startswith('coco'):
            files = glob(head + "_*" + tail)

            # assert len(files) == n_files

            total_data = []
            for file in files:
                with open(file, 'r') as fin:
                    total_data.extend(json.load(fin))

            print("Number of Generated Samples:", len(total_data))
            with open(cfg.result_file_path, 'w') as fout:
                json.dump(total_data, fout)

            for file in files:
                os.remove(file)

            if wrapper.task == 'caption':
                os.system(f'python ./evaluation/cocoeval.py --result_file_path={cfg.result_file_path}')
            elif wrapper.task == 'vqa':
                os.system(f'python ./evaluation/vqa_eval.py --result_file_path={cfg.result_file_path}')

        elif cfg.dataset.name == 'nocaps':
            domain = ['in', 'near', 'out', 'all']

            for d in domain:
                print(f"----{d}-domain-----")

                files = glob(head + f"_{d}_*" + tail)

                assert len(files) == n_files

                total_data = []
                for file in files:
                    with open(file, 'r') as fin:
                        total_data.extend(json.load(fin))

                print("Number of Generated Samples:", len(total_data))
                result_file_path = head + f"_{d}" + tail
                with open(result_file_path, 'w') as fout:
                    json.dump(total_data, fout)

                for file in files:
                    os.remove(file)

                if wrapper.task == 'caption':
                    os.system(f'python ./evaluation/cocoeval.py --result_file_path={result_file_path}')
                elif wrapper.task == 'vqa':
                    os.system(f'python ./evaluation/vqa_eval.py --result_file_path={result_file_path}')


