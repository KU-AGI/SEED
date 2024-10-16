# 导入必要的库和模块
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import argparse
import hydra
from omegaconf import OmegaConf
import os
import lmdb
import pickle
from typing import Optional
import transformers
from dataclasses import dataclass, field
from torchvision import transforms

from torch.multiprocessing import Process, set_start_method, Lock, Pool
import torch.multiprocessing as mp
import pyrootutils
import tqdm
import uuid
import json
import time

import webdataset as wds

# from scripts.seed_llama_inference_14B_coco import tokenizer_cfg

# import multiprocessing

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)


@dataclass
class ConfigPathArguments:
    image_processor: Optional[str] = field(default=None, metadata={"help": "config path of image processor"})
    image_transform: Optional[str] = field(default=None, metadata={"help": "config path of image transform"})
    tokenizer: Optional[str] = field(default=None, metadata={"help": "config path of tokenizer used to initialize tokenizer"})
    data: Optional[str] = field(default=None, metadata={"help": "config path of tokenizer used to initialize tokenizer"})


@dataclass
class ProcessArguments:
    save_dir: Optional[str] = field(
        default=None, metadata={"help": "save dictionary of result which will be written into a sequence of .tar"})
    gpus: Optional[int] = field(default=4, metadata={"help": "number of gpus to be used"})
    batch_size: Optional[int] = field(default=256, metadata={"help": "batch size"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "number of workers to load data per device"})


parser = transformers.HfArgumentParser((ConfigPathArguments, ProcessArguments))
cfg_path, args = parser.parse_args_into_dataclasses()

class DINOBackbone(nn.Module):
    def __init__(self, dinov2):
        super().__init__()
        self.dinov2 = dinov2

    def forward(self, x):
        enc_out = self.dinov2.forward_features(x)
        return rearrange(
            enc_out["x_norm_patchtokens"],
            "b (h w ) c -> b c h w",
            h=int(np.sqrt(enc_out["x_norm_patchtokens"].shape[-2]))
        )

def main():
    print(cfg_path, args)
    os.makedirs(args.save_dir, exist_ok=True)

    # lock = mp.Lock()
    # torch.multiprocessing.spawn(run_worker, args=(lock, ), nprocs=args.gpus, join=True)

    # with Pool(processes=args.gpus) as pool:
    #     pool.map(run_worker, [(i, lock) for i in range(args.gpus)])

    children = []
    for i in range(args.gpus):
        subproc = mp.Process(target=run_worker, args=(i, )) # 각 프로세서에서 실행되는 함수 -> run_worker
        children.append(subproc)
        subproc.start()

    for i in range(args.gpus):
        children[i].join()


# 定义worker函数
def run_worker(gpu):
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:6668', world_size=args.gpus, rank=gpu)
    torch.cuda.set_device(gpu)

    sub_save_dir = os.path.join(args.save_dir, 'part-{:04d}'.format(gpu))
    os.makedirs(sub_save_dir, exist_ok=True)

    save_pattern = sub_save_dir + "/%07d.tar"
    if cfg_path.image_processor is not None:
        processor_cfg = OmegaConf.load(cfg_path.image_processor)
        processor = hydra.utils.instantiate(processor_cfg)
    else:
        processor = None

    transform = transforms.Compose([
        transforms.Resize((448,448), antialias=True),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
    ])

    normalize_vit = transforms.Normalize(mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)),
    normalize_diffusion = transforms.Normalize(mean=[0.5], std=[0.5])

    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    backbone = DINOBackbone(dinov2).eval().cuda()
    tokenizer = None

    data_cfg = OmegaConf.load(cfg_path.data)
    dataset = hydra.utils.instantiate(data_cfg, tokenizer=tokenizer, image_processor=processor, image_transform=transform)

    print('Init Done')

    # 初始化DistributedSampler和DataLoader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    print(dataset)
    # 在每个GPU设备上运行模型
    with wds.ShardWriter(save_pattern, maxcount=10000) as sink:
        with torch.no_grad():
            time1 = time.time()
            for batch in tqdm.tqdm(data_loader):
                time2 = time.time()
                if gpu == 0:
                    print('time: ', time2 - time1)
                time1 = time2
                image_tensor = batch['pixel_values'].cuda()
                texts = batch['text']
                metadatas = batch['metadata']
                # key_strs = batch['__key__']
                # print(image_tensor.shape)
                # image_ids = tokenizer.encode_image(image_torch=image_tensor, compress_rate=args.compress_rate)
                # image_ids = tokenizer.encode_image(image_torch=image_tensor)
                features = backbone(image_tensor)

                # Convert to numpy npz
                features = features.cpu().numpy()

                # print(image_ids.shape)

                # # mmc4
                # key_str = uuid.uuid4().hex
                # sample = {'image_ids': image_ids.cpu().tolist(), 'text': texts, 'metadata': metadatas}
                # sink.write({'__key__': key_str, 'pkl': pickle.dumps(sample)})
                
                # pair dataset
                for feature, metadata, text in zip(features, metadatas, texts):
                    key_str = uuid.uuid4().hex
                    sample = {'feature': feature, 'text': text, 'metadata': json.loads(metadata)}
                    # sink.write({'__key__': key_str, 'json': sample})
                    sink.write({'__key__': key_str, 'pkl': pickle.dumps(sample)})



if __name__ == '__main__':
    # with multiprocessing.Pool(args.gpus) as pool:
    #     pool.map(run_worker, range(args.gpus))

    # set_start_method('spawn')
    main()

# python3 src/tools/extract_image_ids_to_torchdata_parallel.py --tokenizer configs/tokenizer/seed_llama_tokenizer.yaml --image_transform configs/processer/blip_transform.yaml --data configs/data/caption_torchdata_preprocess.yaml --save_dir dataset/seed_llama/caption/unsplash_cc3m/ --batch_size 1024 --num_workers 8 --gpus 8
