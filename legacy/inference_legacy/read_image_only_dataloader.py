import torch
import yaml
from torch.utils.data import Dataset, DataLoader, IterableDataset
from PIL import Image
import json
#from model.tokenizer import Tokenizer
import copy
import torchvision.transforms as transforms
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import webdataset as wds
import braceexpand
from collections.abc import Callable
from pycocotools.coco import COCO

import hydra
from omegaconf import OmegaConf


transform_cfg_path = 'configs/transform/clip_transform.yaml'
transform_cfg = OmegaConf.load(transform_cfg_path)
transform = hydra.utils.instantiate(transform_cfg)

transform_train = transform
transform_val = transform
    
class FinetuneData(IterableDataset):
    def __init__(self,
                 config_path,
                 transform=transform_train,
                 max_words=30,
                 tokenizer=None,
                 shardshuffle=100,
                 resampled=True,
                 world_size=1,
                 rank=0
                 ):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        wds_urls = []
        self.total_num_samples = 0
        for urls, num_samples in self.config['META']:
            urls = expand_urls(urls)
            wds_urls.extend(urls)
            self.total_num_samples += num_samples

        self.transform = transform
        self.max_words = max_words
        self.tokenizer = tokenizer

        self.dataset = (
            wds.WebDataset(
                urls,
                # 싱글 노드 방식: wds.single_node_only (default)
                # 멀티 노드 방식: wds.split_by_node
                # url level partioning
                nodesplitter=wds.split_by_node,
                # url level shuffle
                shardshuffle=shardshuffle,
                # deterministic shuffle (재현성)
                detshuffle=False,
                # infinite url
                resampled=resampled,
                handler=wds.ignore_and_continue,
            )
            .shuffle(  # sample level shuffle
                size=(1 if shardshuffle is None else shardshuffle * 10),
                initial=(0 if shardshuffle is None else 100),
            )
            .decode("pil", handler=wds.ignore_and_continue)
            .to_tuple("jpg", "txt", "json", handler=wds.ignore_and_continue)
            .map_tuple(
                transform,
                self.identity, #self.tokenize,
                self.identity,
                handler=wds.ignore_and_continue,
            )
            .with_length(int(int(self.total_num_samples) / world_size))
        )

        self.world_size = world_size
        self.rank = rank

        self.name = config_path

    def identity(self, x):
        return x

    def tokenize(self, x):
        return torch.tensor(self.tokenizer.encode(x), dtype=torch.int64)[1:]

    def __len__(self):
        return len(self.dataset)
    
    def process_text(self, text):
        text_tokens = self.tokenize(text)
        return text_tokens

    def __iter__(self):
        for i, (img, txt_tokens, meta) in enumerate(self.dataset):
            yield img, txt_tokens
    
    def groups(self):
        return list(self.group_indices.values())


def expand_urls(urls):
    def decode(urls):
        urllist = urls.split("::")
        result = []
        for url in urllist:
            result.extend(braceexpand.braceexpand(url))
        return result

    if isinstance(urls, str):
        return decode(urls)
    elif isinstance(urls, tuple):
        results = []
        for urls_ in urls:
            results += decode(urls_)
        return results
    else:
        return list(urls)
    
import random
class ComibinedDatasetIterator:
    def __init__(self, datasets, weights):
        self.datasets = [iter(dataset) for dataset in datasets]
        self.weights = weights
        self.randome_generator = random.Random()

    def __next__(self):
        (dataset, ) = self.randome_generator.choices(self.datasets, self.weights, k=1)
        return next(dataset)
    
class CombinedDataset(IterableDataset):
    def __init__(self, datasets_configs, datasets, rank=None, world_size=None, weights=None, length=None):
        self.datasets = datasets
        
        weights = weights if weights is not None else [1] * len(datasets)
        self.weights = [w/sum(weights) for w in weights]
        self.randome_generator = random.Random()
        if length is None:
            self.length = sum([len(dataset) for dataset in datasets])
        else:
            self.length = length
    
    def __iter__(self):
        return ComibinedDatasetIterator(self.datasets, self.weights) 

    def __len__(self):
        return self.length
    
    
def custom_collate_fn(batch):
    images = [_[0] for _ in batch]
    images = torch.stack(images, dim=0)
    
    texts = [_[1] for _ in batch]
    return images, texts


reverse_transform = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[1/0.26862954, 1/0.26130258, 1/0.27577711]),
        transforms.Normalize(mean=[-0.48145466, -0.4578275, -0.40821073], std=[1, 1, 1]),
    ])

max_seq_len = 1024


def pack(token_sequences, max_seq_len=128, batch_size=3):
    ## sort token_sequences by length
    token_sequences = sorted(token_sequences, key=lambda x: len(x), reverse=True)
    print(token_sequences)

    ## batch by snaek order
    packed_ds = []
    curr_token_ids = [torch.tensor([], dtype=torch.int64) for _ in range(batch_size)]

    for i in range(0, len(token_sequences), batch_size):
        for j in range(batch_size):
            if i+j >= len(token_sequences):
                break
            if len(curr_token_ids[j]) + len(token_sequences[i+j]) < max_seq_len:
                curr_token_ids[j] = torch.cat([curr_token_ids[j], token_sequences[i+j]], dim=0)
    for j in range(batch_size):
        packed_ds.append(curr_token_ids[j])

    return packed_ds



### test code
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True)
    args = parser.parse_args()
    
    
     
    from tqdm import tqdm

    ## combined dataset test code
    dataset_configs = ['configs/data/cc15m.yaml', 'configs/data/laion-coco.yaml', 'configs/data/mscoco.yaml']
    datasets = []
    for config in dataset_configs:
        dataset = FinetuneData(config, 
                                transform=transform_train,
                                shardshuffle=100,
                                resampled=True,
                                world_size=1,)
        datasets.append(dataset)

    
    combined_dataset = CombinedDataset(datasets_configs=dataset_configs, 
                                       datasets=datasets,
                                       rank=0, 
                                       world_size=1,
                                       weights=[1, 1, 1])
        
    dataloader = DataLoader(combined_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn, num_workers=2)

    for data_iter_step, batch in enumerate(tqdm(dataloader)):
        images, texts = batch
        print(images)
        print(images.shape)
        print(texts)
  

        if data_iter_step >= 0:
            break
        #print(images.shape)
