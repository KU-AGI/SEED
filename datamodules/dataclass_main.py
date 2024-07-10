import torch
import os
import json
import hydra
import argparse
import time
import random

from PIL import Image
import base64
from io import BytesIO

from tqdm import tqdm
import webdataset as wds
from omegaconf import OmegaConf
from utils.config import build_config

from torch.utils.data import Dataset, DataLoader, IterableDataset
from datamodules.data_utils import expand_urls
from pycocotools.coco import COCO


""" Single batch should be like this: (task, data)"""


# task_type = 0 or task_type = 4 (T2I, I2T Generation)
class Finetune_Pair(IterableDataset):
    def __init__(self, config, transform=None, tokenizer=None, world_size=1):
        super().__init__() 
        
        # 1. Read dataset config
        print(f"\nRead dataset config from config file...")
        self.config = config
        self.task_type = self.config.task_type
        
        # 2. Read dataset from .tar files
        self.wds_urls = [] 
        self.data_size = self.config.data_size                                          # It is not that important, because we use IterableDataset.
        
        urls = expand_urls(str(self.config.data_dir))
        self.wds_urls = [url for url in urls]                                           # url example: "00000_000000.tar"             
        print(f"Using {self.data_size} samples.\n")


        # 3. Set other variables
        self.transform = transform
        self.tokenizer = tokenizer
        self.world_size = world_size
        
        # self.max_length = self.config.max_length                                        # max_length for same tokenized length (text1, text2, ... textn)
        # self.reverse_ratio = self.config.reverse_ratio                                  # TODO
        self.shardshuffle = self.config.shardshuffle
        self.resampled = self.config.resampled
        self.bsz = self.config.batch_size                                               # TODO
        
        
        # 4. Make dataset
        self.dataset = (
            wds.WebDataset(
                self.wds_urls,                                                          # num of files (10K sample)
                nodesplitter=wds.split_by_node,                                         # multi-node
                shardshuffle=self.shardshuffle,                                         # shard shuffle (each .tar file)
                detshuffle=False,                                                       # deterministic shuffle
                resampled=self.resampled,                                               # resampled for training desired epochs
                handler=wds.ignore_and_continue,
            )
            .shuffle(  
                size=(1 if self.shardshuffle is None else self.shardshuffle * 10),      # size of shuffle buffer
                initial=(0 if self.shardshuffle is None else 100),                      # when to start shuffling
            )
            .decode("pil", handler=wds.ignore_and_continue)
            .to_tuple("jpg", "txt", "json", handler=wds.ignore_and_continue) 
            .map_tuple(
                self.transform,
                self.tokenize if self.tokenizer is not None else self.identity,
                self.identity,
                handler=wds.ignore_and_continue,
                )
            .with_length(int(self.data_size / world_size))                        # length of dataset per device(GPU)
        )
    
    def __len__(self):
        # TODO: Change to config
        return 1000000 

    def identity(self, x):
        return x

    def tokenize(self, x):
        token_tensor = torch.tensor(self.tokenizer.encode(x), dtype=torch.int64)[1:]
        return token_tensor
       
    # 선 실행 / single sample generate
    def __iter__(self):
        for image, text, _ in self.dataset:
            yield self.task_type, image, text # int, pil, tensor
                    
    # 후 실행 / single batch generate
    def collate_fn(self, batch):
        task_type, img, txt_token = zip(*batch)
        # IMG is not tokenizing yet. It will be tokenized in the model.
        return list(task_type), list(img), list(txt_token)

    def bsz_fn(self):
        return self.bsz
    

# task_type = 1 (MultiModal Prompt Image Generation; Editing)
class Finetune_Edit(IterableDataset):
    def __init__(self, config, transform=None, tokenizer=None, world_size=1):
        super().__init__() 
        
        # 1. Read dataset config
        print(f"\nRead dataset config from config file...")
        self.config = config
        self.task_type = self.config.task_type
        
        # 2. Read dataset from .tar files
        self.wds_urls = [] 
        self.data_size = self.config.data_size                                          # It is not that important, because we use IterableDataset.
        
        urls = expand_urls(str(self.config.data_dir))
        self.wds_urls = [url for url in urls]                                           # url example: "00000_000000.tar"             
        print(f"Using {self.data_size} samples.\n")


        # 3. Set other variables
        self.transform = transform
        self.tokenizer = tokenizer
        self.world_size = world_size
        
        # self.max_length = self.config.max_length                                        # max_length for same tokenized length (text1, text2, ... textn)
        # self.reverse_ratio = self.config.reverse_ratio                                  
        self.shardshuffle = self.config.shardshuffle
        self.resampled = self.config.resampled
        self.bsz = self.config.batch_size                                               
        
        
        # 4. Make dataset
        self.dataset = (
            wds.WebDataset(
                self.wds_urls,                                                          # num of files (10K sample)
                nodesplitter=wds.split_by_node,                                         # multi-node
                shardshuffle=self.shardshuffle,                                         # shard shuffle (each .tar file)
                detshuffle=False,                                                       # deterministic shuffle
                resampled=self.resampled,                                               # resampled for training desired epochs
                handler=wds.ignore_and_continue,
            )
            .shuffle(  
                size=(1 if self.shardshuffle is None else self.shardshuffle * 10),      # size of shuffle buffer
                initial=(0 if self.shardshuffle is None else 100),                      # when to start shuffling
            )
            .decode("pil", handler=wds.ignore_and_continue)
            .to_tuple("json", handler=wds.ignore_and_continue) 
            .map_tuple(
                self.identity,
                handler=wds.ignore_and_continue,
                )
            .with_length(int(self.data_size / world_size))                        # length of dataset per device(GPU)
        )
    

    def identity(self, x):
        return x


    def tokenize(self, x):
        token_tensor = torch.tensor(self.tokenizer.encode(x), dtype=torch.int64)[1:]
        return token_tensor
       
       
    # 선 실행 / single sample generate
    def __iter__(self):
        for sample in self.dataset:
            sample = sample[0]              # sample: tuple (len = 1)
            
            # multi images (2 images)
            images = [self.transform(Image.open(BytesIO(base64.b64decode(img)))) for img in sample['image']]  # list
            # single text
            text = sample['text']
            text = self.tokenize(text)      # tensor
            
            yield self.task_type, images, text # list of pil, tensor
            
            
    # 후 실행 / single batch generate
    def collate_fn(self, batch):
        task_type, img, txt_token = zip(*batch)
        return list(task_type), list(img), list(txt_token)


    def bsz_fn(self):
        return self.bsz
    

# task_type = 2 (Image Conversation)
class Finetune_Conversation(IterableDataset):
    def __init__(self, config, transform=None, tokenizer=None, world_size=1):
        super().__init__() 
        
        # 1. Read dataset config
        print(f"\nRead dataset config from config file...")
        self.config = config
        self.task_type = self.config.task_type
        

        # 2. Read dataset from .tar files
        self.wds_urls = [] 
        self.data_size = self.config.data_size                                          # It is not that important, because we use IterableDataset.
        
        urls = expand_urls(str(self.config.data_dir))
        self.wds_urls = [url for url in urls]                                           # url example: "00000_000000.tar"             
        print(f"Using {self.data_size} samples.\n")


        # 3. Set other variables
        self.transform = transform
        self.tokenizer = tokenizer
        self.world_size = world_size
        
        # self.max_length = self.config.max_length                                        # max_length for same tokenized length (text1, text2, ... textn)
        # self.reverse_ratio = self.config.reverse_ratio                                  
        self.shardshuffle = self.config.shardshuffle
        self.resampled = self.config.resampled
        self.bsz = self.config.batch_size                                               
       
        
        # 4. Make dataset
        self.dataset = (
            wds.WebDataset(
                self.wds_urls,                                                          # num of files (10K sample)
                nodesplitter=wds.split_by_node,                                         # multi-node
                shardshuffle=self.shardshuffle,                                         # shard shuffle (each .tar file)
                detshuffle=False,                                                       # deterministic shuffle
                resampled=self.resampled,                                               # resampled for training desired epochs
                handler=wds.ignore_and_continue,
            )
            .shuffle(  
                size=(1 if self.shardshuffle is None else self.shardshuffle * 10),      # size of shuffle buffer
                initial=(0 if self.shardshuffle is None else 100),                      # when to start shuffling
            )
            .decode("pil", handler=wds.ignore_and_continue)
            .to_tuple("jpg", "txt", handler=wds.ignore_and_continue) 
            .map_tuple(
                self.transform,
                self.process,
                handler=wds.ignore_and_continue,
                )
            .with_length(int(self.data_size / world_size))                        # length of dataset per device(GPU)
        )
    
    def tokenize(self, x):
        token_tensor = torch.tensor(self.tokenizer.encode(x), dtype=torch.int64)[1:]
        return token_tensor
       
    def process(self, x):
        texts = x.split("###")
        texts_tokenized = [self.tokenize(text) for text in texts] # list
        return texts_tokenized
       
    # 선 실행 / single sample generate
    def __iter__(self):
        for image, texts in self.dataset:
            yield self.task_type, image, texts # pil, list of tensor
            
    # 후 실행 / single batch generate
    def collate_fn(self, batch):
        task_type, img, txt_token = zip(*batch)
        return list(task_type), list(img), list(txt_token)
    
    def bsz_fn(self):
        return self.bsz
  
  
# task_type = 3 (Multi-image Understanding)
class Finetune_Understand(IterableDataset):
    def __init__(self, config, transform=None, tokenizer=None, world_size=1):
        super().__init__() 
        
        # 1. Read dataset config
        print(f"\nRead dataset config from config file...")
        self.config = config
        self.task_type = self.config.task_type
        
        
        # 2. Read dataset from .tar files
        self.wds_urls = [] 
        self.data_size = self.config.data_size                                          # It is not that important, because we use IterableDataset.
        
        urls = expand_urls(str(self.config.data_dir))
        self.wds_urls = [url for url in urls]                                           # url example: "00000_000000.tar"
        print(f"Using {self.data_size} samples.\n")


        # 3. Set other variables
        self.transform = transform
        self.tokenizer = tokenizer
        self.world_size = world_size
        
        # self.max_length = self.config.max_length                                        # max_length for same tokenized length (text1, text2, ... textn)
        # self.reverse_ratio = self.config.reverse_ratio                                  
        self.shardshuffle = self.config.shardshuffle
        self.resampled = self.config.resampled
        self.bsz = self.config.batch_size                                               
       
        
        # 4. Make dataset
        self.dataset = (
            wds.WebDataset(
                self.wds_urls,                                                          # num of files (10K sample)
                nodesplitter=wds.split_by_node,                                         # multi-node
                shardshuffle=self.shardshuffle,                                         # shard shuffle (each .tar file)
                detshuffle=False,                                                       # deterministic shuffle
                resampled=self.resampled,                                               # resampled for training desired epochs
                handler=wds.ignore_and_continue,
            )
            .shuffle(  
                size=(1 if self.shardshuffle is None else self.shardshuffle * 10),      # size of shuffle buffer
                initial=(0 if self.shardshuffle is None else 100),                      # when to start shuffling
            )
            .decode("pil", handler=wds.ignore_and_continue)
            .to_tuple("json", handler=wds.ignore_and_continue) 
            .map_tuple(
                self.identity,
                handler=wds.ignore_and_continue,
                )
            .with_length(int(self.data_size / world_size))                        # length of dataset per device(GPU)
        )
    

    def identity(self, x):
        return x


    def tokenize(self, x):
        token_tensor = torch.tensor(self.tokenizer.encode(x), dtype=torch.int64)[1:]
        return token_tensor
       
       
    # 선 실행 / single sample generate
    def __iter__(self):
        for sample in self.dataset:
            sample = sample[0]              # sample: tuple (len = 1)
            
            # multi images (2 images)
            images = [self.transform(Image.open(BytesIO(base64.b64decode(img)))) for img in sample['image']] # list
            # multi texts (2 texts)
            texts = [self.tokenize(txt) for txt in sample['text']] # list
            
            yield self.task_type, images, texts
            
            
    # 후 실행 / single batch generate
    def collate_fn(self, batch):
        task_type, img, txt_token = zip(*batch)
        return list(task_type), list(img), list(txt_token)
    
    def bsz_fn(self):
        return self.bsz
    

# task_type = 5 (Image QA))
class Finetune_VQA(IterableDataset):
    def __init__(self, config, transform=None, tokenizer=None, world_size=1):
        super().__init__() 
        
        # 1. Read dataset config
        print(f"\nRead dataset config from config file...")
        self.config = config
        self.task_type = self.config.task_type
        
        
        # 2. Read dataset from .tar files
        self.wds_urls = [] 
        self.data_size = self.config.data_size                                          # It is not that important, because we use IterableDataset.
        
        urls = expand_urls(str(self.config.data_dir))
        self.wds_urls = [url for url in urls]                                           # url example: "00000_000000.tar"             
        print(f"Using {self.data_size} samples.\n")


        # 3. Set other variables
        self.transform = transform
        self.tokenizer = tokenizer
        self.world_size = world_size
        
        # self.max_length = self.config.max_length                                        # max_length for same tokenized length (text1, text2, ... textn)
        # self.reverse_ratio = self.config.reverse_ratio                                  
        self.shardshuffle = self.config.shardshuffle
        self.resampled = self.config.resampled
        self.bsz = self.config.batch_size                                               
       
        
        # 4. Make dataset
        self.dataset = (
            wds.WebDataset(
                self.wds_urls,                                                          # num of files (10K sample)
                nodesplitter=wds.split_by_node,                                         # multi-node
                shardshuffle=self.shardshuffle,                                         # shard shuffle (each .tar file)
                detshuffle=False,                                                       # deterministic shuffle
                resampled=self.resampled,                                               # resampled for training desired epochs
                handler=wds.ignore_and_continue,
            )
            .shuffle(  
                size=(1 if self.shardshuffle is None else self.shardshuffle * 10),      # size of shuffle buffer
                initial=(0 if self.shardshuffle is None else 100),                      # when to start shuffling
            )
            .decode("pil", handler=wds.ignore_and_continue)
            .to_tuple("jpg", "txt", handler=wds.ignore_and_continue) 
            .map_tuple(
                self.transform,
                self.process,
                handler=wds.ignore_and_continue,
                )
            .with_length(int(self.data_size / world_size))                        # length of dataset per device(GPU)
        )
    

    def identity(self, x):
        return x


    def tokenize(self, x):
        token_tensor = torch.tensor(self.tokenizer.encode(x), dtype=torch.int64)[1:]
        return token_tensor
       
       
    def process(self, x):
        output = [self.tokenize(txt.strip()) for txt in x.split("###")]
        return output
        
       
    # 선 실행 / single sample generate
    def __iter__(self):
        for image, texts in self.dataset:
            yield self.task_type, image, texts
            
            
    # 후 실행 / single batch generate
    def collate_fn(self, batch):
        task_type, img, txt_token = zip(*batch)
        return list(task_type), list(img), list(txt_token)


    def bsz_fn(self):
        return self.bsz



# task_type = -1 (pretraining - interleaved)
class Finetune_Interleaved(IterableDataset):
    def __init__(self, config, transform=None, tokenizer=None, world_size=1):
        super().__init__()
        
        # 1. Read dataset config
        print(f"\nRead dataset config from config file...")
        self.config = config
        self.task_type = self.config.task_type
        
        # 2. Read dataset from .tar files
        self.wds_urls = [] 
        self.data_size = self.config.data_size                                          # It is not that important, because we use IterableDataset.
        
        urls = expand_urls(str(self.config.data_dir))
        self.wds_urls = [url for url in urls if os.path.isfile(url)]                    # url example: "00000_000000.tar"             
        print(f"Using {self.data_size} samples.\n")


        # 3. Set other variables
        self.transform = transform
        self.tokenizer = tokenizer
        self.world_size = world_size
        
        # self.reverse_ratio = self.config.reverse_ratio                                  
        self.shardshuffle = self.config.shardshuffle
        self.resampled = self.config.resampled
        self.bsz = self.config.batch_size                                               
       
        
        # 4. Make dataset
        self.dataset = (
            wds.WebDataset(
                self.wds_urls,                                                          # num of files (10K sample)
                nodesplitter=wds.split_by_node,                                         # multi-node
                shardshuffle=self.shardshuffle,                                         # shard shuffle (each .tar file)
                detshuffle=False,                                                       # deterministic shuffle
                resampled=self.resampled,                                               # resampled for training desired epochs
                handler=wds.ignore_and_continue,
            )
            .shuffle(  
                size=(1 if self.shardshuffle is None else self.shardshuffle * 10),      # size of shuffle buffer
                initial=(0 if self.shardshuffle is None else 100),                      # when to start shuffling
            )
            .decode("pil", handler=wds.ignore_and_continue)
            .to_tuple("json", handler=wds.ignore_and_continue) 
            .map_tuple(
                self.identity,
                handler=wds.ignore_and_continue,
                )
            .with_length(int(self.data_size / world_size))                        # length of dataset per device(GPU)
        )
    

    def identity(self, x):
        return x

    def tokenize(self, x):
        token_tensor = torch.tensor(self.tokenizer.encode(x), dtype=torch.int64)[1:]
        return token_tensor
       
       
    # 선 실행 / single sample generate
    def __iter__(self):
        for sample in self.dataset:
            sample = sample[0]              # sample: tuple (len = 1)
            
            text_tokens = [self.tokenize(txt) for txt in sample['text_list']]
            images = [self.transform(Image.open(BytesIO(base64.b64decode(img_info['image_base64'])))) for img_info in sample['image_info']]
            
            matched_text_index = set([img_info['matched_text_index'] for img_info in sample['image_info']])
            captionid2imageid = {}
            for i, img_info in enumerate(sample['image_info']):
                captionid2imageid[img_info['matched_text_index']] = i
            
            yield text_tokens, matched_text_index, captionid2imageid, images
            
               
    # 후 실행 / single batch generate
    def collate_fn(self, batch):
        text_tokens, matched_text_index, captionid2imageid, images = zip(*batch)
        return list(text_tokens), list(matched_text_index), list(captionid2imageid), list(images)

    
    def bsz_fn(self):
        return self.bsz


# for validation/test
class CocoDataset(Dataset):
    # Args are different from the above datasets.
    def __init__(self, root_dir, karpathy_file, transform=None, tokenizer=None, start_idx=None, end_idx=None):
        super().__init__() 
        
        self.root_dir = root_dir
        self.karpathy = json.load(open(karpathy_file, 'r'))
        self.coco = COCO(karpathy_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # TODO
        transform_cfg_path = 'configs/transform/clip_transform.yaml'
        transform_cfg = OmegaConf.load(transform_cfg_path)
        transform = hydra.utils.instantiate(transform_cfg)
        
        self.transform = transform
        self.tokenizer = tokenizer
        
        self.start_idx = start_idx
        self.end_idx = end_idx


    def tokenize(self, x):
        return torch.tensor(self.tokenizer.encode(x), dtype=torch.int64)[1:] # remove bos token
    

    def __len__(self):
        if self.start_idx is not None and self.end_idx is not None:
            return len(self.karpathy['images'][self.start_idx:self.end_idx])
        elif self.start_idx is not None:
            return len(self.karpathy['images'][self.start_idx:])
        elif self.end_idx is not None:
            return len(self.karpathy['images'][:self.end_idx])
        # both None
        return len(self.karpathy['images'])
    
    
    def __getitem__(self, idx):
        try:
            if self.start_idx is not None:
                idx = idx + self.start_idx
                
            # set image path
            img_path = self.karpathy['images'][idx]['file_name']
            # train set
            if img_path.find('train') != -1:
                train_path = '/data/magvlt2/data/coco/images/train2014' # TODO (AICA 기준)
                img = Image.open(os.path.join(train_path, img_path)).convert('RGB')
            # val/test set
            else:
                img = Image.open(os.path.join(self.root_dir, img_path)).convert('RGB')
                
            if self.transform is not None:
                img = self.transform(img)

            # caption
            image_id = self.karpathy['images'][idx]['id']        
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)

            txt_token = [self.tokenize(instance['caption']) for instance in anns]
            text_token = txt_token[random.randint(0, len(txt_token)-1)]
            
            return img, text_token, img_path # img path = image_id
        
        except:
            return self.__getitem__(idx+1) # recursive call 주의
    
    def collate_fn(self, batch):
        
        images = [_[0] for _ in batch]
        images = torch.stack(images, dim=0)
        txt_tokens = [_[1] for _ in batch]
        image_ids = [_[2] for _ in batch]
        
        return images, txt_tokens, image_ids
        # images, txt_tokens, image_ids = zip(*batch)
        # Stack images into a single tensor
        # images = torch.stack(images, dim=0) # TODO: 위 데이터셋 클래스와 전달 방식 통일
        # return images, list(txt_tokens), list(image_ids)



# for iterableDataset's dataLoader
class WeightedDataLoaderSampler:
    def __init__(self, dataloaders, data_weights=None):
        self.dataloaders = dataloaders
        
        if data_weights is None:
            self.data_weights = [1.0] * len(dataloaders)
        else:
            # print(data_weights)
            assert len(dataloaders) == len(data_weights), "data_weights should have the same length as dataloaders!"
            # normalize
            total_weight = sum(data_weights)
            self.data_weights  = [float(w) / total_weight for w in data_weights]
        self.iterators = [iter(dl) for dl in dataloaders]

    def __iter__(self):
        while len(self.iterators) > 0:
            r = random.random()  # [0, 1)
            s = 0.0

            for i, (loader, weight) in enumerate(zip(self.iterators, self.data_weights)):
                s += weight
                if r < s:
                    try:
                        item = next(loader)
                        yield item
                    except StopIteration:
                        # Remove the exhausted loader
                        self.iterators.pop(i)
                        self.data_weights.pop(i)
                        new_total = sum(self.data_weights)
                        if new_total > 0:
                            self.data_weights = [w / new_total for w in self.data_weights]
                    break

            # If only one loader left, simply return items from it
            if len(self.iterators) == 1:
                try:
                    while True:
                        yield next(self.iterators[0])
                    # yield next(self.iterators[0])
                except StopIteration:
                    break  # In case the last loader is also exhausted

        # If no loaders left, raise StopIteration
        raise StopIteration
    
    
# v2
class WeightedDataLoaderSampler_2:
    def __init__(self, dataloaders, data_weights=None):
        self.dataloaders = dataloaders
        
        if data_weights is None:
            self.data_weights = [1.0] * len(dataloaders)
        else:
            # print(data_weights)
            assert len(dataloaders) == len(data_weights), "data_weights should have the same length as dataloaders!"
            # normalize
            total_weight = sum(data_weights)
            self.data_weights  = [float(w) / total_weight for w in data_weights]
        self.iterators = [iter(dl) for dl in dataloaders]
        self.random_generator = random.Random()


    # def __iter__(self):
    #     loader = self.random_generator.choices(self.iterators, self.data_weights, k=1)[0]
    #     yield next(loader)
        
    def __iter__(self):
        while self.iterators:
            try:
                loader = self.random_generator.choices(self.iterators, self.data_weights, k=1)[0]
                yield next(loader)
            except StopIteration:
                # Remove the exhausted iterator
                self.iterators.remove(loader)
                if not self.iterators:
                    break
        


""" 디버깅 """    
# Test/Debug Code
if __name__ == '__main__':
    
    config_path = "/home/ubuntu/MAGVLT2/MultiModalLLM/configs/data/dataloader_debug.yaml"

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--bsz', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    
    # configs
    print(f"\nRead dataset configs from {config_path}...")
    dataset_config, _ = build_config(cfg_path=config_path)
    
    tokenizer = hydra.utils.instantiate(OmegaConf.load(dataset_config.tokenizer_cfg_path), load_diffusion=True) 
    transform = hydra.utils.instantiate(OmegaConf.load(dataset_config.transform_cfg_path))
    world_size = dataset_config.world_size
    
    start_time = time.time()
    
    # make datsets
    train_dataset = []
    i = 0

    while True:
        try:
            data_config = dataset_config[f"data_{i}"]
            
            if data_config.task_type == 0 or data_config.task_type == 4:
                train_dataset.append(Finetune_Pair(config=data_config, 
                                                    transform=transform,
                                                    tokenizer=tokenizer,
                                                    world_size=world_size))
            elif data_config.task_type == 1: 
                train_dataset.append(Finetune_Edit(config=data_config, 
                                                    transform=transform,
                                                    tokenizer=tokenizer,
                                                    world_size=world_size))
            elif data_config.task_type == 2:
                train_dataset.append(Finetune_Conversation(config=data_config, 
                                                    transform=transform,
                                                    tokenizer=tokenizer,
                                                    world_size=world_size))
            elif data_config.task_type == 3:
                train_dataset.append(Finetune_Understand(config=data_config, 
                                                    transform=transform,
                                                    tokenizer=tokenizer,
                                                    world_size=world_size))
            else:
                train_dataset.append(Finetune_VQA(config=data_config, 
                                                    transform=transform,
                                                    tokenizer=tokenizer,
                                                    world_size=world_size))
            
            # add idx
            i += 1
            
        # end of dataset
        except KeyError:
            break
    
    
    dataset_processing_time = time.time() - start_time
    
    if len(train_dataset) == 1:
        train_dataloader = DataLoader(train_dataset[0], batch_size=train_dataset[0].bsz_fn(), collate_fn=train_dataset[0].collate_fn, num_workers=args.num_workers)
    else:
        print(f"Provided {len(train_dataset)} datasets!")
        data_weights = [0.5, 0.5]
        _loaders = []
        
        for i in range(len(train_dataset)):
            _loaders.append(DataLoader(train_dataset[i], batch_size=train_dataset[i].bsz_fn(), collate_fn=train_dataset[i].collate_fn, num_workers=args.num_workers))
        print("_loaders is as follows: ", _loaders)
        
        train_dataloader = WeightedDataLoaderSampler(dataloaders=_loaders, data_weights=data_weights)        
        
    dataloader_processing_time = time.time() - start_time - dataset_processing_time
    print(f"dataset_processing_time: {dataset_processing_time:.4f} sec / dataloader_processing_time: {dataloader_processing_time:.4f} sec\n")
    
    
    # ========================= Debug =========================
    
    start_iter_time = time.time()
    
    for batch_idx, batch in enumerate(train_dataloader):
        
        if batch_idx < 10:
            print("### DEBUG ###")
            # print(type(batch))          # <class 'list'>
            # print(len(batch))           # 3 (=return of collate_fn)  
            # print(len(batch[0]))        # 8 (batch size)
            print(batch[0])
            
            # for i in range(len(batch)):
            #     print(f"{i}th")
            #     print(batch[i].shape)
            #     if i ==2:
            #         print(batch[i])

        else:       
            break   

    end_iter_time = time.time() - start_iter_time
    
    print(f"iteration_time: {end_iter_time:.4f} sec\n")
    print ('Done!')
     
        
# CUDA_VISIBLE_DEVICES=0 python dataclass_main.py --bsz 8 --num_workers 4