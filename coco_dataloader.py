import os
import torch
import json
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader, IterableDataset

import random
import hydra
from PIL import Image
from omegaconf import OmegaConf

transform_cfg_path = 'configs/transform/clip_transform.yaml'
transform_cfg = OmegaConf.load(transform_cfg_path)
transform = hydra.utils.instantiate(transform_cfg)

transform_train = transform
transform_val = transform

class CocoDataset(Dataset):
    def __init__(self, 
                 root_dir, 
                 karpathy_file, 
                 transform=transform_train,
                 tokenizer=None,
                 start_index=None,
                 end_index=None,
                 ):
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(karpathy_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.tokenizer = tokenizer
        with open(karpathy_file, 'r') as f:
            self.karpathy = json.load(f)
        
        self.start_index = start_index
        self.end_index = None if end_index == "None" else end_index

    def __len__(self):
        if self.start_index is not None and self.end_index is not None:
            return len(self.karpathy['images'][self.start_index:self.end_index])
        elif self.start_index is not None:
            return len(self.karpathy['images'][self.start_index:])
        elif self.end_index is not None:
            return len(self.karpathy['images'][:self.end_index])
        return len(self.karpathy['images'])
    
    def tokenize(self, x):
        return torch.tensor(self.tokenizer.encode(x), dtype=torch.int64)[1:]
    
    def __getitem__(self, idx):
        try:
            if self.start_index is not None:
                idx = idx + self.start_index
            img_path = self.karpathy['images'][idx]['file_name']
            if img_path.find('train') != -1:
                train_path = '/data/magvlt2/data/coco/images/train2014'
                img = Image.open(os.path.join(train_path, img_path)).convert('RGB')
            else:
                img = Image.open(os.path.join(self.root_dir, img_path)).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

            # caption
            image_id = self.karpathy['images'][idx]['id']        
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)

            captions = [instance['caption'] for instance in anns]

            captions = captions[random.randint(0, len(captions)-1)]
            return img, captions, img_path #image_id
        except:
            return self.__getitem__(idx+1)
        
    def collate_fn(self, batch):
        images = [_[0] for _ in batch]
        images = torch.stack(images, dim=0)

        txt_tokens = [_[1] for _ in batch]

        image_ids = [_[2] for _ in batch]
        return images, txt_tokens, image_ids
    
    
if __name__ == '__main__':
    karpathy_file = '/ssd0/data/coco/annotations/karpathy/dataset_coco_test.json'
    root_dir = '/ssd0/data/coco/images/val2014'
    dataset = CocoDataset(root_dir, karpathy_file, tokenizer=None)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn, num_workers=32)

    for data_iter_step, batch in enumerate(dataloader):
        images, captions, image_id = batch

        print(images.shape)
        print(captions)
        print(image_id)
        print(image_id[0])
        print(type(image_id[0]))
        break