import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import ConcatDataset, random_split
from torch.utils.data import Sampler, DistributedSampler
from io import BytesIO
from PIL import Image

import random
import os
import json
import tarfile

class CompressionDataset(Dataset):
    def __init__(self, compression_level, transform=None):
        self.root = '/home/zheedong/Projects/SEED/data/cc3m_llava_long_caption'

        self.annotation_path = f"{self.root}/annotations/{compression_level}/captions.json"

        self.image_root = f"{self.root}/images"

        with open(self.annotation_path, encoding='utf-8') as f:
            self.annotation_data = json.load(f)
        
        self.compression_level = compression_level
        self.transform = transform

    def __len__(self):
        return len(self.annotation_data)

    def __getitem__(self, idx):
        """
        Return image caption pair

        Args:
            idx (int): image-text pair index

        Returns:
            image, list[str]: image and list of captions
        """
            
        image = Image.open(os.path.join(self.image_root, self.annotation_data[idx]['image']))
        caption = self.annotation_data[idx]["caption"]
            
        if self.transform:
            image = image.convert('RGB')
            image = self.transform(image)

        return image, caption, self.compression_level

class CompressionLevelSampler(Sampler):
    def __init__(self, concat_dataset, batch_size, drop_last=True):
        self.concat_dataset = concat_dataset
        self.batch_size = batch_size
        # Calculate the boundaries of each dataset within the ConcatDataset
        self.drop_last = drop_last  # Define drop_last here
        self.dataset_boundaries = self._calculate_dataset_boundaries()

    def _calculate_dataset_boundaries(self):
        boundaries = []
        total = 0
        for dataset in self.concat_dataset.datasets:
            total += len(dataset)
            boundaries.append(total)
        return boundaries

    def __iter__(self):
        dataset_idx = 0
        while dataset_idx < len(self.dataset_boundaries):
            start_idx = 0 if dataset_idx == 0 else self.dataset_boundaries[dataset_idx - 1]
            end_idx = self.dataset_boundaries[dataset_idx]
            indices = list(range(start_idx, end_idx))
            random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i + self.batch_size]
            dataset_idx += 1

    def __len__(self):
        return sum(len(dataset) for dataset in self.concat_dataset.datasets) // self.batch_size

import torch
from torch.utils.data import DistributedSampler, Sampler

class DistributedCompressionLevelSampler(Sampler):
    def __init__(self, concat_dataset, batch_size, num_replicas=None, rank=None, shuffle=True, drop_last=True):
        if num_replicas is None:
            if not torch.distributed.is_initialized():
                raise RuntimeError("Requires distributed package to be initialized")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_initialized():
                raise RuntimeError("Requires distributed package to be initialized")
            rank = torch.distributed.get_rank()

        self.concat_dataset = concat_dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.dataset_boundaries = self._calculate_dataset_boundaries()
        self.total_size = sum(len(dataset) for dataset in self.concat_dataset.datasets)
        self.num_samples = self.total_size // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def _calculate_dataset_boundaries(self):
        boundaries = []
        total = 0
        for dataset in self.concat_dataset.datasets:
            total += len(dataset)
            boundaries.append(total)
        return boundaries

    def __iter__(self):
        # evenly distribute indices starting from the current rank
        indices = list(range(self.rank, self.total_size, self.num_replicas))
        # shuffle indices if required
        if self.shuffle:
            if self.rank == 0:
                random.shuffle(indices)
            # Ensure all ranks have the same indices if shuffling by broadcasting from rank 0
            indices = torch.tensor(indices).to(torch.int64)
            torch.distributed.broadcast(indices, src=0)
            indices = indices.tolist()

        # Subsample indices for this replica
        assert len(indices) == self.num_samples

        # Now yield batches
        for i in range(0, len(indices), self.batch_size):
            yield indices[i:i + self.batch_size]

    def __len__(self):
        return self.num_samples // self.batch_size

class CompressionDataModule(LightningDataModule):
    def __init__(self, cfg=None, transform=None, compression_level=0):
        super().__init__()
        self.cfg = cfg
        self.local_batch_size = cfg.experiment.local_batch_size
        self.val_batch_size = cfg.experiment.val_batch_size
        self.num_workers = cfg.dataset.num_workers
        self.transform = transform
        self.compression_level = compression_level

    def setup(self):
        self.datasets = []
        self.datasets = CompressionDataset(compression_level=self.compression_level, transform=self.transform)
        self.train_size = int(0.98 * len(self.datasets))
        self.var_size = len(self.datasets) - self.train_size
        self.train_dataset, self.val_dataset = random_split(self.datasets, [self.train_size, self.var_size])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.local_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            # drop_last=True,
            # shuffle=False,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            # drop_last=True,
            # shuffle=False,
        )