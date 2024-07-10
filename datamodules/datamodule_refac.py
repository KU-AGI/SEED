import os
import time
import json
import random
import hydra
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from transformers import AutoTokenizer

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

# custom module
from datamodules.dataclass_main import Finetune_Pair, Finetune_Edit, Finetune_Conversation, Finetune_Understand, Finetune_VQA, Finetune_Interleaved, CocoDataset, WeightedDataLoaderSampler, WeightedDataLoaderSampler_2

# ===================== Data Module WRAPPER =====================
class MainDataModule(pl.LightningDataModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer    
        self.transform = hydra.utils.instantiate(OmegaConf.load(config.transform_cfg_path))
    
        self.train_dataset = []
        self.val_dataset = []
        self.data_weights = config.dataset.train_config.weights
        self.world_size = config.dist.n_gpus
        self.rank = 0  
        self.seed = config.experiment.seed
        
        # dataloader config
        self.per_device_train_bsz = config.experiment.local_batch_size # actually, not meaningful
        self.per_device_val_bsz = config.experiment.val_batch_size
        self.num_workers = config.dataset.num_workers

        self.one_epoch_data_size = config.dataset.train_config.one_epoch_data_size
       
        self.total_training_steps = ((config.experiment.max_epochs * self.one_epoch_data_size) / self.world_size) / self.per_device_train_bsz

    def setup(self, stage=None):
        
        """
        In this function, "config" is the config of specific data{i}.
        """
    
        # make train dataset
        i = 0
        while True:
            try:
                data_config = self.config.dataset.train_config[f"data_{i}"]
                
                self.train_dataset.append(Finetune_Pair(config=data_config, 
                                                    transform=self.transform,
                                                    tokenizer=self.tokenizer,
                                                    world_size=self.world_size))
                # add idx
                i += 1
                
            # end of dataset
            except KeyError:
                print(f"Total {i} datasets are loaded.")
                break            
                
    
        # val dataset (for metric)
        end_index = self.config.dataset.val_config.end_index if self.config.dataset.val_config.end_index is not None else None
        self.val_dataset = CocoDataset(
            root_dir=self.config.dataset.val_config.root_dir,
            karpathy_file=self.config.dataset.val_config.karpathy_file_path,
            transform=self.transform,
            tokenizer=None,
            start_idx=self.config.dataset.val_config.start_index,
            end_idx=end_index,
        )
        print("DataModule Setup Done!")

    def train_dataloader(self): 
        if len(self.train_dataset) == 1:
            return DataLoader(
                self.train_dataset[0],
                batch_size=self.per_device_train_bsz,
                collate_fn=self.train_dataset[0].collate_fn,
                num_workers=self.num_workers
            )
        
        else:
            _loaders = []
            for i in range(len(self.train_dataset)):
                _loaders.append(DataLoader(
                    self.train_dataset[i],
                    batch_size=self.per_device_train_bsz,
                    collate_fn=self.train_dataset[i].collate_fn,
                    num_workers=self.num_workers
                    )
                )
            return WeightedDataLoaderSampler_2(dataloaders=_loaders, data_weights=self.data_weights)        
        
    def val_dataloader(self):
        return DataLoader(
                self.val_dataset, 
                num_workers=self.num_workers, 
                batch_size=self.per_device_val_bsz,
                collate_fn=self.val_dataset.collate_fn
            )

if __name__ == "__main__":
    # test
    config = OmegaConf.load("/home/zheedong/Projects/SEED/configs/data/refac_data.yaml")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_cfg_path)
    dm = MainDataModule(config=config, tokenizer=tokenizer)
    dm.setup()
    print("DataModule Setup Done!")