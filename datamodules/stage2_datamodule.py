from typing import Optional
from collections.abc import Callable

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

from datamodules.datasets.mapstyle import MapStyleDataset, VQAMapStyleDataset
from datamodules.datasets.web_ds import WebDatasetPartitionedShard
from datamodules.datasets.dataclass import Items
#from it2it.datamodules.datasets.coco import COCO



class DataModule(pl.LightningDataModule):
    DATASETS = {"mapstyle", "webdataset"}
    #DATASETS = {"mapstyle", "webdataset"}

    def __init__(
        self, cfg,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        pin_memory: bool = False,
        epoch: int = 0,
        total_gpus: int = 1,
    ):

        assert cfg.dataset.type in self.DATASETS, f"{cfg.dataset.type} isn't currently supported.."
        super().__init__()
        self.cfg = cfg
        self.ds_type = cfg.dataset.type
        self.ds_name = cfg.dataset.name
        self.ds_loc = cfg.dataset.loc
        self.tokenizer_type = cfg.dataset.tokenizer.type
        self.bpe_pdrop = cfg.dataset.tokenizer.hparams.bpe_pdrop
        self.train_batch_size = cfg.experiment.local_batch_size
        self.val_batch_size = cfg.experiment.val_batch_size
        self.num_workers = cfg.dataset.num_workers
        self.shuffle = cfg.dataset.shuffle
        self.seed = cfg.experiment.seed
        self.text_ctx = cfg.dataset.tokenizer.hparams.context_length

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.pin_memory = pin_memory
        self.epoch = epoch
        self.total_gpus = total_gpus
        self.gt_text = cfg.dataset.gt_text

        self.data_train: Optional[IterableDataset] = None
        self.data_val: Optional[IterableDataset] = None
        self.data_test: Optional[IterableDataset] = None


    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        if self.ds_type == "mapstyle":
            if self.ds_name.startswith("vqa"):
                ds = VQAMapStyleDataset
            else:
                ds = MapStyleDataset
        elif self.ds_type == "webdataset":
            ds = WebDatasetPartitionedShard
        else:
            raise ValueError("this should be detected in __init__()")

        self.data_train = ds(
            name=self.ds_name,
            loc=self.ds_loc,
            split="train",
            transform=self.train_transform,
            tokenizer_type=self.tokenizer_type,
            bpe_pdrop=self.bpe_pdrop,
            seed=self.seed,
            shuffle=self.shuffle,
            epoch=self.epoch,
            text_ctx=self.text_ctx,
            total_gpus=self.total_gpus,
            gt_text=self.gt_text
        )

        from coco_dataloader import CocoDataset
        karpathy_file = '/ssd0/data/coco/annotations/karpathy/dataset_coco_test.json'
        root_dir = '/ssd0/data/coco/images/val2014'
        start_index = 0
        end_index = 256

        self.data_val = CocoDataset(root_dir, karpathy_file, tokenizer=None, start_index=start_index, end_index=end_index)

        if not self.ds_type == "webdataset":
            self.data_test = ds(
                name=self.ds_name,
                loc=self.ds_loc,
                split=self.cfg.experiment.test_split,
                transform=self.val_transform,
                tokenizer_type=self.tokenizer_type,
                bpe_pdrop=None,
                seed=self.seed,
                shuffle=False,
                epoch=self.epoch,
                text_ctx=self.text_ctx,
                total_gpus=self.total_gpus,
                gt_text=self.gt_text,
                is_test=True
            )


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=Items
        )

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.data_val,
            batch_size=self.val_batch_size,
            collate_fn=self.data_val.collate_fn,
            pin_memory=self.pin_memory,
            num_workers=4
        )
        return val_dataloader

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=Items
        )

    def set_train_epoch(self, epoch):
        self.data_train.set_epoch(epoch)
