from typing import Optional
from collections.abc import Callable

import torch
import braceexpand
import webdataset as wds
from torch.utils.data import IterableDataset
from datamodules.tokenizers import TokenizerUtils
from datamodules.datasets.dataclass import Item
import random


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


class WebDatasetPartitionedShard(IterableDataset, TokenizerUtils):
    names = {"keroberony100m", "keroberony788m", "cc15m",
             "it2it20m", "it2it20m_nc", "it2it120m",
             "signals17m", "it2it22m",
             "coco", "cc15m_vg_sbu", "cc15m_vg_sbu_kb100m", "cc15m_vg_sbu_signals17m",
             "coco2014", "cc3m_coco"}
    splits = {"train", "val"}
    locs = {"bcloud", "gcp"}
    urls_info = {
        ("it2it120m", "bcloud", "train",): (
            "/data/public/rw/keroberony/v1.0.0-beta2/webdataset/100m/{000000..008191}.tar",
            "/data/public/rw/datasets/cc3m_resized/wds_shards/train/cc3m-{000000..000287}.tar",
            "/data/public/rw/datasets/CC12M/wds_shards/cc-{000000..001175}.tar",
            "/data/public/rw/datasets/coco/wds_shards/train/2014/{000000..000040}.tar",
            "/data/public/rw/datasets/coco/wds_shards/train/2017/{000000..000058}.tar",
            "/data/public/rw/datasets/VG/wds_shards/train/{000000..000539}.tar",
            "/data/public/rw/datasets/sbu/wds_shards/train/{000000..000084}.tar",
        ),
        ("it2it120m", "bcloud", "val",): (
            "/data/public/rw/datasets/coco/wds_shards/val/2014/{000000..000404}.tar",
        ),

        ("it2it20m", "bcloud", "train",): (
            "/data/public/rw/datasets/cc3m_resized/wds_shards/train/cc3m-{000000..000287}.tar",
            "/data/public/rw/datasets/CC12M/wds_shards/cc-{000000..001175}.tar",
            "/data/public/rw/datasets/coco/wds_shards/train/2014/{000000..000040}.tar",
            "/data/public/rw/datasets/coco/wds_shards/train/2017/{000000..000058}.tar",
            "/data/public/rw/datasets/VG/wds_shards/train/{000000..000539}.tar",
            "/data/public/rw/datasets/sbu/wds_shards/train/{000000..000084}.tar",
        ),
        ("it2it20m", "bcloud", "val",): (
            "/data/public/rw/datasets/coco/wds_shards/val/2014/{000000..000404}.tar",
        ),
        ("it2it20m_nc", "bcloud", "train",): (
            "/data/public/rw/datasets/cc3m_resized/wds_shards/train/cc3m-{000000..000287}.tar",
            "/data/public/rw/datasets/CC12M/wds_shards/cc-{000000..001175}.tar",
            "/data/public/rw/datasets/VG/wds_shards/train/{000000..000539}.tar",
            "/data/public/rw/datasets/sbu/wds_shards/train/{000000..000084}.tar",
        ),
        ("it2it20m_nc", "bcloud", "val",): (
            "/data/public/rw/datasets/coco/wds_shards/val/2014/{000000..000404}.tar",
        ),
        ("cc15m", "gcp", "train",): (
            "pipe:gsutil cat gs://mindalle2/cc3m_webds/train/cc3m-{000000..000287}.tar",
            "pipe:gsutil cat gs://mindalle2/cc12m_webds/cc-{000000..001175}.tar",
        ),
        ("cc15m", "bcloud", "train",): (
            "/ssd0/data/cc3m/{00000..00300}.tar",
            "/ssd0/data/cc12m/{00000..01242}.tar",
        ),
        ("cc15m", "bcloud", "val",): (
            "/ssd0/data/cc3m/{00300..00331}.tar",
        ),
        # ("cc15m", "bcloud", "val",):
        #     "/data/public/rw/datasets/cc3m_resized/wds_shards/val/cc3m-{000000..000071}.tar",
        ("cc15m", "gcp", "train",): (
            "pipe:gsutil cat gs://mindalle2/cc3m_webds/train/cc3m-{000000..000287}.tar",
            "pipe:gsutil cat gs://mindalle2/cc12m_webds/cc-{000000..001175}.tar",
        ),
        (
            "cc15m",
            "gcp",
            "val",
        ): "pipe:gsutil cat gs://mindalle2/cc3m_webds/val/cc3m-{000000..000071}.tar",
        ("keroberony100m", "bcloud", "train",): (
            "/data/public/rw/keroberony/v1.0.0-beta2/webdataset/100m/{000000..008191}.tar",
            #"/data/public/rw/datasets/cc3m_resized/wds_shards/train/cc3m-{000000..000287}.tar",
            #"/data/public/rw/datasets/CC12M/wds_shards/cc-{000000..001175}.tar",
        ),
        # (
        #     "keroberony100m",
        #     "bcloud",
        #     "val",
        # ): "/data/public/rw/datasets/cc3m_resized/wds_shards/val/cc3m-{000000..000071}.tar",
        ("keroberony100m", "bcloud", "val",): (
            "/data/public/rw/datasets/coco/wds_shards/val/2014/{000000..000404}.tar",
        ),
        ("keroberony100m", "gcp", "train",): (
            "pipe:gsutil cat gs://mindalle2/keroberony/releases/v1.0.0-beta1/webdataset/{000000..008191}.tar",
            "pipe:gsutil cat gs://mindalle2/cc3m_webds/train/cc3m-{000000..000287}.tar",
            "pipe:gsutil cat gs://mindalle2/cc12m_webds/cc-{000000..001175}.tar",
        ),
        (
            "keroberony100m",
            "gcp",
            "val",
        ): "pipe:gsutil cat gs://mindalle2/cc3m_webds/val/cc3m-{000000..000071}.tar",

        ("signals17m", "bcloud", "train",): (
            "/data/public/rw/datasets/signals17m/webdatasets/train/signals17m-{000000..001739}.tar",
        ),
        ("signals17m", "bcloud", "val",): (
            "/data/public/rw/datasets/coco/wds_shards/val/2014/{000000..000404}.tar",
        ),
        # it2it22m: signals17m + coco + VG + sbu
        ("it2it22m", "bcloud", "train",): (
            "/data/public/rw/datasets/signals17m/webdatasets/train/signals17m-{000000..001739}.tar",
            "/data/public/rw/datasets/coco/wds_shards/train/2014/{000000..000040}.tar",
            "/data/public/rw/datasets/coco/wds_shards/train/2017/{000000..000058}.tar",
            "/data/public/rw/datasets/VG/wds_shards/train/{000000..000539}.tar",
            "/data/public/rw/datasets/sbu/wds_shards/train/{000000..000084}.tar",
        ),
        ("it2it22m", "bcloud", "val",): (
            "/data/public/rw/datasets/coco/wds_shards/val/2014/{000000..000404}.tar",
        ),
        # coco
        ("coco", "bcloud", "train",): (
            "/data/public/rw/datasets/coco/wds_shards/train/2014/{000000..000040}.tar",
            "/data/public/rw/datasets/coco/wds_shards/train/2017/{000000..000058}.tar",
        ),
        ("coco", "bcloud", "val",): (
            "/data/public/rw/datasets/coco/wds_shards/val/2014/{000000..000404}.tar",
        ),
        # cc15m + VG + SBU
        ("cc15m_vg_sbu", "bcloud", "train",): (
            "/data/old_public_rw_datasets_99/workspace/rw/datasets/cc3m_resized/wds_shards/train/cc3m-{000000..000287}.tar",
            "/data/old_public_rw_datasets_99/workspace/rw/datasets/CC12M/wds_shards/cc-{000000..001175}.tar",
            "/data/old_public_rw_datasets_99/workspace/rw/datasets/VG/wds_shards/train/{000000..000539}.tar",
            "/data/old_public_rw_datasets_99/workspace/rw/datasets/sbu/wds_shards/train/{000000..000084}.tar",
        ),
        ("cc15m_vg_sbu", "bcloud", "val",): (
            "/data/old_public_rw_datasets_99/workspace/rw/datasets/cc3m_resized/wds_shards/train/cc3m-{000000..000071}.tar",
        ),
        # c15m + VG + SBU + KB100m
        ("cc15m_vg_sbu_kb100m", "bcloud", "train",): (
            "/data/public/rw/datasets/cc3m_resized/wds_shards/train/cc3m-{000000..000287}.tar",
            "/data/public/rw/datasets/CC12M/wds_shards/cc-{000000..001175}.tar",
            "/data/public/rw/datasets/VG/wds_shards/train/{000000..000539}.tar",
            "/data/public/rw/datasets/sbu/wds_shards/train/{000000..000084}.tar",
            "/data/public/rw/keroberony/v1.0.0-beta2/webdataset/100m/{000000..008191}.tar",
        ),
        ("cc15m_vg_sbu_kb100m", "bcloud", "val",): (
            "/data/public/rw/datasets/coco/wds_shards/val/2014/{000000..000404}.tar",
        ),
        # cc15m + VG + SBU + signals17m
        ("cc15m_vg_sbu_signals17m", "bcloud", "train",): (
            "/data/public/rw/datasets/cc3m_resized/wds_shards/train/cc3m-{000000..000287}.tar",
            "/data/public/rw/datasets/CC12M/wds_shards/cc-{000000..001175}.tar",
            "/data/public/rw/datasets/VG/wds_shards/train/{000000..000539}.tar",
            "/data/public/rw/datasets/sbu/wds_shards/train/{000000..000084}.tar",
            "/data/public/rw/datasets/signals17m/webdatasets/train/signals17m-{000000..001739}.tar",
        ),
        ("cc15m_vg_sbu_signals17m", "bcloud", "val",): (
            "/data/public/rw/datasets/coco/wds_shards/val/2014/{000000..000404}.tar",
        ),
        # coco2014
        ("coco2014", "bcloud", "train",): (
            "/data/public/rw/datasets/coco/wds_shards/train/2014/{000000..000040}.tar",
        ),
        ("coco2014", "bcloud", "val",): (
            "/data/public/rw/datasets/coco/wds_shards/val/2014/{000000..000404}.tar",
        ),
        # cc3m + coco
        ("cc3m_coco", "bcloud", "train",): (
            "/ssd0/data/cc3m/{00000..00320}.tar",
            "/ssd0/data/mscoco/{00000..00059}.tar",
        ),
        ("cc3m_coco", "bcloud", "val",): (
            "/ssd0/data/cc3m/{00321..00331}.tar",
        ),
    }

    def __init__(
        self,
        name: str,
        loc: str,
        split: str,
        transform: Optional[Callable] = None,
        tokenizer_type: Optional[str] = None,
        bpe_pdrop: Optional[float] = None,
        seed=None,
        shuffle=True,
        epoch=0,
        text_ctx=77,
        total_gpus=1,
        gt_text=False,
        is_test=False,
    ):
        assert name in self.names, f"{name} is not in {self.names}"
        assert split in self.splits, f"{split} is not in {self.splits}"
        assert loc in self.locs, f"{loc} is not in {self.locs}"
        assert seed is not None, "seed should be specified"

        super().__init__()

        self.name = name
        try:
            urls = self.urls_info[(name, loc, split)]
        except KeyError:
            raise ValueError(f"{name}_{loc}_{split} is not supported..")
        self.urls = expand_urls(urls)
        assert isinstance(self.urls[0], str)

        self._split = split
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch
        self.transform = transform
        self.gt_text = gt_text
        self.build_tokenizer(tokenizer_type, text_ctx, lowercase=True, dropout=bpe_pdrop)

        self.n_samples = (
            int((self._get_n_samples(name, split) // total_gpus) * 0.8)
            if split == "train"
            else self._get_n_samples(name, split)
        )

    def __len__(self):
        return self.n_samples

    def _get_n_samples(self, name, split):
        if name == "keroberony100m":
            return 100 * 1000000 if split == "train" else 10000
        elif name == "keroberony788m":
            return 788 * 1000000 if split == "train" else 10000
        elif name == "cc15m":
            return 14146678 if split == "train" else 10000
        elif name == "it2it20m":
            if split == "train":
                ret = 14146678 # cc15m
                ret += 414113 + 591753 # coco(2014 + 2017)
                ret += 4867814 + 540868 # vg
                ret += 1000000 # sbu
            else:
                # ret = 202654 + 25014 # coco(2014 + 2017)
                ret = 10000
            return ret
        elif name == "it2it20m_nc":
            if split == "train":
                ret = 14146678 # cc15m
                ret += 4867814 + 540868 # vg
                ret += 1000000 # sbu
            else:
                # ret = 202654 + 25014 # coco(2014 + 2017)
                ret = 10000
            return ret
        elif name == "it2it120m":
            if split == "train":
                ret = 100 * 1000000 # kb100
                ret += 14146678 # cc15m
                ret += 414113 + 591753 # coco(2014 + 2017)
                ret += 4867814 + 540868 # vg
                ret += 1000000 # sbu
            else:
                ret = 10000
            return ret
        elif name == 'signals17m':
            if split == "train":
                ret = 17 * 1000000
            else:
                ret = 10000
            return ret
        elif name == 'it2it22m':
            if split == "train":
                ret = 17 * 1000000
                ret += 414113 + 591753  # coco(2014 + 2017)
                ret += 4867814 + 540868  # vg
                ret += 1000000  # sbu
            else:
                ret = 10000
            return ret
        elif name == 'coco':
            if split == "train":
                ret = 414113 + 591753  # coco(2014 + 2017)
            else:
                ret = 10000
            return ret
        elif name == 'cc15m_vg_sbu':
            if split == "train":
                ret = 14146678 # cc15m
                ret += 4867814 + 540868  # vg
                ret += 1000000  # sbu
            else:
                ret = 10000
            return ret
        elif name == 'cc15m_vg_sbu_kb100m':
            if split == "train":
                ret = 14146678 # cc15m
                ret += 4867814 + 540868  # vg
                ret += 1000000  # sbu
                ret += 100 * 1000000 # kb100
            else:
                ret = 10000
            return ret
        elif name == 'cc15m_vg_sbu_signals17m':
            if split == "train":
                ret = 14146678 # cc15m
                ret += 4867814 + 540868  # vg
                ret += 1000000  # sbu
                ret += 17 * 1000000 # signals17m
            else:
                ret = 10000
            return ret
        elif name == 'coco2014':
            if split == "train":
                ret = 414113
            else:
                ret = 10000
            return ret
        elif name == 'cc3m_coco':
            if split == "train":
                ret = 3000000 + 414113
            else:
                ret = 10000
            return ret

        else:
            raise ValueError()

    def __iter__(self):
        urls = self.urls.copy()

        if self.shuffle:
            # below is taken from DistributedSampler
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(urls), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(urls)))
        urls = [urls[idx] for idx in indices]

        # ignore handler should be used to run a job in gcp,
        # otherwise fetching tars in GCS sometimes raise connection error.
        # please see  https://github.xiaofsu.com/webdataset/webdataset/issues/112
        ds = wds.WebDataset(
            urls, nodesplitter=wds.split_by_node, handler=wds.ignore_and_continue
        )
        if self.shuffle:
            ds = ds.shuffle(1000) # url level shuffle
        ds = ds.decode("pil")
        if self.shuffle:
            ds = ds.shuffle(10 * 1000) # sample level shuffle?

        ds = ds.to_tuple(
            "jpg", "txt", "__url__", handler=wds.ignore_and_continue
        )  # without ignore handler, sometimes raise KeyValueError

        for img, txt, url in ds:
            # txt, mask = self.tokenizer.padded_tokens_and_mask([txt], self.text_ctx)
            if 'signals17m' in url:
                gt_txts = [gt_txt for gt_txt in txt.split('.') if gt_txt.strip()]
                if len(gt_txts) == 0:
                    gt_txts = ['']
                txt = random.choice(gt_txts) + '.'

            txt_item = self.get_input(txt, pre_proc=self.pre_caption)
            img = self.transform(img)

            item = Item(img=img, txt=txt_item.txt, txt_mask=txt_item.txt_mask, txt_pos_id=txt_item.pos_id, gt_txt=[txt])
            yield item

    def set_epoch(self, epoch):
        self.epoch = epoch
