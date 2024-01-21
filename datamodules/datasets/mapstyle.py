import os
from typing import Optional
from collections.abc import Callable
import random
import numpy as np
import json

import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from datamodules.datasets.coco import CocoCaptions
from datamodules.tokenizers import TokenizerUtils
from torch.utils.data import ConcatDataset
from datamodules.datasets.dataclass import Item


class MapStyleDataset(VisionDataset, TokenizerUtils):
    names = {"cc3m", "cc15m", "coco", "coco2014", "nocaps"}
    splits = {"train", "val", "test", "test_dev"}
    locs = {"bcloud"}

    def __init__(
        self,
        name: str,
        loc: str,
        split: str,
        transform: Optional[Callable] = None,
        tokenizer_type: Optional[str] = None,
        bpe_pdrop: Optional[float] = None,
        text_ctx: int = 77,
        gt_text: bool = False,
        is_test: bool = False,
        **ignore_kwargs,
    ):
        assert name in self.names, f"{name} is not in {self.names}"
        assert split in self.splits, f"{split} is not in {self.splits}"
        assert loc in self.locs, f"{loc} is not in {self.locs}"

        if name in ["cc3m", "cc15m"]:
            super().__init__("/data/public/rw/datasets/", transform=transform)
        elif name.startswith("coco"):
            super().__init__('/ssd0/data/coco', transform=transform)

        self.name = name
        self.split = split
        self.gt_text = gt_text
        self.build_tokenizer(tokenizer_type, text_ctx, lowercase=True, dropout=bpe_pdrop)

        self.items = []
        if name in ["cc3m", "cc15m"]:
            if split == "train":
                if name == "cc3m":
                    list_names = [f"{self.root}/cc3m_resized/train_list.txt"]
                elif name == "cc15m":
                    list_names = [
                        f"{self.root}/cc3m_resized/train_list.txt",
                        f"{self.root}/CC12M/resized/cc12m_with_hash_no_url_only_valid.tsv",
                    ]
            else:
                list_names = [f"{self.root}/cc3m_resized/val_list.txt"]

            for idx, list_name in enumerate(list_names):
                for line in open(list_name, "r").readlines():
                    toks = line.strip().split("\t")
                    assert len(toks) == 2
                    (imgpath, text) = toks
                    if split == "train":
                        if idx == 0:
                            self.items.append(
                                (os.path.join(self.root, "cc3m_resized", imgpath), text)
                            )
                        else:
                            self.items.append(
                                (
                                    os.path.join(
                                        self.root,
                                        "CC12M/resized/images",
                                        f"{imgpath}.jpg",
                                    )
                                    if name == "cc15m"
                                    else os.path.join(self.root, "cc3m_resized", imgpath),
                                    text,
                                )
                            )
                    else:
                        self.items.append(
                            (os.path.join(self.root, "cc3m_resized", imgpath), text)
                        )
        elif name.startswith("coco"):
            data_list = []
            if split == "test":
                data_list.append(CocoCaptions(root=f'{self.root}/images/val2014', annFile=f'{self.root}/annotations/dataset_coco_test.json'))
            else:
                data_list.append(CocoCaptions(root=f'{self.root}/images/{split}2014',
                                              annFile=f'{self.root}/annotations/captions_{split}2014.json'))
            if name == 'coco' and split=="train":
                data_list.append(CocoCaptions(root=f'{self.root}/images/{split}2017',
                                              annFile=f'{self.root}/annotations/captions_{split}2017.json'))
            self.items = ConcatDataset(data_list)
        elif name == "nocaps":
            #self.transform = self._build_transform("val", cfg.dataset.transform_hparams)
            self.transform = transform
            self.build_tokenizer(tokenizer_type, text_ctx, lowercase=True, dropout=bpe_pdrop)

            # load annotation file
            ann_file_path = "/data/project/rw/it2it/datasets/nocaps/nocaps_val_4500_captions.json"
            anns = json.load(open(ann_file_path, "r"))
            img_anns = anns["images"]
            img_dir = "/data/public/rw/datasets/open_images_v5/validation" if split == "val" \
                else "/data/public/rw/datasets/open_images_v5/test"

            # build training items
            self.items = []
            for ann in img_anns:
                filename = ann["file_name"]
                item = {
                    "id": ann["id"],
                    "domain": ann["domain"],
                    "imgpath": os.path.join(img_dir, filename[:2], filename[2:5], filename),
                    "captions": [],
                    "coco_url": ann["coco_url"]
                }
                self.items.append(item)

            # if validation, we attach GT captions in training items for qualitative check
            if split == "val":
                cap_anns = anns["annotations"]
                for ann in cap_anns:
                    self.items[ann["image_id"]]["captions"].append(
                        ann["caption"]
                    )

            print(f"total items: {len(self.items)}")

        self.custom_len = None

    def set_custom_length(self, l):
        assert len(self.items) >= l
        self.custom_len = l

    def __len__(self):
        if self.custom_len is not None:
            return self.custom_len
        if self.split == 'val':
            return 5000
        return len(self.items)

    def __getitem__(self, item: int):
        if self.name in ["cc3m", "cc15m"]:
            imgpath, txt = self.items[item]
            gt_txt = txt
            img = Image.open(imgpath)

            input, input_mask = self.get_input(txt, pre_proc=self.pre_caption)
            img = self.transform(img)
            domain = None

        elif self.name.startswith("coco"):
            imgpath, img, gt_txt = self.items[item]

            if len(gt_txt) > 5:
                gt_txt = gt_txt[:5]
            elif len(gt_txt) < 5:
                gt_txt.append(gt_txt[:(5 - len(gt_txt))])

            if self.transform:
                img = self.transform(img)

            # text = ' '.join(text)  # text is a list of sentences. Concat them.
            if self.split == "train":
                rnd_txt = random.randint(0, len(gt_txt)-1)
                txt = gt_txt[rnd_txt]
            else:
                txt = gt_txt[0]

            txt_item = self.get_input(txt, pre_proc=self.pre_caption)
            domain = None

        elif self.name == 'nocaps':
            instance = self.items[item]

            # load image
            imgpath = instance["imgpath"]
            img = Image.open(imgpath).convert("RGB")
            img = self.transform(img)

            # prepare text token
            txt = instance["captions"][0] if self.split == "val" else "null sentence"
            gt_txt = instance["captions"] if self.split == "val" else None
            txt_item = self.get_input(txt, pre_proc=self.pre_caption)
            domain = instance["domain"]

        item = Item(imgpath=imgpath, img=img, txt=txt_item.txt, txt_mask=txt_item.txt_mask, txt_pos_id=txt_item.pos_id, gt_txt=gt_txt, domain=domain)
        return item

    def set_epoch(self, epoch):
        self.epoch = epoch



class VQAMapStyleDataset(VisionDataset, TokenizerUtils):
    names = {"vqav2"}
    splits = {"train", "val", "test", "test_dev"}
    locs = {"bcloud"}

    def __init__(
        self,
        name: str,
        loc: str,
        split: str,
        transform: Optional[Callable] = None,
        tokenizer_type: Optional[str] = None,
        bpe_pdrop: Optional[float] = None,
        text_ctx: int = 77,
        gt_text: bool = False,
        is_test: bool = False,
        **ignore_kwargs,
    ):
        assert name in self.names, f"{name} is not in {self.names}"
        assert split in self.splits, f"{split} is not in {self.splits}"
        assert loc in self.locs, f"{loc} is not in {self.locs}"

        if name == 'vqav2':
            super().__init__("/data/public/rw/team-mmu/dataset/VQAv2/", transform=transform)

        self.name = name
        self.split = split
        self.gt_text = gt_text
        self.text_ctx = text_ctx
        self.items = []

        self.is_test = is_test

        if name == 'vqav2':
            if not is_test:
                fpath = f"/data/public/rw/team-mmu/ALBEF/data/vqa_{split}.json"
                assert os.path.exists(fpath)
                with open(fpath, 'r') as fin:
                    data = json.load(fin)

                for dt in data:
                    question_id, dataset, imgpath, question, answers = None, None, None, None, None
                    question_id = dt['question_id']
                    dataset = dt['dataset']
                    imgpath = os.path.join(self.root, dt['image'])
                    question = dt['question']
                    if 'answer' in dt:
                        answers = dt['answer']

                    self.items.append((question_id, dataset, imgpath, question, answers))
            else:
                if not split == 'test':
                    annData = json.load(open(f"/data/public/rw/team-mmu/dataset/VQAv2/v2_mscoco_{split}2014_annotations.json"))
                    quesData = json.load(open(f"/data/public/rw/team-mmu/dataset/VQAv2/v2_OpenEnded_mscoco_{split}2014_questions.json"))
                else:
                    annData = {'annotations': None}
                    quesData = json.load(open(f"/data/public/rw/team-mmu/dataset/VQAv2/v2_OpenEnded_mscoco_{split}2015_questions.json"))

                anns = annData['annotations']
                ques = quesData['questions']

                n_q = len(ques)
                for i in range(n_q):
                    ques_dt = ques[i]
                    question_id, dataset, imgpath, question, answers = None, None, None, None, None
                    if anns is not None:
                        ann_dt = anns[i]
                        answers = [answer['answer'] for answer in ann_dt['answers']]

                    question_id = ques_dt['question_id']
                    question = ques_dt['question']
                    dataset = 'vqa'
                    imgfname = f'COCO_{quesData["data_subtype"]}_{str(ques_dt["image_id"]).zfill(12)}.jpg'
                    imgpath = os.path.join(self.root, quesData["data_subtype"], imgfname)

                    self.items.append((question_id, dataset, imgpath, question, answers))


        self.max_ques_len = 30

        self.build_tokenizer(tokenizer_type, text_ctx, lowercase=True, dropout=bpe_pdrop, sep_token='?')

    def __len__(self):
        if self.split == 'val' and not self.is_test:
            return 10000
        return len(self.items)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, item: int):
        question_id, dataset, imgpath, question, answers = self.items[item]

        answer = None
        if answers is not None:
            answer_weight = {}
            for answer in answers:
                if answer in answer_weight.keys():
                    answer_weight[answer] += 1 / len(answers)
                else:
                    answer_weight[answer] = 1 / len(answers)

            answers = list(answer_weight.keys())
            weights = list(answer_weight.values())

            answer = np.random.choice(answers, p=weights)

        txt_item = self.get_QA_input(question, answer=answer, max_ques_len=self.max_ques_len)
        img = self.transform(Image.open(imgpath))

        item = Item(imgpath=imgpath, img=img, txt=txt_item.txt, txt_mask=txt_item.txt_mask, txt_pos_id=txt_item.pos_id, id=question_id, gt_txt=answer, cond_txt=question)
        return item

