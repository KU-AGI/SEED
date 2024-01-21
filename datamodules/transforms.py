from collections.abc import Callable

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from datamodules.utils import convert_image_to_rgb

class DalleAugmentation(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, img: int):
        if hasattr(F, '_get_image_size'):
            w, h = F._get_image_size(img)
        else:
            w, h = F.get_image_size(img)
        s_min = min(w, h)

        off_h = torch.randint(low=3 * (h - s_min) // 8,
                              high=max(3 * (h - s_min) // 8 + 1, 5 * (h - s_min) // 8),
                              size=(1,)).item()
        off_w = torch.randint(low=3 * (w - s_min) // 8,
                              high=max(3 * (w - s_min) // 8 + 1, 5 * (w - s_min) // 8),
                              size=(1,)).item()

        img = F.crop(img, top=off_h, left=off_w, height=s_min, width=s_min)

        t_max = max(min(s_min, round(9 / 8 * self.size)), self.size)
        t = torch.randint(low=self.size, high=t_max + 1, size=(1,)).item()
        img = F.resize(img, [t, t])
        return img


class DalleTransform(Callable):
    splits = {"train", "val"}

    def __init__(self, cfg, split: str):
        assert split in self.splits, f"{split} is not in {self.splits}"
        self._resolution = cfg.dataset.transform.hparams.resolution
        if split == 'train':
            self._transforms = transforms.Compose(
                [
                    convert_image_to_rgb,
                    DalleAugmentation(size=self._resolution),
                    transforms.RandomCrop(size=(self._resolution, self._resolution)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
        else:
            self._transforms = transforms.Compose(
                [
                    convert_image_to_rgb,
                    transforms.Resize(size=self._resolution),
                    transforms.CenterCrop(size=self._resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )

    def __call__(self, sample):
        return self._transforms(sample)


