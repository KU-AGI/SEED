from datamodules.compression_datamodule import CompressionDataModule
from torchvision import transforms
import webdataset as wds
import os
import io
from PIL import Image
import torch

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

if __name__ == '__main__':
    dataloader = CompressionDataModule(batch_size=128, num_workers=32, transform=transform)
    dataloader.setup()
    for batch in dataloader.train_dataloader():
        print(batch)