import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from torchvision.datasets import ImageFolder
import torch
from torch.utils.data import Dataset

# Custom Hugging Face dataset class to be compatible with PyTorch
class HuggingFaceDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

        self.class_names = self.dataset.features['label'].names

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']  # Assuming 'image' key holds the image
        # image to RGB 3 chanel
        image = image.convert("RGB")
        label = sample['label']  # Assuming 'label' key holds the label
        text = self.class_names[label]

        if self.transform:
            image = self.transform(image)

        return image, text, label


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4, image_size=448):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        # Transforms for the dataset (train and validation)
        self.image_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)),
        ])

        self.train_dataset = load_dataset("ILSVRC/imagenet-1k", split="train")
        self.val_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation")

    def setup(self, stage=None):
        # Create PyTorch-compatible datasets
        self.train_dataset = HuggingFaceDataset(self.train_dataset, transform=self.image_transforms)
        self.val_dataset = HuggingFaceDataset(self.val_dataset, transform=self.image_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        # Optionally, a test dataloader can be created in a similar way
        pass

# Example of using the DataModule
if __name__ == "__main__":
    data_module = ImageNetDataModule(batch_size=64)
    data_module.prepare_data()
    data_module.setup()

    # Example: iterate over the training dataloader
    for batch in data_module.train_dataloader():
        images, labels = batch
        print(images.shape, labels.shape)
        break