from torch.utils.data import Dataset
from PIL import Image
import json

class COCOValDataSet(Dataset):
    def __init__(self, transform):
        self.transform = transform
        self.coco_val_data_path = '/home/zheedong/Projects/SEED/coco/annotations/captions_val2014.json' 
        with open(self.coco_val_data_path, 'r') as f:
            self.coco_data = json.load(f)
        self.image_root = '/home/zheedong/Projects/SEED/coco/images/val2014'

    def __len__(self):
        return 30000

    def __getitem__(self, idx):
        image_path = self.image_root + '/' + self.coco_data['images'][idx]['file_name']
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, '', self.coco_data['images'][idx]['file_name']