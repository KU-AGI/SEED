from transformers import CLIPModel, AutoProcessor, AutoTokenizer
import torch
import json
from PIL import Image
from tqdm import tqdm
import os
from torch.nn.functional import cosine_similarity
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel
import statistics

ORIGINAL_IMAGE_PATH_ROOT = "/home/zheedong/Projects/DCI/data/densely_captioned_images/photos"
IMAGE_PATH_ROOT = "/home/zheedong/Projects/SEED/seed_llama_I_text2img_generation_DCI_Long_Caption"
DCI_LORA_MODEL_PATH = "/home/zheedong/Projects/DCI/models/dci_pick1"

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model_id = "openai/clip-vit-base-patch32"
# clip_model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model = CLIPModel.from_pretrained(clip_model_id)

USE_DCI = False
if USE_DCI:
    loaded = PeftModel.from_pretrained(model, DCI_LORA_MODEL_PATH)
    model = loaded.merge_and_unload()
    print("Use sDCI CLIP")
else:
    print("Use CLIP")

model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(clip_model_id)
processor = AutoProcessor.from_pretrained(clip_model_id)

with open("/home/zheedong/Projects/SEED/selected_captions_with_short_caption.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Only check image name in IMAGE_PATH_ROOT
# Regardless image extension is jpg or png
data = [d for d in data if os.path.exists(os.path.join(IMAGE_PATH_ROOT, d["image"]))]

class DCIDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data[idx]["image"]
        short_caption = self.data[idx]["short_caption"]
        long_caption = self.data[idx]["long_caption"]
        original_image = Image.open(os.path.join(ORIGINAL_IMAGE_PATH_ROOT, image_name))
        image = Image.open(os.path.join(IMAGE_PATH_ROOT, image_name))
        if self.transforms:
            original_image = self.transforms(images=original_image, return_tensors="pt")["pixel_values"]
            original_image = original_image.squeeze(0)

            image = self.transforms(images=image, return_tensors="pt")["pixel_values"]
            image = image.squeeze(0)
        return original_image, image, short_caption, long_caption
    
i2i_similarity_list = []
short_similarity_list = []
long_similarity_list = []

dataset = DCIDataset(data, processor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

for d in tqdm(dataloader):
    original_image, image, short_caption, long_caption = d
   
    original_image = original_image.to(device) 
    image = image.to(device)
    short_caption = tokenizer(short_caption, return_tensors="pt", padding=True, truncation=True).to(device)
    long_caption = tokenizer(long_caption, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        original_image_features = model.get_image_features(pixel_values=original_image)
        image_features = model.get_image_features(pixel_values=image)

        i2i_similarity = cosine_similarity(original_image_features, image_features)
        i2i_similarity_list += i2i_similarity.tolist()

        short_text_features = model.get_text_features(**short_caption)
        long_text_features = model.get_text_features(**long_caption)

        short_similarity = cosine_similarity(image_features, short_text_features)
        short_similarity_list += short_similarity.tolist()

        long_similarity = cosine_similarity(image_features, long_text_features)
        long_similarity_list += long_similarity.tolist()

print(f"Image Path Root: {IMAGE_PATH_ROOT}")
print(f"I2I CLIP Score: {sum(i2i_similarity_list) * 100 / len(i2i_similarity_list)} | Standard Deviation: {statistics.stdev(i2i_similarity_list)}")
print(f"T2I CLIP Score with Short Caption: {sum(short_similarity_list) * 100 / len(short_similarity_list)} | Standard Deviation: {statistics.stdev(short_similarity_list)}")
print(f"T2I CLIP Score with Long Caption: {sum(long_similarity_list) * 100 / len(long_similarity_list)} | Standard Deviation: {statistics.stdev(long_similarity_list)}")
