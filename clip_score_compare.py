import json
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from transformers import CLIPModel, AutoProcessor, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

ORIGINAL_IMAGE_ROOT = "/home/zheedong/Projects/DCI/data/densely_captioned_images/photos"

with open("/home/zheedong/Projects/SEED/dci_random_captions_choice.json", "r", encoding="utf-8") as f:
    random_data = json.load(f)

with open("/home/zheedong/Projects/SEED/selected_captions_with_short_caption.json", "r", encoding="utf-8") as f:
    selected_data = json.load(f)

def json_list_to_dict(json_list):
    json_dict = {}
    for json_data in json_list:
        json_dict[json_data["image"]] = json_data
    return json_dict

clip_model_id = "openai/clip-vit-base-patch32"
# clip_model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
print(f"Base Model: {clip_model_id}")
model = CLIPModel.from_pretrained(clip_model_id)

model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(clip_model_id)
processor = AutoProcessor.from_pretrained(clip_model_id)

random_dict = json_list_to_dict(random_data)

for cur_data in selected_data:
    image_name = cur_data["image"]
    short_caption = cur_data["short_caption"]
    selected_long_caption = cur_data["long_caption"]
    random_long_caption = random_dict[image_name]["long_caption"]

    original_image = Image.open(f"{ORIGINAL_IMAGE_ROOT}/{image_name}").convert("RGB")

    original_image_tensor = processor(images=original_image, return_tensors="pt").to(device) 
    original_image_feature = model.get_image_features(pixel_values=original_image_tensor.pixel_values)

    short_caption_tensor = tokenizer(short_caption, return_tensors="pt").to(device)
    short_caption_feature = model.get_text_features(input_ids=short_caption_tensor.input_ids, attention_mask=short_caption_tensor.attention_mask)
    
    selected_long_caption_tensor = tokenizer(selected_long_caption, return_tensors="pt").to(device)
    selected_long_caption_feature = model.get_text_features(input_ids=selected_long_caption_tensor.input_ids, attention_mask=selected_long_caption_tensor.attention_mask)
    
    random_long_caption_tensor = tokenizer(random_long_caption, return_tensors="pt").to(device)
    random_long_caption_feature = model.get_text_features(input_ids=random_long_caption_tensor.input_ids, attention_mask=random_long_caption_tensor.attention_mask)
    
    short_caption_score = torch.cosine_similarity(original_image_feature, short_caption_feature)
    selected_long_caption_score = torch.cosine_similarity(original_image_feature, selected_long_caption_feature)
    random_long_caption_score = torch.cosine_similarity(original_image_feature, random_long_caption_feature)

    