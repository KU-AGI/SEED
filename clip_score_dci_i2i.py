import os
from PIL import Image
from transformers import CLIPModel, AutoProcessor, AutoTokenizer
import torch
from torch.nn.functional import cosine_similarity
from peft import PeftModel
from tqdm import tqdm

DCI_LORA_MODEL_PATH = "/home/zheedong/Projects/DCI/models/dci_pick1"
DAC_CKPT_PATH = "/home/zheedong/Projects/DCI/dac_checkpoints/LLM_cp.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

original_image_root = "/home/zheedong/Projects/DCI/data/densely_captioned_images/photos"
generated_image_root = "/home/zheedong/Projects/SEED/seed_tokenizer_img2img_variation_DCI"
# generated_image_root = "/home/zheedong/Projects/SEED/seed_llama_I_text2img_generation_DCI_Short_Caption"
# generated_image_root = "/home/zheedong/Projects/SEED/seed_llama_I_text2img_generation_DCI_Long_Caption"

clip_model_id = "openai/clip-vit-base-patch32"
# clip_model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
print(f"Base Model: {clip_model_id}")
model = CLIPModel.from_pretrained(clip_model_id)
loaded = PeftModel.from_pretrained(model, DCI_LORA_MODEL_PATH)
model = loaded.merge_and_unload()
print(f"Use sDCI CLIP")

model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(clip_model_id)
processor = AutoProcessor.from_pretrained(clip_model_id)

image_list = [image_name for image_name in os.listdir(generated_image_root) if image_name.endswith(".jpg")]

clip_score = 0
total_len = 0

with torch.no_grad():
    for image_name in tqdm(image_list):
        total_len += 1

        original_image = Image.open(f"{original_image_root}/{image_name}").convert("RGB")
        generated_image = Image.open(f"{generated_image_root}/{image_name}").convert("RGB")

        original_image_tensor = processor(images=original_image, return_tensors="pt").to(device)
        original_image_feature = model.get_image_features(pixel_values=original_image_tensor.pixel_values)

        generated_image_tensor = processor(images=generated_image, return_tensors="pt").to(device)
        generated_image_feature = model.get_image_features(pixel_values=generated_image_tensor.pixel_values)

        clip_score += cosine_similarity(original_image_feature, generated_image_feature)

clip_score /= total_len
clip_score *= 100
print(f"Image Path: {generated_image_root}")
print(f"CLIP Score: {clip_score.item()} | CLIP Model: {clip_model_id} | Total Length: {total_len}")