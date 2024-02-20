import os
import open_clip
import torch
from PIL import Image
from tqdm import tqdm
from glob import glob

from transformers import CLIPProcessor, CLIPModel

def calculate_clip_s(origial_image_path, generated_image_path, model_clip, preprocess_clip):
    original_image = Image.open(origial_image_path)
    generated_image = Image.open(generated_image_path)
    #original_image = preprocess_clip(original_image).unsqueeze(0).to('cuda')
    #generated_image = preprocess_clip(generated_image).unsqueeze(0).to('cuda')
    original_image = preprocess_clip(images=original_image, return_tensors='pt')['pixel_values'].to('cuda')
    generated_image = preprocess_clip(images=generated_image, return_tensors='pt')['pixel_values'].to('cuda')
    with torch.no_grad():
        # original_image_features = model_clip.encode_image(original_image)
        # generated_image_features = model_clip.encode_image(generated_image)
        original_image_features = model_clip.get_image_features(pixel_values=original_image)
        generated_image_features = model_clip.get_image_features(pixel_values=generated_image)
    s = torch.cosine_similarity(original_image_features, generated_image_features, dim=-1)
    return s.item()

def calculate_clip_s_for_folder(original_image_folder, generated_image_folder):
    s_list = []
    file_list = glob(os.path.join(generated_image_folder, '*.jpg'))
    #model_clip, _,  preprocess_clip = open_clip.create_model_and_transforms('ViT-L-14', device='cuda')
    model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
    preprocess_clip = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    for i, file in enumerate(tqdm(file_list)):
        file = os.path.basename(file)
        original_image_path = os.path.join(original_image_folder, file)
        generated_image_path = os.path.join(generated_image_folder, file)
        s = calculate_clip_s(original_image_path, generated_image_path, model_clip, preprocess_clip)
        s_list.append(s)
    if len(s_list) == 0:
        return 0
    print("clip score: ", sum(s_list) / len(s_list))
    return sum(s_list) / len(s_list)

if __name__ == '__main__':
    generated_image_folder = '/ssd0/checkpoints/seed_tokenizer/seed_training_proj_and_transformer/seed_training_proj/30012024_023406/2'
    original_image_folder =  '/ssd0/data/coco/images/val2014'
    s = calculate_clip_s_for_folder(original_image_folder, generated_image_folder)
    print(s)