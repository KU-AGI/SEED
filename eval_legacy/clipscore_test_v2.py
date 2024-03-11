import torch
from PIL import Image
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
import os

def clip_score_batch(image_pairs, model_clip, preprocess_clip, device):
    batch_1 = torch.stack([preprocess_clip(pair[0]) for pair in image_pairs]).to(device)
    batch_2 = torch.stack([preprocess_clip(pair[1]) for pair in image_pairs]).to(device)

    with torch.no_grad():
        image_1_features = model_clip.encode_image(batch_1)
        image_2_features = model_clip.encode_image(batch_2)

    scores = torch.nn.functional.cosine_similarity(image_1_features, image_2_features, dim=1)
    return scores

def process_images_in_batches(folder_path, coco_path, model_clip, preprocess_clip, device, batch_size=512):
    img_names = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    total_score = 0
    cnt = 0

    for sp in tqdm(range(0, len(img_names), batch_size)):
        ep = min(sp + batch_size, len(img_names))
        image_pairs = [(Image.open(f'{folder_path}/{name}').convert('RGB'), Image.open(f'{coco_path}/{name}').convert('RGB')) for name in img_names[sp:ep]]
        scores = clip_score_batch(image_pairs, model_clip, preprocess_clip, device)
        total_score += scores.sum().item()
        cnt += len(image_pairs)

    average_score = total_score / cnt
    print(f"Average CLIP similarity: {average_score}")

# Main execution
device = "cuda"
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
preprocess_clip = Compose([
    Resize(224, interpolation=Image.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

testset_path = 'i2i_reconstruction'
coco_path = 'coco/images/val2014'
process_images_in_batches(testset_path, coco_path, model_clip, preprocess_clip, device, batch_size=8192)
