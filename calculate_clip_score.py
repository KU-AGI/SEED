import os
import open_clip
import torch
from PIL import Image
from tqdm import tqdm
from glob import glob

from transformers import CLIPProcessor, CLIPModel
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.transforms import ToTensor

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

def calculate_lpips(original_image_path, generated_image_path, lpips=None):
    if lpips is None:
        lpips = LearnedPerceptualImagePatchSimilarity(normalize=True, net_type='alex').to('cuda')
    totensor = ToTensor()

    original_image = totensor(Image.open(original_image_path).convert('RGB')).unsqueeze(0).to('cuda')
    generated_image = totensor(Image.open(generated_image_path).convert('RGB')).unsqueeze(0).to('cuda')

    H1, W1 = original_image.shape[-2:]
    H2, W2 = generated_image.shape[-2:]

    if H1 != H2 or W1 != W2:
        min_H = min(H1, H2)
        min_W = min(W1, W2)
        original_image = torch.nn.functional.interpolate(original_image, size=(min_H, min_W), mode='bilinear')
        generated_image = torch.nn.functional.interpolate(generated_image, size=(min_H, min_W), mode='bilinear')

    lpips_score = lpips(original_image, generated_image)
    return lpips_score.item()

def calculate_lpips_for_folder(original_image_folder, generated_image_folder):
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True, net_type='alex').to('cuda')
    lpips_list = []
    file_list = glob(os.path.join(generated_image_folder, '*.jpg'))
    for i, file in enumerate(tqdm(file_list)):
        file = os.path.basename(file)
        original_image_path = os.path.join(original_image_folder, file)
        generated_image_path = os.path.join(generated_image_folder, file)
        lpips_score = calculate_lpips(original_image_path, generated_image_path, lpips)
        lpips_list.append(lpips_score)
    if len(lpips_list) == 0:
        return 0
    print("lpips score: ", sum(lpips_list) / len(lpips_list))
    return sum(lpips_list) / len(lpips_list)

def calculate_psnr(original_image_path, generated_image_path):
    psnr = PeakSignalNoiseRatio()
    totensor = ToTensor()

    original_image = totensor(Image.open(original_image_path).convert('RGB')).unsqueeze(0)
    generated_image = totensor(Image.open(generated_image_path).convert('RGB')).unsqueeze(0)

    H1, W1 = original_image.shape[-2:]
    H2, W2 = generated_image.shape[-2:]

    if H1 != H2 or W1 != W2:
        min_H = min(H1, H2)
        min_W = min(W1, W2)
        original_image = torch.nn.functional.interpolate(original_image, size=(min_H, min_W), mode='bilinear')
        generated_image = torch.nn.functional.interpolate(generated_image, size=(min_H, min_W), mode='bilinear')

    psnr = psnr(original_image, generated_image)

    return psnr.item()

def calculate_psnr_for_folder(original_image_folder, generated_image_folder):
    psnr_list = []
    file_list = glob(os.path.join(generated_image_folder, '*.jpg'))
    for i, file in enumerate(tqdm(file_list)):
        file = os.path.basename(file)
        original_image_path = os.path.join(original_image_folder, file)
        generated_image_path = os.path.join(generated_image_folder, file)
        psnr_score = calculate_psnr(original_image_path, generated_image_path)
        psnr_list.append(psnr_score)
    if len(psnr_list) == 0:
        return 0
    print("psnr score: ", sum(psnr_list) / len(psnr_list))
    return sum(psnr_list) / len(psnr_list)

def calculate_ssim(original_image_path, generated_image_path):
    ssim = StructuralSimilarityIndexMeasure()
    totensor = ToTensor()

    original_image = totensor(Image.open(original_image_path).convert('RGB')).unsqueeze(0)
    generated_image = totensor(Image.open(generated_image_path).convert('RGB')).unsqueeze(0)

    H1, W1 = original_image.shape[-2:]
    H2, W2 = generated_image.shape[-2:]

    if H1 != H2 or W1 != W2:
        min_H = min(H1, H2)
        min_W = min(W1, W2)
        original_image = torch.nn.functional.interpolate(original_image, size=(min_H, min_W), mode='bilinear')
        generated_image = torch.nn.functional.interpolate(generated_image, size=(min_H, min_W), mode='bilinear')

    ssim = ssim(original_image, generated_image)

    return ssim.item()

def calculate_ssim_for_folder(original_image_folder, generated_image_folder):
    ssim_list = []
    file_list = glob(os.path.join(generated_image_folder, '*.jpg'))
    for i, file in enumerate(tqdm(file_list)):
        file = os.path.basename(file)
        original_image_path = os.path.join(original_image_folder, file)
        generated_image_path = os.path.join(generated_image_folder, file)
        ssim_score = calculate_ssim(original_image_path, generated_image_path)
        ssim_list.append(ssim_score)
    if len(ssim_list) == 0:
        return 0
    print("ssim score: ", sum(ssim_list) / len(ssim_list))
    return sum(ssim_list) / len(ssim_list)

if __name__ == '__main__':
    original_image_folder =  '/home/zheedong/Projects/SEED/coco/images/resize_val2014'
    # generated_image_folder = '/home/zheedong/Projects/SEED/results/images/seed_tokenizer_img2img_variation_coco_karpathy_test'
    # generated_image_folder = '/home/zheedong/Projects/SEED/results/images/stable_diffusion_img2img_variation_coco_karpathy_test'
    generated_image_folder = '/home/zheedong/Projects/SEED/results/images/slot_img2img_karpathy_test'
    s = calculate_clip_s_for_folder(original_image_folder, generated_image_folder)
    s = calculate_lpips_for_folder(original_image_folder, generated_image_folder)
    s = calculate_psnr_for_folder(original_image_folder, generated_image_folder)
    s = calculate_ssim_for_folder(original_image_folder, generated_image_folder)