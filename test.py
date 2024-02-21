import hydra
from omegaconf import OmegaConf
import torch
from PIL import Image
from our_tokenizer import SEEDTrainingWrapper
from einops import rearrange

cfg_path = './configs/our_seed_tokenizer.yaml'
cfg = OmegaConf.load(cfg_path)
visual_tokenizer = SEEDTrainingWrapper.load_from_checkpoint('/home/zheedong/Projects/SEED/logs/seed_stage2_proj/lightning_logs/stage2_w_codebook_40epoch/checkpoints/epoch=39-step=7840.ckpt', cfg=cfg, strict=False, map_location="cpu")
visual_tokenizer.eval()

# Set total weight to fp16
for param in visual_tokenizer.image_tokenizer.model.parameters():
    param = param.type(torch.bfloat16)

visual_tokenizer = visual_tokenizer.half()
visual_tokenizer.image_tokenizer = visual_tokenizer.image_tokenizer.half()

visual_tokenizer = visual_tokenizer.to('cuda')

for param in visual_tokenizer.image_tokenizer.model.parameters():
    param.requires_grad = False
    
transform_cfg_path = 'configs/transform/clip_transform.yaml'
transform_cfg = OmegaConf.load(transform_cfg_path)
transform = hydra.utils.instantiate(transform_cfg)
    

## print visual tokenizer weight type
for param in visual_tokenizer.image_tokenizer.model.parameters():
    if param.dtype != torch.bfloat16:
        print("not bfloat16")
        print(param.dtype)
    if param.requires_grad == True:
        print("requires_grad")
        print(param)
        
with torch.no_grad():
    image_path = "images/cat.jpg"
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).to('cuda')
    image_tensor = image_tensor.type(torch.float16)
    img_ids = visual_tokenizer.encode_image(image_torch=image_tensor)

print(img_ids)


save_path_new = "images/cat_new.jpg"

with torch.no_grad():
    images = visual_tokenizer.decode_image(img_ids)
    
images[0].save(save_path_new)


# seed_tokenizer_cfg_path = 'configs/tokenizer/seed_llama_tokenizer_hf.yaml'
# seed_tokenizer_cfg = OmegaConf.load(seed_tokenizer_cfg_path)
# visual_tokenizer  = hydra.utils.instantiate(seed_tokenizer_cfg, device='cuda', load_diffusion=True)
# import pdb; pdb.set_trace()