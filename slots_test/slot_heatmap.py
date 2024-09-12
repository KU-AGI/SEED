from wandb.plots.heatmap import heatmap

from models.slot_attention.utils import ColorMask
from einops import reduce, rearrange
import numpy as np

def get_heatmap(pixel_values, attns):
    colorizer = ColorMask(
        num_slots=32,
        log_img_size=256,
        norm_mean=0,
        norm_std=1,
        reshape_first=True,
    )

    heatmap = colorizer.get_heatmap(
        img=pixel_values,
        attn=reduce(
            attns,
            'b num_h (h w) s -> b s h w',
            h=int(np.sqrt(attns.shape[-2])),
            reduction='mean'
        ),
        recon=[]
    )
    return heatmap

def get_vit_heatmap(pixel_values, attns):
    colorizer = ColorMask(
        num_slots=32,
        log_img_size=224,
        norm_mean=0,
        norm_std=1,
        reshape_first=True,
    )

    # Remove the first token (CLS) from the attention maps
    attns = attns[:, :, :, 1:]

    heatmap = colorizer.get_heatmap(
        img=pixel_values,
        attn=reduce(
            attns,
            'b num_h s (h w) -> b s h w',
            h=int(np.sqrt(attns.shape[-1])),
            reduction='mean'
        ),
        recon=[]
    )
    return heatmap

if __name__ == '__main__':
    import torch
    pixel_values = torch.load("image.pt")
    attns = torch.load("morph_7.pt")
    heatmap = get_vit_heatmap(pixel_values, attns)
    from torchvision.utils import save_image
    save_image(heatmap, "heatmap_morph_7.png")