import os
from typing import Any, List
from tqdm import tqdm
import time
import json
import argparse

import hydra
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, seed_everything
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import PIL
from utils.config import build_config
import pyrootutils
from datamodules import build_datamodule

from models.seed_llama_tokenizer import ImageTokenizer, ImageTokenizer2
import transformers

from pytorch_lightning import loggers as pl_loggers
from functools import partial
from models.seed_qformer.vit import Block
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.utilities import grad_norm

from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from collections import OrderedDict
from calculate_clip_score import calculate_clip_s_for_folder
from datamodules.stage2_datamodule import DataModule


pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

BOI_TOKEN = "<img>"
EOI_TOKEN = "</img>"
IMG_TOKEN = "<img_{:05d}>"

IMG_FLAG = "<image>"
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 8192
image_id_shift = 32000


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        (gt_grad,) = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None



class SEEDTrainingWrapper(LightningModule):
    """Training wrapper for SEED

    Args:
        LightningModule (cfg, model): model should be ImageTokenizer
    """    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # ImageTokenizer model
        # Target model to train
        self.image_tokenizer = ImageTokenizer2(model_path=cfg.checkpoint_path.model_path,
                            fp16=cfg.optimizer.fp16,
                            from_pretrained=True,
                            diffusion_model_path=cfg.checkpoint_path.diffusion_model_path,
                            load_diffusion=True,
                            )

        self.B = None
        
        # My code
        # For make clip embedding directly from [b, 32, 32] to [b, 1024]
        self.depth = 4
        self.embedding_block = nn.ModuleList([
            Block(dim=32,
                    num_heads=16,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6)) for i in range(self.depth)
        ])
        self.embedding_proj = nn.Linear(32 * 32, 1024).to(self.device)

        self.transform_224 = transforms.Resize((224, 224), antialias=True)

        # diffusions
        self.feature_extractor = self.image_tokenizer.diffusion_model.feature_extractor
        self.image_encoder = self.image_tokenizer.diffusion_model.image_encoder
        self.image_normalizer = self.image_tokenizer.diffusion_model.image_normalizer
        self.image_noising_scheduler = self.image_tokenizer.diffusion_model.image_noising_scheduler
        self.tokenizer = self.image_tokenizer.diffusion_model.tokenizer
        self.text_encoder = self.image_tokenizer.diffusion_model.text_encoder
        self.unet = self.image_tokenizer.diffusion_model.unet
        self.scheduler = self.image_tokenizer.diffusion_model.scheduler
        self.vae = self.image_tokenizer.diffusion_model.vae

        # For SDS
        t_range = [0.2, 0.6]
        t_range = [0.02, 0.98]
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.image_noising_scheduler.alphas_cumprod  # for convenience

        # For logging
        self.sample_embed_ind = None
        self.pil_to_tensor = transforms.ToTensor()
        self.sample_image_ind = 0
        self.logged_original_image = set()


    def setup(self, stage):
        for p in self.image_tokenizer.parameters():
            p.requires_grad = False

        # Setup training parameter
        self.image_tokenizer.model.train()
        for param in self.image_tokenizer.model.parameters():
            param.requires_grad = True

        for param in self.image_tokenizer.model.Qformer.parameters():
            param.requires_grad = False

        # Freeze ViT Encoder
        for param in self.image_tokenizer.model.visual_encoder.parameters():
            param.requires_grad = False

        # Diffusion frozen
        for param in self.image_tokenizer.diffusion_model.image_encoder.parameters():
            param.requires_grad = False
        for param in self.image_tokenizer.diffusion_model.image_normalizer.parameters():
            param.requires_grad = False
        for param in self.image_tokenizer.diffusion_model.text_encoder.parameters():
            param.requires_grad = False
        # In this case, unet is frozen
        for param in self.image_tokenizer.diffusion_model.unet.parameters():
            param.requires_grad = False
        for param in self.image_tokenizer.diffusion_model.vae.parameters():
            param.requires_grad = False

        # For fp16
        if self.cfg.optimizer.fp16:
            self.image_tokenizer = self.image_tokenizer.half()
            self.image_encoder = self.image_encoder.half()
            self.embedding_proj = self.embedding_proj.half()
            for blk in self.embedding_block:
                blk = blk.half()

    def load_sample_images(self, dataloader):
        self.sample_images = []
        for idx, batch in enumerate(dataloader):
            if idx > 1:
                break
            else:
                self.sample_images.append(batch.img.to(self.device))
        self.sample_images = torch.cat(self.sample_images, dim=0)

    def get_clip_text_embedding(self, batch_text):
        """CLIP text embedding

        Args:
            batch_text (List): List contains text. [b, 32]

        Returns:
            float: clip text embedding [b, 1024]
        """        
        gt_text_clip_embeddings = []
        with torch.no_grad():
            for idx in range(self.B):
                gt_text_clip_embeddings.append(
                    self.tokenizer(batch_text[idx]).squeeze().to(self.device)
                )
            gt_text_clip_embeddings = torch.stack(gt_text_clip_embeddings, dim=0)

            # gt_img_clip_embeddings = self.model_clip.encode_image(batch.img.to(self.device))
            gt_text_clip_embeddings = self.image_encoder.encode_text(
                gt_text_clip_embeddings.to(self.device)
            )
        return gt_text_clip_embeddings
    
    def get_clip_img_embedding(self, batch_img):
        """CLIP image embedding

        Args:
            batch_img (torch.Tensor): Image tensor [b, 3, 224, 224]

        Returns:
            float: clip image embedding [b, 1024]
        """        
        return self.image_encoder(batch_img).image_embeds.to(self.device)

    def get_causal_embeddings(self, image):
        return self.image_tokenizer.model.get_causal_embeddings(image)

    def get_stage2_quant(self, img):
        # Causal embedding is trained in stage 1.
        with torch.no_grad():
            causal_embeddings = self.get_causal_embeddings(img)

        # [b, 32, 32]
        query_output_down = self.image_tokenizer.model.encode_task_layer(causal_embeddings)

        # For debug
        quant = query_output_down
        loss_embed = None
        embed_ind = None

        # Simple flatten and linear projection
        # Debug : Try to use transformer
        # for blk in self.embedding_block:
        #     quant = blk(quant)
        quant = quant.view(quant.shape[0], -1)
        quant = self.embedding_proj(quant)

        return quant, loss_embed, embed_ind

    def get_original_stage2_quant(self, img):
        causal_embeddings = self.get_causal_embeddings(img)

        # TODO: query_output should be trained to be similar with text embedding
        # Image embedding is cross attentioned.
        # Notice: query_output_down is match to clip embedding?
        # [b, 32, 32]
        query_output_down = self.image_tokenizer.model.encode_task_layer(causal_embeddings)

        # quant [b, 32, 32], loss_embed [b, 32, 768], embed_ind [b, 32]
        quant, loss_embed, embed_ind = self.image_tokenizer.model.quantize(query_output_down)

        #------------------------
        # Stage 2 - 2 : Reconstruction Caual Embedding
        #------------------------

        # quant embedding dimension is [b, 32, 32]
        # decoder_task_layer upscale it to [b, 32, 768]
        # [b, 32, 32] => [b, 32, 768]
        query_output_up = self.image_tokenizer.model.decode_task_layer(quant)

        # Transformer decoder
        query_output_up = self.image_tokenizer.model.get_transformer_decoded_embedding(query_output_up)

        # query_output_up_pos_image should be similar to original causal_embeddings
        # Maximize cosine similarity between query_output_up_pos_image and causal_embeddings

        #------------------------
        # Stage 2 - 3 : Reconstruction Generation Embedding
        #------------------------

        # MLP
        reverse_output_proj = self.image_tokenizer.model.get_mlp_decoded_embedding(query_output_up)

        return reverse_output_proj, loss_embed, embed_ind

    def train_diffusion(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_training_steps: int = 1000,
        guidance_scale: float = 1,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        noise_level: int = 0,
        image_embeds: Optional[torch.FloatTensor] = None,
        negative_image_embeds: Optional[torch.FloatTensor] = None,
        clean_image: Optional[torch.FloatTensor] = None,
    ):
        """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, either `prompt_embeds` will be
                used or prompt is initialized to `""`.
            image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image` or tensor representing an image batch. The image is encoded to its CLIP embedding which the
                `unet` is conditioned on. The image is _not_ encoded by the `vae` and then used as the latents in the
                denoising process like it is in the standard Stable Diffusion text-guided image variation process.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 20):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 10.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            noise_level (`int`, *optional*, defaults to `0`):
                The amount of noise to add to the image embeddings. A higher `noise_level` increases the variance in
                the final un-noised images. See [`StableUnCLIPPipeline.noise_image_embeddings`] for more details.
            image_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated CLIP embeddings to condition the `unet` on. These latents are not used in the denoising
                process. If you want to provide pre-generated latents, pass them to `__call__` as `latents`.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                [`~ pipeline_utils.ImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When returning
                a tuple, the first element is a list with the generated images.
        """
        # image encoding components
        # feature_extractor: CLIPImageProcessor
        # image_encoder: CLIPVisionModelWithProjection

        # # image noising components
        # image_normalizer: StableUnCLIPImageNormalizer
        # image_noising_scheduler: KarrasDiffusionSchedulers

        # # regular denoising components
        # tokenizer: CLIPTokenizer
        # text_encoder: CLIPTextModel
        # unet: UNet2DConditionModel
        # scheduler: KarrasDiffusionSchedulers

        # vae: AutoencoderKL
        # for param in self.
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.image_tokenizer.diffusion_model.vae_scale_factor
        width = width or self.unet.config.sample_size * self.image_tokenizer.diffusion_model.vae_scale_factor

        if prompt is None and prompt_embeds is None:
            # prompt = len(image) * [""] if isinstance(image, list) else ""
            # Changed because we get image_embeds as input
            prompt = image_embeds.shape[0] * [""] if isinstance(image_embeds, torch.Tensor) else ""

        # 1. Check inputs. Raise error if not correct
        self.image_tokenizer.diffusion_model.check_inputs(
            prompt=prompt,
            image=image,
            height=height,
            width=width,
            callback_steps=callback_steps,
            noise_level=noise_level,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            # Only image_embeds is not None, [b, 1024]
            image_embeds=image_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        batch_size = batch_size * num_images_per_prompt

        device = image_embeds.device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = False

        # 3. Encode input prompt
        text_encoder_lora_scale = (cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None)
       
        # [b, 77, 1024] 
        # Now img2img, prompt_embeds is None
        prompt_embeds = self.image_tokenizer.diffusion_model._encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Encoder input image
        # [b, 1024]
        noise_level = torch.tensor([noise_level], device=device)
        image_embeds = self.image_tokenizer.diffusion_model._encode_image(
            image=image,
            device=device,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            noise_level=noise_level,
            generator=generator,
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
        )

        # 5. Prepare timesteps
        # self.scheduler.set_timesteps(num_training_steps, device=device)
        # # 21 length list
        # timesteps = self.scheduler.timesteps

        # Convert images to latent space
        latents = self.vae.encode(clean_image).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        # 5. Prepare timesteps
        # self.scheduler.set_timesteps(bsz, device=device)
        # 21 length list
        # timesteps = self.scheduler.timesteps
        timesteps = torch.randint(0, num_training_steps, (batch_size,), device=device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)

        noisy_latents = self.image_noising_scheduler.add_noise(latents, noise, timesteps)
        noisy_latents = torch.cat([noisy_latents] * 2) if do_classifier_free_guidance else noisy_latents

        # Get the text embedding for conditioning
        encoder_hidden_states = prompt_embeds
        # For debug
        # if self.image_noising_scheduler.config.prediction_type == "epsilon":
        #     target = noise
        # elif self.image_noising_scheduler.config.prediction_type == "v_prediction":
        target = self.image_noising_scheduler.get_velocity(latents, noise, timesteps)

        # Predict the noise residual and compute loss
        # model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        if do_classifier_free_guidance:
            timesteps = torch.cat([timesteps] * 2)

        model_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                class_labels=image_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return loss

    def logging_train(self, quant, loss_embed, gt_img_clip_embeddings, loss):
        self.log(
            "train/generation_embed_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train/codebook_loss_embed",
            loss_embed.mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # Cosine similarity logging
        self.log(
            "train/stage_2_codebook_cosine_similarity",
            F.cosine_similarity(quant, gt_img_clip_embeddings).mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )


    def diff_loss(
        self,
        image_embeds,
        clean_image,
        guidance_scale=100,
        grad_scale=1,
        prompt=None,
        prompt_embeds=None,
    ):
        """Score distillation sampling"""
        if prompt is None and prompt_embeds is None:
            # prompt = len(image) * [""] if isinstance(image, list) else ""
            # Changed because we get image_embeds as input
            prompt = image_embeds.shape[0] * [""] if isinstance(image_embeds, torch.Tensor) else ""

        # 2. Define call parameters
        batch_size = image_embeds.shape[0]

        device = image_embeds.device

        do_classifier_free_guidance = False

        # Convert images to latent space
        # latents = self.vae.encode(clean_image).latent_dist.sample()
        # latents = latents * self.vae.config.scaling_factor

        # 3. Encode input prompt

        # [b, 77, 1024]
        # Now img2img, prompt_embeds is None
        prompt_embeds = self.image_tokenizer.diffusion_model._encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=None,
            lora_scale=None,
        )

        image_embeds = self.image_tokenizer.diffusion_model._encode_image(
            image=None,
            device=device,
            batch_size=batch_size,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            noise_level=0,
            generator=None,
            image_embeds=image_embeds,
            negative_image_embeds=None,
        )

        latents = self.vae.encode(clean_image).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=device
        )

        # add noise
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        # pred noise
        if do_classifier_free_guidance:
            latents_noisy = torch.cat([latents_noisy] * 2)
            t = torch.cat([t] * 2)

        model_pred = self.unet(
                latents_noisy,
                t,
                encoder_hidden_states=prompt_embeds,
                class_labels=image_embeds,
                return_dict=False,
            )[0]
        target = noise
        if do_classifier_free_guidance:
            target = torch.cat([noise] * 2)
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # noise_pred = self.unet(
        #     latent_model_input,
        #     t,
        #     encoder_hidden_states=prompt_embeds,
        #     class_labels=image_embeds,
        # ).sample
        #
        # # perform guidance (high scale from paper!)
        # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #
        # noise_pred = noise_pred_uncond + guidance_scale * (
        #         noise_pred_text - noise_pred_uncond
        # )
        #
        # loss = 0.5 * F.mse_loss(noise_pred, latents.float(), reduction='mean')

        return loss

    def sds(
        self,
        image_embeds,
        clean_image,
        guidance_scale=100,
        grad_scale=1,
        prompt=None,
        prompt_embeds=None,
    ):
        """Score distillation sampling"""
        if prompt is None and prompt_embeds is None:
            # prompt = len(image) * [""] if isinstance(image, list) else ""
            # Changed because we get image_embeds as input
            prompt = image_embeds.shape[0] * [""] if isinstance(image_embeds, torch.Tensor) else ""

        # 2. Define call parameters
        batch_size = image_embeds.shape[0]

        device = image_embeds.device

        # Convert images to latent space
        # latents = self.vae.encode(clean_image).latent_dist.sample()
        # latents = latents * self.vae.config.scaling_factor

        
         # 3. Encode input prompt
       
        # [b, 77, 1024] 
        # Now img2img, prompt_embeds is None
        prompt_embeds = self.image_tokenizer.diffusion_model._encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=None,
            lora_scale=None,
        )

        image_embeds = self.image_tokenizer.diffusion_model._encode_image(
            image=None,
            device=device,
            batch_size=batch_size,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            noise_level=0,
            generator=None,
            image_embeds=image_embeds,
            negative_image_embeds=None,
        )
        do_classifier_free_guidance = True

        latents = self.vae.encode(clean_image).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step, self.max_step + 1, [1], dtype=torch.long, device=device
        )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)

        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            class_labels=image_embeds,
        ).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # w(t), sigma_t^2
        self.alphas = self.alphas.to(device)
        w = 1 - self.alphas[t]
        grad = grad_scale * w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        targets = (latents - grad)
        # loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]
        # Why not mean?
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='mean')

        return loss
    
    def logging_val(self, quant, loss_embed, gt_img_clip_embeddings, loss):
        self.log(
            "val/generation_embed_loss",
            loss,
            sync_dist=True,
        )

        self.log(
            "val/codebook_loss_embed",
            loss_embed.mean(),
            sync_dist=True,
        )
        # Cosine similarity logging
        self.log(
            "val/stage_2_codebook_cosine_similarity",
            F.cosine_similarity(quant, gt_img_clip_embeddings).mean(),
            sync_dist=True,
        )
    
    def training_step(self, batch, batch_idx: int):
        self.B = batch.img.shape[0]
        # gt_text is a list of string
        # Encoding text in list to ascii
        batch.gt_txt = [[text[0].encode("ascii", "ignore").decode()] for text in batch.gt_txt]

        clip_size_image = self.transform_224(batch.img)
        # For cosine similarity logging
        # gt_txt_clip_embeddings = self.get_clip_text_embedding(batch.gt_txt)
        gt_img_clip_embeddings = self.get_clip_img_embedding(clip_size_image.float())
        with torch.no_grad():
            causal_embeddings = self.get_causal_embeddings(clip_size_image)

        # TODO: query_output should be trained to be similar with text embedding
        # Image embedding is cross attentioned.
        # Notice: query_output_down is match to clip embedding?
        # [b, 32, 32]
        query_output_down = self.image_tokenizer.model.encode_task_layer(causal_embeddings)

        # quant [b, 32, 32], loss_embed [b, 32, 768], embed_ind [b, 32]
        quant, loss_embed, embed_ind = self.image_tokenizer.model.quantize(query_output_down)
        loss_embed = loss_embed.mean()

        #------------------------
        # Stage 2 - 2 : Reconstruction Caual Embedding
        #------------------------

        # quant embedding dimension is [b, 32, 32]
        # decoder_task_layer upscale it to [b, 32, 768]
        # [b, 32, 32] => [b, 32, 768]
        query_output_up = self.image_tokenizer.model.decode_task_layer(quant)

        # Transformer decoder
        query_output_up = self.image_tokenizer.model.get_transformer_decoded_embedding(query_output_up)

        # query_output_up_pos_image should be similar to original causal_embeddings
        # Maximize cosine similarity between query_output_up_pos_image and causal_embeddings

        #------------------------
        # Stage 2 - 3 : Reconstruction Generation Embedding
        #------------------------

        # MLP
        image_embeds = self.image_tokenizer.model.get_mlp_decoded_embedding(query_output_up)

        # loss_sds = self.sds(
        #     image_embeds=image_embeds,
        #     clean_image=clip_size_image.float(),
        #     guidance_scale=10,
        #     grad_scale=1,
        # )

        loss_sds = self.diff_loss(
            image_embeds=image_embeds,
            clean_image=clip_size_image.float(),
            guidance_scale=10,
            grad_scale=1,
        )

        loss_recon = F.mse_loss(image_embeds, gt_img_clip_embeddings)

        self.log(
            "train/loss_sds",
            loss_sds,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train/loss_recon",
            loss_recon,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train/loss_embed",
            loss_embed,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train/clip_cosine_similarity",
            F.cosine_similarity(image_embeds, gt_img_clip_embeddings).mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )


        # stage_2_loss = self.get_stage_2_loss(batch, batch_idx)
        # stage_diffusion_loss, img_embeds = self.get_stage_diffusion_loss(batch, batch_idx)

        # gt_img_clip_embeddings = self.get_clip_img_embedding(batch.img)
        # loss = F.mse_loss(quant, gt_img_clip_embeddings)
        
        loss = loss_sds + loss_recon + loss_embed
        # params = OrderedDict(self.image_tokenizer.model.quantize.named_parameters())
        # grads = torch.autograd.grad(loss, params.values(), allow_unused=True)
        # print(grads)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int):
        image, captions, image_id = batch
        gt_img_clip_embeddings = self.get_clip_img_embedding(image.float())

        image = self.transform_224(image)
        image_embeds, _, _ = self.get_original_stage2_quant(image)

        self.log(
            "valid/clip_cosine_similarity",
            F.cosine_similarity(image_embeds, gt_img_clip_embeddings).mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        reconstructed_images = self.image_tokenizer.diffusion_model(
            image_embeds=image_embeds,
            negative_image_embeds=None,
            guidance_scale=10,
            noise_level=0,
            latents=self.image_tokenizer.latents,
        ).images
        # save image
        save_path = f"{self.cfg.result_path}/{self.global_step}"
        os.makedirs(save_path, exist_ok=True)

        for i, img in enumerate(reconstructed_images):
            img.save(f"{save_path}/{image_id[i]}")


        #
        # tensor_images = []
        # for img in reconstructed_images:
        #     tensor_images.append(self.pil_to_tensor(img).unsqueeze(0))
        # tensor_images = torch.cat(tensor_images, dim=0)
        #
        # # Check if image is already logged
        # if batch_idx not in self.logged_original_image:
        #     self.logger.experiment.add_images(
        #         f"original/image_batch_{batch_idx}",
        #         batch.img,
        #     )
        #
        #     self.logger.experiment.add_images(
        #         f"original/image_batch_{batch_idx}_seed_reconstructed",
        #         tensor_images,
        #     )
        #
        #     # logging original caption
        #     self.logger.experiment.add_text(
        #         f"original/gt_text_image_batch_{batch_idx}",
        #         batch.gt_txt[0][0],
        #     )
        #
        #     self.logged_original_image.add(batch_idx)
        # else:
        #     self.logger.experiment.add_images(
        #         f"images/image_batch_{batch_idx}",
        #         tensor_images,
        #         global_step=self.sample_image_ind,
        #     )
        #     self.sample_image_ind += 1

        # logging weight distribution to check if weight is updated (gradient is flowing)
        # self.logger.experiment.add_histogram(
        #     "weight_distribution/image_tokenizer/model/Qformer/bert/encoder/layer/0/attention/self/value",
        #     next(self.image_tokenizer.model.Qformer.bert.encoder.layer[0].attention.self.value.parameters()),
        #     global_step=self.global_step,
        # )
        #
        # self.logger.experiment.add_histogram(
        #     "weight_distribution/image_tokenizer/model/Qformer/bert/encoder/layer/1/attention/self/value",
        #     next(self.image_tokenizer.model.Qformer.bert.encoder.layer[1].attention.self.value.parameters()),
        #     global_step=self.global_step,
        # )
        #
        # self.logger.experiment.add_histogram(
        #     "weight_distribution/image_tokenizer/model/Qformer/bert/encoder/layer/7/attention/self/value",
        #     next(self.image_tokenizer.model.Qformer.bert.encoder.layer[7].attention.self.value.parameters()),
        #     global_step=self.global_step,
        # )
        #
        # self.logger.experiment.add_histogram(
        #     "weight_distribution/embedding_proj",
        #     next(self.embedding_proj.parameters()),
        #     global_step=self.global_step,
        # )

    def configure_optimizers(self):
        if self.trainer.max_steps != -1:
            num_training_steps = self.trainer.max_steps
        else:
            limit_batches = self.trainer.limit_train_batches

            dataset = self.trainer._data_connector._train_dataloader_source.dataloader()
            batches = len(dataset)

            batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

            num_devices = max(1, self.trainer.num_devices)
            effective_accum = self.trainer.accumulate_grad_batches * num_devices
            num_training_steps = (batches // effective_accum) * self.trainer.max_epochs

        print(f"#### num_training_steps: {num_training_steps}")
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-4)
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=50, num_training_steps=num_training_steps)

        lr_scheduler_config = {
            "scheduler": scheduler,
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config,
                "gradient_clip_val": self.cfg.optimizer.grad_clip_val,}
        
    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        # {'grad_2.0_norm/weight': 0.0003, 'grad_2.0_norm/bias': 0.0, 'grad_2.0_norm_total': 0.0003}
        norms_0 = grad_norm(self.image_tokenizer.model.Qformer.bert.encoder.layer[0].attention.self.value, norm_type=2)
        for norm in norms_0.keys():
            self.logger.experiment.add_scalar(
                f"grad_norm/image_tokenizer/model/Qformer/bert/encoder/layer/0/attention/self/value/{norm}",
                norms_0[norm],
                global_step=self.global_step,
            )
        norms_1 = grad_norm(self.image_tokenizer.model.Qformer.bert.encoder.layer[1].attention.self.value, norm_type=2)
        for norm in norms_1.keys():
            self.logger.experiment.add_scalar(
                f"grad_norm/image_tokenizer/model/Qformer/bert/encoder/layer/1/attention/self/value/{norm}",
                norms_1[norm],
                global_step=self.global_step,
            )
        norms_7 = grad_norm(self.image_tokenizer.model.Qformer.bert.encoder.layer[7].attention.self.value, norm_type=2)
        for norm in norms_7.keys():
            self.logger.experiment.add_scalar(
                f"grad_norm/image_tokenizer/model/Qformer/bert/encoder/layer/7/attention/self/value/{norm}",
                norms_7[norm],
                global_step=self.global_step,
            )
        norms_proj = grad_norm(self.embedding_proj, norm_type=2)
        for norm in norms_proj.keys():
            self.logger.experiment.add_scalar(
                f"grad_norm/embedding_proj/{norm}",
                norms_proj[norm],
                global_step=self.global_step,
            )

    def on_validation_epoch_end(self):
        original_image_dir = '/ssd0/data/coco/images/val2014'
        generated_image_dir = f"{self.cfg.result_path}/{self.global_step}"
        clip_score = calculate_clip_s_for_folder(original_image_dir, generated_image_dir)

        print(f"clip score: {clip_score}")
        self.log_dict({
            'clip_score': clip_score,
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    cfg, cfg_yaml = build_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.tokenizer_cfg_path = "configs/tokenizer/seed_llama_tokenizer.yaml"
    seed_everything(cfg.experiment.seed, workers=True)

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    os.makedirs(cfg.result_file_path, exist_ok=True)

    datamodule = DataModule(
        cfg=cfg,
        train_transform=transform,
        val_transform=transform,
        pin_memory=False,
        epoch=cfg.experiment.max_epochs,
        total_gpus=cfg.dist.n_gpus,
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.result_file_path)
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        accelerator=device,
        num_nodes=cfg.dist.n_nodes,
        devices=cfg.dist.n_gpus,
        # strategy=DDPStrategy(
        #     find_unused_parameters=True,
        #     ddp_comm_hook=default_hooks.fp16_compress_hook
        #     if cfg.optimizer.fp16_grad_comp
        #     else None,
        # ),
        strategy="ddp",
        max_epochs=cfg.experiment.max_epochs,
        deterministic=False,
        logger=tb_logger,
        log_every_n_steps=1,
        val_check_interval=cfg.experiment.val_check_interval,
        # check_val_every_n_epoch=cfg.experiment.check_val_every_n_epoch,
        enable_checkpointing=True,
        # Debug
        num_sanity_val_steps=0,
        # overfit_batches=cfg.experiment.overfit_batches,
        callbacks=[ModelSummary(max_depth=3), lr_logger],
        accumulate_grad_batches=cfg.experiment.grad_accumulation
    )

    wrapper = SEEDTrainingWrapper(cfg).to(device)

    trainer.fit(model=wrapper, datamodule=datamodule)
    trainer.strategy.barrier()
