import os
from typing import Any, List
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

from tqdm import tqdm
import time
import json

import argparse

import argparse
import time

import hydra
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, seed_everything
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import PIL
from torchvision.transforms.functional import to_pil_image
from utils.config import build_config
import pyrootutils
from datamodules import build_datamodule
from datamodules.tokenizers import TokenizerUtils

from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
import torch.nn.functional as F
from pytorch_lightning.strategies import DDPStrategy, DDPFullyShardedStrategy
import open_clip
from info_nce import InfoNCE, info_nce
from models.seed_llama_tokenizer import ImageTokenizer
import transformers

from pytorch_lightning import loggers as pl_loggers
from functools import partial
from models.seed_qformer.vit import Block
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.utilities import grad_norm

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from PIL import Image
import pdb

from coco_dataloader import CocoDataset
from torch.utils.data import DataLoader
from calculate_clip_score import calculate_clip_s_for_folder

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

BOI_TOKEN = "<img>"
EOI_TOKEN = "</img>"
IMG_TOKEN = "<img_{:05d}>"

IMG_FLAG = "<image>"
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 8192
IMAGE_ID_SHIFT = 32000

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
        self.image_tokenizer = ImageTokenizer(model_path=cfg.checkpoint_path.model_path,
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

        # For logging
        self.sample_embed_ind = None
        self.pil_to_tensor = transforms.ToTensor()
        self.sample_image_ind = 0
        self.logged_original_image = set()
        
        self.stage = 2
    
    def random_initialize_stage2_model_weights(self):
        """Random initialize stage 2 model weights
        """        
        # Random initialize stage 2 model weights
        for param in self.image_tokenizer.model.parameters():
            param.requires_grad = False

        # For fp16
        if self.cfg.optimizer.fp16:
            self.image_tokenizer = self.image_tokenizer.half()
            self.image_encoder = self.image_encoder.half()
            self.embedding_proj = self.embedding_proj.half()
            for blk in self.embedding_block:
                blk = blk.half()

    def setup(self, stage):
        # Setup training parameter
        self.image_tokenizer.model.train()
        for param in self.image_tokenizer.model.parameters():
            param.requires_grad = True

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
            
        if self.stage == 2:
            for param in self.image_tokenizer.model.parameters():
                param.requires_grad = False
                
            # unFreeze stage 2 model and initialize with random weights
            # for param in self.image_tokenizer.model.encode_task_layer.parameters():
            #     #nn.init.xavier_uniform_(param) 
            #     nn.init.normal_(param, mean=0.0, std=0.02)              
            #     param.requires_grad = True 
            # for param in self.image_tokenizer.model.quantize.parameters():
            #     nn.init.normal_(param, mean=0.0, std=0.02)
            #     param.requires_grad = True
            # for param in self.image_tokenizer.model.decode_task_layer.parameters():
            #     nn.init.normal_(param, mean=0.0, std=0.02)
            #     param.requires_grad = True
            
            for param in self.image_tokenizer.model.blocks_image.parameters():
                nn.init.normal_(param, mean=0.0, std=0.02)
                param.requires_grad = True
            
            for param in self.image_tokenizer.model.image_down.parameters():
                nn.init.normal_(param, mean=0.0, std=0.02)
                param.requires_grad = True
            for param in self.image_tokenizer.model.distill_image_proj.parameters():
                nn.init.normal_(param, mean=0.0, std=0.02)
                param.requires_grad = True
            
        ## make dump folder
        os.makedirs(self.cfg.result_path, exist_ok=True)

        # For fp16
        if self.cfg.optimizer.fp16:
            self.image_tokenizer = self.image_tokenizer.half()
            self.image_encoder = self.image_encoder.half()
            self.embedding_proj = self.embedding_proj.half()
            for blk in self.embedding_block:
                blk = blk.half()
        
        # For test training
        # self.image_tokenizer.model.distill_image_proj = nn.Linear(32 * 32, 1024).to(self.device)
    
    def on_validation_epoch_start(self):
        os.makedirs(f"{self.cfg.result_path}/{self.current_epoch}", exist_ok=True)

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

            '''
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
            '''
            quant = None
            loss_embed = None
            embed_ind = None
                        
            query_output_up = self.image_tokenizer.model.get_transformer_decoded_embedding(causal_embeddings)
            
            quant = self.image_tokenizer.model.get_mlp_decoded_embedding(query_output_up)

            return quant, loss_embed, embed_ind

    def get_original_stage2_quant(self, img):
        causal_embeddings = self.get_causal_embeddings(img)
        
        ''' bypass
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
        '''
        query_output_up = causal_embeddings
        reverse_output_proj = self.image_tokenizer.model.get_mlp_decoded_embedding(query_output_up)

        return reverse_output_proj #, loss_embed, embed_ind


    def get_stage_1_loss(self, batch, batch_idx: int, is_validation=False):
        """_summary_

        Args:
            batch (_type_): _description_
            batch_idx (int): _description_
            is_validation (bool, optional): _description_. Defaults to False.
        """
        device = batch.img.device
        image = self.transform_224(batch.img)
        text = [text[0].encode("ascii", "ignore").decode() for text in batch.gt_txt]
        with torch.no_grad():
            image_embeds = self.image_tokenizer.model.ln_vision(
                self.image_tokenizer.model.visual_encoder(image)
            )
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.image_tokenizer.model.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.image_tokenizer.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats = F.normalize(
            self.image_tokenizer.model.vision_proj(
                query_output.last_hidden_state), dim=-1
        )

        text_tokens = self.image_tokenizer.model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        text_output = self.image_tokenizer.model.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )

        text_feat = F.normalize(
            self.image_tokenizer.model.text_proj(
                text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        ###============== Image-text Contrastive ===================###
        sim_q2t = torch.matmul(
            image_feats.unsqueeze(1), text_feat.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.image_tokenizer.model.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.image_tokenizer.model.temp  # [batch_size, batch_size*num_gpu]

        bs = image.size(0)
        targets = torch.arange(bs, dtype=torch.long).to(device)

        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        self.log(
                "train/loss_itc",
                loss_itc,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return loss_itc

    def forward_stage_2(self,batch, batch_idx: int):
        """_summary_
        Original forward function for stage 2        

        Args:
            batch (_type_): _description_
            batch_idx (int): _description_

        Returns:
            _type_: _description_
        """        

        # # Causal embedding is trained in stage 1.
        # # [b, 32, 768]
        # causal_embeddings = self.get_causal_embeddings(batch.img)

        # # [b, 32, 768] = > [b, 32, 32]
        # query_output_down = self.image_tokenizer.model.encode_task_layer(causal_embeddings)

        # # quant [b, 32, 32], loss_embed [b, 32, 768], embed_ind [b, 32]
        # quant, loss_embed, embed_ind = self.image_tokenizer.model.quantize(query_output_down)

        # embed_ind = embed_ind.reshape(quant.shape[0], -1)

        # # [b, 32, 32] => [b, 32, 768]
        # query_output_up = self.image_tokenizer.model.decode_task_layer(quant)

        # quant_embedding = self.image_tokenizer.model.quantize.get_codebook_entry(embed_ind)

        # # [b, 32, 32] => [b, 32, 768]
        # query_output_up = self.image_tokenizer.model.decode_task_layer(quant_embedding)

        # [b, 32, 768] => [b, 32, 768]
        query_output_up = self.image_tokenizer.model.get_transformer_decoded_embedding(query_output_up)

        # [b, 32, 768] => [b, 32, 32] => [b, 1024]
        reverse_output_proj = self.image_tokenizer.model.get_mlp_decoded_embedding(query_output_up)

        return reverse_output_proj

    def make_image_from_image_embedding_and_save(self, image_embedding, image_id, save_folder):
        with torch.no_grad():
            reconstructed_images = self.image_tokenizer.diffusion_model(
                image_embeds=image_embedding,
                negative_image_embeds=None,
                guidance_scale=10,
                noise_level=0,
                latents=self.image_tokenizer.latents,
            ).images
            
            # save image
            reconstructed_images[0].save(f"{save_folder}/{image_id}")
        

    def get_stage_2_loss(self, batch, batch_idx: int, is_validation=False):
        """_summary_

        Args:
            batch (_type_): _description_
            batch_idx (int): _description_
        """        
        #------------------------
        # Stage 2 Training
        #------------------------
        img = self.transform_224(batch.img)

        #------------------------
        # Stage 2 - 1 : Codebook Training
        #------------------------
        causal_embeddings = self.get_causal_embeddings(img)

        # # TODO: query_output should be trained to be similar with text embedding
        # # Image embedding is cross attentioned.
        # # Notice: query_output_down is match to clip embedding?
        # # [b, 32, 32]
        # query_output_down = self.image_tokenizer.model.encode_task_layer(causal_embeddings)

        # # quant [b, 32, 32], loss_embed [b, 32, 768], embed_ind [b, 32]
        # quant, loss_embed, embed_ind = self.image_tokenizer.model.quantize(query_output_down)
        

        #------------------------
        # Stage 2 - 2 : Reconstruction Caual Embedding
        #------------------------

        # # quant embedding dimension is [b, 32, 32]
        # # decoder_task_layer upscale it to [b, 32, 768]
        # # [b, 32, 32] => [b, 32, 768]
        # query_output_up = self.image_tokenizer.model.decode_task_layer(causal_embeddings)

        # Transformer decoder
        query_output_up = self.image_tokenizer.model.get_transformer_decoded_embedding(causal_embeddings)
        
        

        # query_output_up_pos_image should be similar to original causal_embeddings
        # Maximize cosine similarity between query_output_up_pos_image and causal_embeddings

        #loss_recon = F.cosine_similarity(query_output_up, causal_embeddings).mean()        
        
        #------------------------
        # Stage 2 - 3 : Reconstruction Generation Embedding
        #------------------------

        # MLP
        # query_output_up = causal_embeddings
        reverse_output_proj = self.image_tokenizer.model.get_mlp_decoded_embedding(query_output_up)

        gt_img_clip_embeddings = self.get_clip_img_embedding(img)
    
        loss_generation_embed = F.mse_loss(reverse_output_proj, gt_img_clip_embeddings)

        #loss_total = loss_embed - loss_recon + loss_generation_embed
        loss_total = loss_generation_embed
        loss_total = loss_total.mean()

        # loss_dict = {"loss_embed": loss_embed, "loss_recon": loss_recon,
        #         "loss_generation_embed": loss_generation_embed,
        #         "loss": loss_total}
        
        loss_dict = {"loss_generation_embed": loss_generation_embed,
                     "loss": loss_total}

        #------------------------
        # Logging
        #------------------------
        #generation_embedding_cosine_similarity = F.cosine_similarity(reverse_output_proj, gt_img_clip_embeddings).mean()

        #self.logging_train(generation_embedding_cosine_similarity, loss_dict)
        self.logging_train(None, loss_dict)

        return loss_total

    def logging_train(self, generation_embedding_cosine_similarity, loss_dict):
        # self.log(
        #     "train/generation_embedding_cosine_similarity",
        #     generation_embedding_cosine_similarity,
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        # )

        # self.log(
        #     "train/codebook_loss_embed",
        #     loss_dict["loss_embed"].mean(),
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        # )

        # self.log(
        #     "train/reconstruction_loss",
        #     loss_dict["loss_recon"].mean(),
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        # )

        self.log(
            "train/generation_embed_loss",
            loss_dict["loss_generation_embed"].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
    
    def logging_val(self, quant, loss_embed, gt_img_clip_embeddings, loss):
        self.log(
            "val/generation_embed_loss",
            loss,
            sync_dist=True,
        )

        # self.log(
        #     "val/codebook_loss_embed",
        #     loss_embed.mean(),
        #     sync_dist=True,
        # )
        # # Cosine similarity logging
        # self.log(
        #     "val/stage_2_codebook_cosine_similarity",
        #     F.cosine_similarity(quant, gt_img_clip_embeddings).mean(),
        #     sync_dist=True,
        # )
    
    def training_step(self, batch, batch_idx: int):
        self.B = batch.img.shape[0]
        
        # gt_text is a list of string
        # Encoding text in list to ascii
        #batch.gt_txt = [[text[0].encode("ascii", "ignore").decode()] for text in batch.gt_txt]

        #stage_1_loss = self.get_stage_1_loss(batch, batch_idx)
        
        stage_2_loss = self.get_stage_2_loss(batch, batch_idx)
        
        # mock loss 1
        #stage_2_loss = torch.tensor(0.1979, requires_grad=True)

        #return stage_1_loss
        return stage_2_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int, save_path=None):
        image, captions, image_id = batch
        #image = self.transform_224(batch.img)
        image_embeds, _, _ = self.get_stage2_quant(image)
        
        with torch.no_grad():
            reconstructed_images = self.image_tokenizer.diffusion_model(
                image_embeds=image_embeds,
                negative_image_embeds=None,
                guidance_scale=10,
                noise_level=0,
                latents=self.image_tokenizer.latents,
            ).images
            
            # save image
            if save_path is None:
                save_path = f"{self.cfg.result_path}/{self.current_epoch}"
            reconstructed_images[0].save(f"{save_path}/{image_id[0]}")

            # tensor_images = []
            # for img in reconstructed_images:
            #     tensor_images.append(self.pil_to_tensor(img).unsqueeze(0))
            # tensor_images = torch.cat(tensor_images, dim=0)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-8)
        #scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=100, num_training_steps=5000)
        num_training_steps = self.cfg.experiment.max_epochs * (1000000 / 4 / 1024)
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)


        lr_scheduler_config = {
            "scheduler": scheduler,
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config,
                "gradient_clip_val": self.cfg.optimizer.grad_clip_val,}
        
    def on_validation_epoch_end(self):
        original_image_dir = '/ssd0/data/coco/images/val2014'
        generated_image_dir = f"{self.cfg.result_path}/{self.current_epoch}"
        clip_score = calculate_clip_s_for_folder(original_image_dir, generated_image_dir)
        
        print(f"clip score: {clip_score}")
        self.log_dict({
            'clip_score': clip_score,
        },on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.log(
            "clip_score_coco_karpathy",
            clip_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        

if __name__ == "__main__":
    cfg, cfg_yaml = build_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # cfg.tokenizer_cfg_path = "configs/tokenizer/seed_llama_tokenizer.yaml"
    seed_everything(cfg.experiment.seed, workers=True)

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    os.makedirs(cfg.result_file_path, exist_ok=True)

    datamodule = build_datamodule(
        cfg=cfg,
        train_transform=transform,
        val_transform=transform,
        pin_memory=False,
        epoch=cfg.experiment.max_epochs,
        total_gpus=cfg.dist.n_gpus,
    )

    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    #val_dataloader = datamodule.val_dataloader()
    
    karpathy_file = '/ssd0/data/coco/annotations/karpathy/dataset_coco_test.json'
    root_dir = '/ssd0/data/coco/images/val2014'
    start_index = 0
    end_index = 256
    val_dataset = CocoDataset(root_dir, karpathy_file, tokenizer=None, start_index=start_index, end_index=end_index)
    val_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=val_dataset.collate_fn, num_workers=4)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.result_file_path)
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        accelerator=device,
        num_nodes=cfg.dist.n_nodes,
        devices=cfg.dist.n_gpus,
        strategy=DDPStrategy(
            find_unused_parameters=True,
            # ddp_comm_hook=default_hooks.fp16_compress_hook
            # if cfg.optimizer.fp16_grad_comp
            # else None,
        ),
        #strategy="ddp",
        max_epochs=cfg.experiment.max_epochs,
        deterministic=True,
        logger=tb_logger,
        log_every_n_steps=1,
        # val_check_interval=cfg.experiment.val_check_interval,
        # check_val_every_n_epoch=cfg.experiment.check_val_every_n_epoch,
        enable_checkpointing=cfg.experiment.enable_checkpointing,
        #enable_checkpointing=True,
        # Debug
        num_sanity_val_steps=0,
        precision='bf16',
        # overfit_batches=cfg.experiment.overfit_batches,
        callbacks=[ModelSummary(max_depth=3), lr_logger],
        accumulate_grad_batches=cfg.experiment.grad_accumulation
    )

    #wrapper = SEEDTrainingWrapper(cfg).to(device)
    # checkpoint_path = '/home/zheedong/Projects/SEED/logs/seed_stage_1_training_debug/lightning_logs/version_50_stage_1_final_token_init_weight_new_version/checkpoints/epoch=31-step=25024.ckpt'
    # wrapper = SEEDTrainingWrapper.load_from_checkpoint(checkpoint_path, cfg=cfg, strict=False).to(device)
    # wrapper.setup("fit")
    
    wrapper = SEEDTrainingWrapper(cfg).to(device)
    wrapper.setup("fit")

    trainer.fit(
        wrapper, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
    )
    trainer.strategy.barrier()
