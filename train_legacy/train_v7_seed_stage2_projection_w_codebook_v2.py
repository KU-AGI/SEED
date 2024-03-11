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
from models.seed_llama_tokenizer import ImageTokenizer
import transformers

from pytorch_lightning import loggers as pl_loggers
from functools import partial
from models.seed_qformer.vit import Block
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.utilities import grad_norm

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

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
                            vit_precision=cfg.optimizer.vit_precision,
                            diffusion_precision=cfg.optimizer.diffusion_precision,
                            from_pretrained=True,
                            diffusion_model_path=cfg.checkpoint_path.diffusion_model_path,
                            load_diffusion=True,
                            )

        self.B = None

        self.transform_224 = transforms.Resize((224, 224), antialias=True)

        # For diffusion DDP
        if self.image_tokenizer.diffusion_model is not None:
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
        self.pil_to_tensor = transforms.ToTensor()
        self.sample_image_ind = 0
        self.logged_original_image = set()
        
        self.stage = cfg.experiment.stage
    
    def random_initialize_stage2_model_weights(self):
        """Random initialize stage 2 model weights
        """        
        # Random initialize stage 2 model weights
        for param in self.image_tokenizer.model.parameters():
            param.requires_grad = False
        
        # unFreeze stage 2 model and initialize with random weights
        for param in self.image_tokenizer.model.encode_task_layer.parameters():
            #nn.init.xavier_uniform_(param) 
            nn.init.normal_(param, mean=0.0, std=0.02)              
            param.requires_grad = True 
        for param in self.image_tokenizer.model.quantize.parameters():
            nn.init.normal_(param, mean=0.0, std=0.02)
            param.requires_grad = True
        for param in self.image_tokenizer.model.decode_task_layer.parameters():
            nn.init.normal_(param, mean=0.0, std=0.02)
            param.requires_grad = True
        
        for param in self.image_tokenizer.model.blocks_image.parameters():
            nn.init.normal_(param, mean=0.0, std=0.02)
            param.requires_grad = True
        
        for param in self.image_tokenizer.model.image_down.parameters():
            nn.init.normal_(param, mean=0.0, std=0.02)
            param.requires_grad = True
        for param in self.image_tokenizer.model.distill_image_proj.parameters():
            nn.init.normal_(param, mean=0.0, std=0.02)
            param.requires_grad = True
        
    def save_config(self):
        config_save_path = os.path.join(self.logger.log_dir, "config.yaml")
        with open(config_save_path, "w") as f:
            json.dump(self.cfg, f, indent=4)
    
    def on_train_start(self):
        print("Save config")
        self.save_config()

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
            self.random_initialize_stage2_model_weights()
            
        ## make dump folder
        os.makedirs(self.cfg.result_path, exist_ok=True)
    
    def on_validation_epoch_start(self):
        os.makedirs(f"{self.cfg.result_path}/{self.current_epoch}", exist_ok=True)

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
            
            query_output_down = self.image_tokenizer.model.encode_task_layer(causal_embeddings)

            # quant [b, 32, 32], loss_embed [b, 32, 768], embed_ind [b, 32]
            quant, loss_embed, embed_ind = self.image_tokenizer.model.quantize(query_output_down)
            
            # bypass code book
            # quant = query_output_down
            
            query_output_up = self.image_tokenizer.model.decode_task_layer(quant)
            
            query_output_up = self.image_tokenizer.model.get_transformer_decoded_embedding(query_output_up)
            
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
        device = self.device
        img = batch.img.to(device)

        #------------------------
        # Stage 2 - 1 : Codebook Training
        #------------------------
        with torch.no_grad():
            causal_embeddings = self.get_causal_embeddings(img)

        # TODO: query_output should be trained to be similar with text embedding
        # Image embedding is cross attentioned.
        # Notice: query_output_down is match to clip embedding?
        # [b, 32, 32]
        query_output_down = self.image_tokenizer.model.encode_task_layer(causal_embeddings)

        # quant [b, 32, 32], loss_embed [b, 32, 768], embed_ind [b, 32]
        quant, loss_embed, embed_ind = self.image_tokenizer.model.quantize(query_output_down)
        
        # bypass code book
        # quant = query_output_down
        

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

        #loss_recon = F.cosine_similarity(query_output_up, causal_embeddings).mean()
        loss_recon = F.mse_loss(query_output_up, causal_embeddings)
        
        
        #------------------------
        # Stage 2 - 3 : Reconstruction Generation Embedding
        #------------------------

        # MLP
        # query_output_up = causal_embeddings
        reverse_output_proj = self.image_tokenizer.model.get_mlp_decoded_embedding(query_output_up)

        gt_img_clip_embeddings = self.get_clip_img_embedding(img)
    
        loss_generation_embed = F.mse_loss(reverse_output_proj, gt_img_clip_embeddings)

        loss_total = loss_embed + loss_recon + loss_generation_embed
        loss_total = loss_total.mean()

        # loss_dict = {"loss_embed": loss_embed, "loss_recon": loss_recon,
        #         "loss_generation_embed": loss_generation_embed,
        #         "loss": loss_total}
        
        loss_dict = {"loss_generation_embed": loss_generation_embed,
                     "loss_embed": loss_embed,
                     "loss_recon": loss_recon,
                     "loss": loss_total}

        #------------------------
        # Logging
        #------------------------
        #generation_embedding_cosine_similarity = F.cosine_similarity(reverse_output_proj, gt_img_clip_embeddings).mean()

        #self.logging_train(generation_embedding_cosine_similarity, loss_dict)
        self.logging_train(None, loss_dict)

        return loss_total

    def logging_train(self, generation_embedding_cosine_similarity, loss_dict):
        self.log(
            "train/generation_embedding_mse_loss",
            loss_dict["loss_generation_embed"].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train/codebook_loss_embed",
            loss_dict["loss_embed"].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train/reconstruction_loss",
            loss_dict["loss_recon"].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train/total_loss",
            loss_dict["loss"].mean(),
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
        
        stage_2_loss = self.get_stage_2_loss(batch, batch_idx)
        
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
            
        if self.logger is not None and isinstance(self.logger, pl_loggers.TensorBoardLogger):
            tb_log_dir = self.logger.log_dir
        else:
            tb_log_dir = self.cfg.result_path
        
        save_path = f"{tb_log_dir}/images/version_{self.logger.version}/epoch_{self.current_epoch}/images"
        os.makedirs(save_path, exist_ok=True)

        tensor_images = []

        for img, cur_id in zip(reconstructed_images, image_id):
            # save PIL image to save_path
            img.save(f"{save_path}/{cur_id}")

            # For tensorboard logging
            tensor_images.append(self.pil_to_tensor(img).unsqueeze(0))

        tensor_images = torch.cat(tensor_images, dim=0)

        # Check if image is already logged
        if batch_idx not in self.logged_original_image:
            self.logger.experiment.add_images(
                f"original/image_batch_{batch_idx}",
                image,
            )

            self.logger.experiment.add_images(
                f"original/image_batch_{batch_idx}_seed_reconstructed",
                tensor_images,
            )

            # logging original caption
            for caption in captions:
                self.logger.experiment.add_text(
                    f"original/gt_text_image_batch_{batch_idx}",
                    caption,
                )

            self.logged_original_image.add(batch_idx)
        else:
            self.logger.experiment.add_images(
                f"images/image_batch_{batch_idx}",
                tensor_images,
                global_step=self.sample_image_ind,
            )
            self.sample_image_ind += 1


    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-8)
        lr = self.cfg.optimizer.max_lr
        betas = (self.cfg.hyperparameters.beta_1, self.cfg.hyperparameters.beta_2)
        weight_decay = self.cfg.hyperparameters.weight_decay
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

        #scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=100, num_training_steps=5000)
        num_training_steps = self.cfg.experiment.total_training_steps
        num_warmup_steps = self.cfg.experiment.num_warmup_steps
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )


        lr_scheduler_config = {
            "scheduler": scheduler,
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config,}
        
    def on_validation_epoch_end(self):
        if self.logger is not None and isinstance(self.logger, pl_loggers.TensorBoardLogger):
            tb_log_dir = self.logger.log_dir
        else:
            tb_log_dir = self.cfg.result_path

        original_image_dir = self.cfg.root_dir
        generated_image_dir = f"{tb_log_dir}/images/version_{self.logger.version}/epoch_{self.current_epoch}/images"
        # generated_image_dir = f"{self.cfg.result_path}/{self.current_epoch}"
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

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        # {'grad_2.0_norm/weight': 0.0003, 'grad_2.0_norm/bias': 0.0, 'grad_2.0_norm_total': 0.0003}
        codebook_norm = grad_norm(self.image_tokenizer.model.quantize.embedding, norm_type=2)
        for norm in codebook_norm.keys():
            self.logger.experiment.add_scalar(
                f"grad_norm/image_tokenizer/model/quantize/{norm}",
                codebook_norm[norm],
                global_step=self.global_step,
            )
        
        transformer_decoder_norm = grad_norm(self.image_tokenizer.model.blocks_image, norm_type=2)
        for norm in transformer_decoder_norm.keys():
            self.logger.experiment.add_scalar(
                f"grad_norm/image_tokenizer/model/blocks_image/{norm}",
                transformer_decoder_norm[norm],
                global_step=self.global_step,
            )
        
        generation_mlp_norm = grad_norm(self.image_tokenizer.model.distill_image_proj, norm_type=2)
        for norm in generation_mlp_norm.keys():
            self.logger.experiment.add_scalar(
                f"grad_norm/image_tokenizer/model/distill_image_proj/{norm}",
                generation_mlp_norm[norm],
                global_step=self.global_step,
            )

if __name__ == "__main__":
    cfg, cfg_yaml = build_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    seed_everything(cfg.experiment.seed, workers=True)

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    os.makedirs(cfg.result_file_path, exist_ok=True)

    datamodule = build_datamodule(
        cfg=cfg,
        train_transform=transform,
        val_transform=transform,
        pin_memory=True,
        epoch=cfg.experiment.max_epochs,
        total_gpus=cfg.dist.n_gpus,
    )

    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    #val_dataloader = datamodule.val_dataloader()
    cfg.experiment.total_training_steps = int(cfg.experiment.max_epochs * (1000000 / 4 / 1024))
    
    karpathy_file = cfg.karpathy_file_path
    root_dir = cfg.root_dir
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
        strategy="ddp",
        max_epochs=cfg.experiment.max_epochs,
        deterministic=cfg.experiment.deterministic,
        logger=tb_logger,
        log_every_n_steps=cfg.experiment.log_every_n_steps,
        # val_check_interval=cfg.experiment.val_check_interval,
        check_val_every_n_epoch=cfg.experiment.check_val_every_n_epoch,
        enable_checkpointing=cfg.experiment.enable_checkpointing,
        num_sanity_val_steps=cfg.experiment.num_sanity_val_steps,
        precision=str(cfg.optimizer.precision),
        callbacks=[ModelSummary(max_depth=3), lr_logger],
        accumulate_grad_batches=cfg.experiment.grad_accumulation,
        gradient_clip_val=cfg.optimizer.grad_clip_val,
    )
    
    wrapper = SEEDTrainingWrapper(cfg).to(device)

    # checkpoint_path = '/ssd0/checkpoints/seed_training_logs_zheedong/stage1_aica/epoch=39-step=11120.ckpt'
    # wrapper = SEEDTrainingWrapper.load_from_checkpoint(checkpoint_path, cfg=cfg, strict=False).to(device)

    wrapper.setup("fit")
    

    trainer.fit(
        wrapper, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
        ckpt_path="/home/zheedong/Projects/SEED/logs/seed_stage2_with_codebook/lightning_logs/version_0/checkpoints/epoch=15-step=3136.ckpt"
    )
    trainer.strategy.barrier()
