import os
from typing import Any, List
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist

import json

import hydra
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, seed_everything
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import pyrootutils
from tqdm import tqdm

from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
import torch.nn.functional as F
from pytorch_lightning.strategies import DDPStrategy, DDPFullyShardedStrategy
from einops import rearrange
import transformers

from pytorch_lightning import loggers as pl_loggers
from functools import partial
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.utilities import grad_norm

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt

from models.seed_qformer.vit import Block
from models.seed_llama_tokenizer import ImageTokenizer

from datamodules.seed_llama_datamodule import SEEDDataModule

from calculate_clip_score import calculate_clip_s_for_folder
from utils.config import build_config

from lavis.models import load_model
from lavis.common.dist_utils import is_dist_avail_and_initialized

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

BOI_TOKEN = "<img>"
EOI_TOKEN = "</img>"
IMG_TOKEN = "<img_{:05d}>"

IMG_FLAG = "<image>"
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 8192
IMAGE_ID_SHIFT = 32000

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]

class SEEDTestWrapper(LightningModule):
    """Training wrapper for SEED

    Args:
        LightningModule (cfg, model): model should be ImageTokenizer
    """    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # ImageTokenizer model
        # Target model to train
        self.image_tokenizer = ImageTokenizer(
            model_path=cfg.checkpoint_path.model_path,
            diffusion_model_path=cfg.checkpoint_path.diffusion_model_path,
            load_diffusion=cfg.stage2.load_diffusion,
            vq_type=cfg.stage2.vq.type,
            discarding_thre=cfg.stage2.vq.discarding_threshold,
            from_pretrained=True if cfg.stage1.init == "SEED" else False,
            vit_precision=cfg.optimizer.vit_precision,
            diffusion_precision=cfg.optimizer.diffusion_precision,
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
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.token_similarity = []
        

    def setup(self, stage):
        self.image_tokenizer.model.eval()

        # Set gradient to False
        for param in self.image_tokenizer.model.parameters():
            param.requires_grad = False

        print("Move stage 2 model to cpu")
        self.image_tokenizer.model.quantize = self.image_tokenizer.model.quantize.to("cpu")
        self.image_tokenizer.model.encode_task_layer = self.image_tokenizer.model.encode_task_layer.to("cpu")
        self.image_tokenizer.model.decode_task_layer = self.image_tokenizer.model.decode_task_layer.to("cpu")
        self.image_tokenizer.model.blocks = self.image_tokenizer.model.blocks.to("cpu")
        self.image_tokenizer.model.blocks_image = self.image_tokenizer.model.blocks_image.to("cpu")
        self.image_tokenizer.model.image_down = self.image_tokenizer.model.image_down.to("cpu")
        self.image_tokenizer.model.distill_image_proj = self.image_tokenizer.model.distill_image_proj.to("cpu") 
            
        ## make dump folder
        os.makedirs(self.cfg.result_file_path, exist_ok=True)
    
    def save_config(self):
        config_save_path = os.path.join(self.logger.log_dir, "config.yaml")
        with open(config_save_path, "w") as f:
            json.dump(self.cfg, f, indent=4)
    
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

    def all_gather_with_grad(self, tensors):
        """
        Performs all_gather operation on the provided tensors.
        Graph remains connected for backward grad computation.
        """
        # Queue the gathered tensors
        world_size = torch.distributed.get_world_size()
        # There is no need for reduction in the single-proc case
        if world_size == 1:
            return tensors

        # tensor_all = GatherLayer.apply(tensors)
        tensor_all = GatherLayer.apply(tensors)

        return torch.cat(tensor_all, dim=0)
    
    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        # if use distributed training
        if not is_dist_avail_and_initialized():
            return tensor

        tensors_gather = [
            torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output

    @torch.no_grad()
    def check_image_text_similarity(self, batch, batch_idx: int, save_dir="image_text_similarity"):
        if len(batch) == 3:
            image, text, image_id = batch
        elif len(batch) == 2:
            image, text = batch
        rank = dist.get_rank()

        with torch.no_grad():
            image_embeds = self.image_tokenizer.model.ln_vision(
                self.image_tokenizer.model.visual_encoder(image)
            )
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        query_tokens = self.image_tokenizer.model.query_tokens.expand(image_embeds.shape[0], -1, -1)

        # Assume image_embeds.shape[0] is the batch size (b) and you have 32 tokens (n)
        b, n, _ = query_tokens.shape

        query_output = self.image_tokenizer.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # Use last hidden state
        # We have 32 tokens, and use last token as image embedding
        # [b, 32, 768]
        image_feats = F.normalize(query_output.last_hidden_state, dim=-1)

        text_tokens = self.image_tokenizer.model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.cfg.dataset.text_max_length,
            return_tensors="pt",
        )

        text_output = self.image_tokenizer.model.Qformer.bert(
            text_tokens.input_ids.to(self.device),
            attention_mask=text_tokens.attention_mask.to(self.device),
            return_dict=True,
        )

        # CLS token
        # [b, 768]
        text_feat = F.normalize(text_output.last_hidden_state[:, 0, :], dim=-1)

        ###============== Image-text Contrastive ===================###

        # image_feats.unsqueeze(1) : [batch_size, 1, num_query_tokens, embed_dim]
        # text_feat_all.unsqueeze(-1) : [batch_size*num_gpu, embed_dim, 1] => broadcast to [batch_size, batch_size*num_gpu, embed_dim, 1]
        # Last two dimensions are broadcasted to all other dimensions
        # [j, 1, n, m] x [k, m, p] => [j, k, n, p]
        # https://pytorch.org/docs/stable/generated/torch.matmul.html
        # sim_q2t : [batch_size, batch_size, num_query_tokens]
        sim_itc = torch.matmul(
            rearrange(image_feats, "bs n d -> bs 1 n d"), rearrange(text_feat, "(bs ngpus) d -> (bs ngpus) d 1", bs=b)
            # image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size, num_query_tokens]

        positive_sample_similarity_per_token = rearrange(torch.diagonal(sim_itc, dim1=0, dim2=1), "n b -> b n")
        # [batch_size, num_query_tokens]

        self.token_similarity.append(positive_sample_similarity_per_token)

        return

    @torch.no_grad()
    def test_step(self, batch, batch_idx: int, save_path=None):
        if self.logger is not None and isinstance(self.logger, pl.loggers.TensorBoardLogger):
            tb_log_dir = self.logger.log_dir
        else:
            tb_log_dir = self.cfg.result_file_path  # Fallback directory if logger is not set

        save_path = f"{tb_log_dir}/histogram"
        os.makedirs(save_path, exist_ok=True)

        with torch.no_grad():
            self.check_image_text_similarity(batch, batch_idx, save_dir=save_path)

    def on_test_epoch_end(self):
        self.token_similarity = torch.cat(self.token_similarity, dim=0).to(dtype=torch.float32)
        # [Validation dataset size, num_query_tokens]
        for token_num in tqdm(range(32)):
            positive_sample_similarity = self.token_similarity[:, token_num]
            
            plt.figure(figsize=(10, 10))
            plt.hist(positive_sample_similarity.cpu().numpy().flatten(), bins=100, range=(-1, 1))
            plt.title(f"Image-Text Similarity (Token {token_num})")
            plt.xlabel("Similarity")
            plt.ylabel("Frequency")

            plt.text(0.05, 0.95, f"Mean: {positive_sample_similarity.mean():.4f}", ha="left", va="top", transform=plt.gca().transAxes)
            plt.text(0.05, 0.90, f"Std: {positive_sample_similarity.std():.4f}", ha="left", va="top", transform=plt.gca().transAxes)
            
            plt.savefig(f"{self.logger.log_dir}/histogram/image_text_similarity_token_num_{token_num}.png")
            plt.close()
        return

        
if __name__ == "__main__":
    cfg, cfg_yaml = build_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    seed_everything(cfg.experiment.seed, workers=True)

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    os.makedirs(cfg.result_file_path, exist_ok=True)

    datamodule = SEEDDataModule(cfg, transform=transform, use_coco_val=cfg.dataset.val_config.use_coco_val)
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    cfg.experiment.total_training_steps = datamodule.total_training_steps

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.result_file_path)

    trainer = pl.Trainer(
        accelerator=device,
        num_nodes=cfg.dist.n_nodes,
        devices=cfg.dist.n_gpus,
        strategy="ddp",
        max_epochs=1,
        deterministic=True,
        logger=tb_logger,
        log_every_n_steps=cfg.experiment.log_every_n_steps,
        # val_check_interval=cfg.experiment.val_check_interval,
        check_val_every_n_epoch=cfg.experiment.check_val_every_n_epoch,
        enable_checkpointing=cfg.experiment.enable_checkpointing,
        num_sanity_val_steps=cfg.experiment.num_sanity_val_steps,
        precision=str(cfg.optimizer.precision),
        callbacks=[ModelSummary(max_depth=3)],
        accumulate_grad_batches=cfg.experiment.grad_accumulation,
        gradient_clip_val=cfg.experiment.grad_clip_val,
    )

    if cfg.load_weight:
        wrapper = SEEDTestWrapper.load_from_checkpoint(cfg.weight_path, cfg=cfg, strict=False)
        print("Loaded model from checkpoint")
    else:
        wrapper = SEEDTestWrapper(cfg).to(device)
        print("Weight from original SEED model")

    # wrapper.setup("fit")
    wrapper.setup("test")
    
    trainer.test(
        wrapper, dataloaders=val_dataloader
    )
    
    trainer.strategy.barrier()
