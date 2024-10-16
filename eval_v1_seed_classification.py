import os
import torch
import torch.distributed as dist
from tqdm import tqdm

import json

import hydra
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, seed_everything
from omegaconf import OmegaConf
import pyrootutils

import torch.nn.functional as F
from pytorch_lightning.strategies import DDPStrategy, DDPFullyShardedStrategy
from einops import rearrange
from pytorch_lightning.callbacks import ModelSummary

from models.seed_llama_tokenizer import ImageTokenizer

from datamodules.imagenet_datamodule import ImageNetDataModule

from utils.config import build_config

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

BOI_TOKEN = "<img>"
EOI_TOKEN = "</img>"
IMG_TOKEN = "<img_{:05d}>"

IMG_FLAG = "<image>"
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 8192
IMAGE_ID_SHIFT = 32000

class SEEDTestWrapper(LightningModule):
    """Training wrapper for Slot

    Args:
        LightningModule (cfg, model): model should be ImageTokenizer
    """

    def __init__(self, cfg, class_names, val_length):
        super().__init__()
        self.cfg = cfg
        self.stage = cfg.experiment.stage
        self.class_names = class_names
        self.val_length = val_length
        self.acc = 0.0

        # Load tokenizer
        self.image_tokenizer = ImageTokenizer(
            model_path="pretrained/seed_tokenizer/seed_quantizer.pt",
            diffusion_model_path="stabilityai/stable-diffusion-2-1-unclip",
            device="cpu",  # For PyTorch Lightning
            load_diffusion=True,
            vq_type="vq2",
            discarding_thre="1000",
            from_pretrained=True,
            vit_precision="fp16",
            diffusion_precision="fp16",
            legacy=True,
        )

    def setup(self, stage):
        for parm in self.image_tokenizer.parameters():
            parm.requires_grad = False

    def save_config(self):
        config_save_path = os.path.join(self.logger.log_dir, "config.yaml")
        with open(config_save_path, "w") as f:
            json.dump(self.cfg, f, indent=4)

    @torch.no_grad()
    def on_test_start(self) -> None:
        self.acc = 0
        self.text_embeddings = []
        for class_name in tqdm(self.class_names):
            prompt = f"A photo of {class_name}"
            # Text embedding
            text_tokens = self.image_tokenizer.model.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64
            )

            text_output = self.image_tokenizer.model.Qformer.bert(
                text_tokens.input_ids.to(self.device),
                attention_mask=text_tokens.attention_mask.to(self.device),
                return_dict=True,
            )

            text_feat = text_output.last_hidden_state  # Only use [CLS] Token
            self.text_embeddings.append(text_feat)

        self.text_embeddings = torch.cat(self.text_embeddings, dim=0)   # [1000, 768]

        self.text_embeddings = F.normalize(self.text_embeddings, dim=-1)
        self.text_embeddings = self.text_embeddings[:, 0, :].contiguous()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
         image, text = batch[0], batch[1]

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
        image_feats = rearrange(query_output.last_hidden_state[:, -1, :], "b d -> b 1 d").contiguous()
        image_feats = F.normalize(image_feats, dim=-1)

        # Compare similarity with text embeddings
        sim = torch.matmul(
            self.text_embeddings,
            rearrange(image_feats, "b 1 d -> d b")
        ).T   # [b, 1000]

        #  Get prediction for batch
        _, pred = sim.max(dim=-1)

        # Get accuracy
        self.acc += (pred == batch[2]).float().sum().item()

        loss_itc = F.cross_entropy(sim, batch[2], label_smoothing=0.1)
        self.print(f"ITC Loss: {loss_itc.item()}")

    def on_test_end(self) -> None:
        self.acc = self.acc / 50000
        self.print(f"Accuracy :{self.acc * 100} %")
        print(f"Accuracy :{self.acc * 100} %")

if __name__ == "__main__":
    cfg, cfg_yaml = build_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(cfg.experiment.seed, workers=True)

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    os.makedirs(cfg.result_file_path, exist_ok=True)

    datamodule = ImageNetDataModule(batch_size=64, image_size=224)
    datamodule.setup()

    class_names = datamodule.val_dataset.class_names

    # Debug
    # datamodule = SEEDDataModule(cfg, transform=transform, use_coco_val=cfg.dataset.val_config.use_coco_val)

    val_dataloader = datamodule.val_dataloader()

    # val_dataloader = datamodule.val_dataloader()
    val_length = len(val_dataloader)

    strategy = DDPStrategy(
        find_unused_parameters=cfg.experiment.find_unused_parameters,
    )

    trainer = pl.Trainer(
        accelerator=device,
        num_nodes=cfg.dist.n_nodes,
        devices=cfg.dist.n_gpus,
        strategy=strategy,
        max_epochs=cfg.experiment.max_epochs,
        deterministic=cfg.experiment.deterministic,
        log_every_n_steps=cfg.experiment.log_every_n_steps,
        val_check_interval=cfg.experiment.val_check_interval,
        # check_val_every_n_epoch=cfg.experiment.check_val_every_n_epoch,
        enable_checkpointing=cfg.experiment.enable_checkpointing,
        num_sanity_val_steps=cfg.experiment.num_sanity_val_steps,
        precision=str(cfg.optimizer.precision),
        callbacks=[ModelSummary(max_depth=3)],
        accumulate_grad_batches=cfg.experiment.grad_accumulation,
        gradient_clip_val=cfg.experiment.grad_clip_val,
    )

    if cfg.load_weight and cfg.resume:
        raise ValueError("Only checkpoint or finetune")

    wrapper = SEEDTestWrapper(cfg, class_names=class_names, val_length=val_length)

    wrapper.setup("fit")

    trainer.test(wrapper, dataloaders=val_dataloader)

    trainer.strategy.barrier()
