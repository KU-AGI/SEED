import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule

class CompressionDataModule(LightningDataModule):
    def __init__(self)