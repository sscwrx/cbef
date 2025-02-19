from dataclasses import dataclass

from click import Option
from pyparsing import Opt
from config.base_config import BaseConfig 
from pathlib import Path 
from typing import Type,Literal, Optional
from dataclasses import field, dataclass

@dataclass
class BaseDatasetConfig(BaseConfig): 
    """Base configuration class."""
    _target: Type = field (default_factory=lambda: BaseDataset)
    n_subjects: int = 0
    samples_per_subject: int = 0
    embeddings_dir: Path = Path("./embeddings")
    """数据目录"""
    
    @property
    def total_samples(self):
        return self.n_subjects * self.samples_per_subject
    


class BaseDataset: 
    config: BaseConfig

    def __init__(self, config: BaseConfig):
        self.config = config
