from abc import abstractmethod
from dataclasses import dataclass

from click import Option
from pyparsing import Opt
from config.base_config import BaseConfig 
from pathlib import Path 
from typing import Tuple, Type,Literal, Optional
from dataclasses import field, dataclass
from abc import abstractmethod
import numpy as np 
from numpy.typing import NDArray
from typing import Dict ,List
@dataclass
class BaseDatasetConfig(BaseConfig): 
    """Base configuration class."""
    _target: Type = field (default_factory=lambda: BaseDataset)
    dataset_name :Optional[str] = None
    n_subjects: int = 0
    samples_per_subject: int = 0
    embeddings_dir: Path = Path("./embeddings")
    """数据目录"""
    


class BaseDataset: 
    config: BaseDatasetConfig

    def __init__(self, config: BaseDatasetConfig):
        self.config = config

    @abstractmethod
    def load_data(self)->Dict[Tuple[int,int],NDArray ]:
        pass

    @property
    def total_samples(self):
        return self.config.n_subjects * self.config.samples_per_subject
    
