from abc import ABC, abstractmethod 
from dataclasses import dataclass
from typing import List
import numpy as np 
from pathlib import Path
from config.base_config import BaseConfig


class MethodConfig(BaseConfig):
    output_dir: Path = Path("./output") 
class BaseMethod(ABC):
        
    config: MethodConfig
    """Base class for all methods."""


    def __init__(self, config: MethodConfig,**kwargs):
        self.config = config
        self.kwargs = kwargs

    @abstractmethod
    def process_feature(self, feature_vector:np.ndarray,seed:int=1) -> np.ndarray:
        """Abstract method to be implemented by subclasses."""
        pass


