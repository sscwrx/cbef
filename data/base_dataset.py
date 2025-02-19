from dataclasses import dataclass
from config.base_config import BaseConfig 
from pathlib import Path 
from typing import Type,Literal

@dataclass
class BaseDatasetConfig(BaseConfig): 
    """Base configuration class."""
    _target: Type
    """目标类型"""
    
    data_type: Literal["face","fingerprint"]
    """数据类型"""

    dataset_name: str 
    """数据集名称"""
    data_dir: Path
    """数据目录"""
    
    n_subjects: int
    """主体数量"""

    samples_per_subject: int
    """每个主体的样本数"""

    embedding_dim: int
    """嵌入维度"""


    @property
    def total_samples(self):
        return self.n_subjects * self.samples_per_subject
    

@dataclass
class FaceDatasetConfig(BaseDatasetConfig):
    """Configuration class for face datasets."""
    
    n_subjects: int = 127
    samples_per_subject: int = 12
    embedding_dim: int = 512 


@dataclass
class FingerprintDatasetConfig(BaseDatasetConfig):
    """Configuration class for fingerprint datasets."""
    n_subjects: int = 100
    samples_per_subject: int = 5
    embedding_dim: int = 299


class BaseDataset: 
    config: BaseDatasetConfig

    def __init__(self, config: BaseConfig):
        self.config = config