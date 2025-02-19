from dataclasses import dataclass, field
from typing import Type, Literal
from data.base_dataset import BaseDataset, BaseDatasetConfig


@dataclass
class FaceDatasetConfig(BaseDatasetConfig):
    """Configuration class for face datasets."""
    _target: Type = field (default_factory=lambda: FaceDataset)
    n_subjects: int = 127
    samples_per_subject: int = 12
    embedding_dim: int = 512 
class FaceDataset(BaseDataset):
    config: FaceDatasetConfig

    def __init__(self, config: FaceDatasetConfig):
        self.config = config

    def load_data(self):
        """Load the data."""
        pass
