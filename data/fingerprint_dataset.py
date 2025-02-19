
from dataclasses import dataclass, field
from typing import Type

from data.base_dataset import BaseDataset, BaseDatasetConfig

@dataclass
class FingerprintDatasetConfig(BaseDatasetConfig):
    """Configuration class for fingerprint datasets."""
    _target: Type = field(default_factory=lambda: FingerprintDataset)
    n_subjects: int = 100
    samples_per_subject: int = 5
    embedding_dim: int = 299




class FingerprintDataset(BaseDataset):
    config: FingerprintDatasetConfig

    def __init__(self, config: FingerprintDatasetConfig):
        self.config = config

    def load_data(self):
        """Load the data."""
        pass