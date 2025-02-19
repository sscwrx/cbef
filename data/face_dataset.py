from dataclasses import dataclass, field
from typing import Type, Literal,Optional
from pathlib import Path
from data.base_dataset import BaseDataset, BaseDatasetConfig


@dataclass
class FaceDatasetConfig(BaseDatasetConfig):
    """Configuration class for face datasets."""

    _target: Type = field (default_factory=lambda: FaceDataset)
    dataset_name:Optional[Literal["LFW","FEI","CASIA-WebFace","ColorFeret"]] = None
    n_subjects: int = 127
    samples_per_subject: int = 12
    embedding_dim: int = 512 
    
class FaceDataset(BaseDataset):
    config: FaceDatasetConfig

    def __init__(self, config: FaceDatasetConfig):
        self.config = config
        assert self.config.dataset_name is not None, "Dataset name must be provided."
        assert Path(self.config.embeddings_dir / self.config.dataset_name).exists(), "Embeddings directory does not exist."
    def load_data(self):
        """Load the data."""
        pass
