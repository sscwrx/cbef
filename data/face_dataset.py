from ast import Dict, Tuple
from dataclasses import dataclass, field
from typing import Type, Literal,Optional,Tuple ,Dict 
from numpy.typing import NDArray
from pathlib import Path

from matplotlib.pyplot import sci
from data.base_dataset import BaseDataset, BaseDatasetConfig
import scipy as sp 
import numpy as np 
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
        embeddings_dict:Dict[Tuple[int,int],NDArray] = {}
        specific_dir = self.config.embeddings_dir / str(self.config.dataset_name) 
        assert specific_dir.exists(), f"{specific_dir} does not exist."
        for file_path in specific_dir.glob(f"*.mat"):
            assert file_path.exists(), f"{file_path} does not exist." 
            user_id, sample_id = map(int, file_path.stem.split('_'))
            data = sp.io.loadmat(file_path)["preOut"]
            data = np.squeeze(data)
            embeddings_dict[(user_id, sample_id)] = data 

        return embeddings_dict 

# 人脸
## 1_1.*
# 指纹
#  1_4.*