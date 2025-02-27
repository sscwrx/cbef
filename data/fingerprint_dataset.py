from dataclasses import dataclass, field
from typing import Type, Literal, Optional,Dict,Tuple
from pathlib import Path
from data.base_dataset import BaseDataset, BaseDatasetConfig
from numpy.typing import NDArray
import scipy as sp
import numpy as np 
@dataclass
class FingerprintDatasetConfig(BaseDatasetConfig):
    """Configuration class for fingerprint datasets."""
    _target: Type = field(default_factory=lambda: FingerprintDataset)
    dataset_name: Literal["FVC2002/Db1_a","FVC2002/Db2_a","FVC2002/Db3_a",
                          "FVC2004/Db1_a","FVC2004/Db2_a","FVC2004/Db3_a"] = "FVC2002/Db1_a"
    n_subjects: int = 100
    samples_per_subject: int = 5
    embedding_dim: int = 299


class FingerprintDataset(BaseDataset):
    config: FingerprintDatasetConfig

    def __init__(self, config: FingerprintDatasetConfig):
        self.config = config
        assert self.config.dataset_name is not None, "Dataset name must be provided."
        assert Path(self.config.embeddings_dir / self.config.dataset_name).exists(), "Embeddings directory does not exist."
    def load_data(self)->Dict[Tuple[int,int],NDArray]:
        """Load the data."""
        embeddings_dict:Dict[Tuple[int,int],NDArray] = {}
        specific_dir: Path = self.config.embeddings_dir / self.config.dataset_name
        assert specific_dir.exists(), f"{specific_dir} does not exist." 
        for file_path in specific_dir.glob(f"*.mat"):
            assert file_path.exists(), f"{file_path} does not exist."
            user_id, sample_id = map(int, file_path.stem.split('_'))
            data = sp.io.loadmat(file_path)["Ftemplate"]
            data = np.squeeze(data)

            # 把前213个元素追加到原始数据尾部
            data = np.append(data, data[:213])
            embeddings_dict[(user_id, sample_id)] = data
        return embeddings_dict
        