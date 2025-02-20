from sympy import Union
from config.base_config import BaseConfig
from pathlib import Path 
from datetime import datetime
from dataclasses import dataclass,field
from data.face_dataset import FaceDatasetConfig
from data.fingerprint_dataset import FingerprintDatasetConfig
from method.base_method import MethodConfig
from metrics.performance.performance_metrics import EERMetricsConfig
from typing import Tuple,Union
from time import time 
from tqdm import tqdm 
import scipy as sp 
import numpy as np
@dataclass
class ExperimentConfig:
    """Configuration class for experiments."""

    expriment_name: str = "Experiment"
    timestamp:str = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    output_dir: Path = Path("./output") /  timestamp   

    def __str__(self):
        """just for pretty print() """
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)




class Experiment:
    """Base class for all experiments."""
    config: ExperimentConfig
    dataset:Union[FaceDatasetConfig,FingerprintDatasetConfig]
    method: MethodConfig
    metrics:EERMetricsConfig

    def __init__(self, config: ExperimentConfig, **kwargs):
        self.config = config
        self.kwargs = kwargs

    def setup(self):
        """Set up the experiment modules."""
        self.dataset = self.dataset.setup()
        self.method = self.method.setup()
        self.metrics = self.metrics.setup()
    
    def perform_generating(self):
        """Perform the experiment to generate hashcodes ."""
        
        # 初始化计时
        start_time = time.time()
        
        # 生成受保护的模板
        for i in tqdm(range(0,self.dataset.n_subjects)):
            for j in range(0,self.dataset.samples_per_subject):
                # 加载特征向量
                path = f"{self.dataset.embeddings_dir}/{self.dataset.dataset_name}/{i+1}_{j+4}.mat" # dataset value is like "FVC2002/Db1_a" 
                assert Path(path).exists(), f"{path} does not exist."
                    
                mat_file = sp.io.loadmat(path)
                fingerprint_vector = mat_file['Ftemplate']  # 与原始代码保持一致


                # 归一化
                fingerprint_vector = np.squeeze(fingerprint_vector)
                # fingerprint_vector = fingerprint_vector / np.linalg.norm(fingerprint_vector)

                # # 生成受保护模板
                # if method == "avet":
                #     protected_template,_,_,_ = absolute_value_equations_transform(fingerprint_vector,seed)
                # elif method == "bi_avet":
                #     protected_template = bi_avet(fingerprint_vector,seed)
                # elif method == "in_avet":
                #     protected_template = in_avet(fingerprint_vector,k=300,g=16,seed=seed)
                # elif method == "bio_hash":
                #     protected_template = biohash(fingerprint_vector, bh_len=40, user_seed=seed)
                # elif method == "baseline":
                #     protected_template = fingerprint_vector  
                # # 保存结果
                # output_path = f"./protectedTemplates/{dataset}/{i+1}_{j+4}"
                # np.savez(output_path, 
                #         protected_template=protected_template,
                #         seed=seed)

        # 计算平均时间
        end_time = time.time()
        mean_time = (end_time - start_time) / 500 
        print('生成500个受保护模板的平均时间是：', mean_time)
        return mean_time

