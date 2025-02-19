import tyro 
from experiment.base_experiment import ExperimentConfig 
from data.base_dataset import FaceDatasetConfig
from method.bio_hash import BioHashConfig

from metrics.performance.performance_metrics import EERMetricsConfig,EERMetrics
import numpy as np 
from config.base_config import BaseConfig

# 数据


# 方法 



arr_input = np.random.rand(10,)

output = method.process_feature(arr_input)


# 建立配置类，确定参数 
method_config = BioHashConfig()
data_config =  FaceDatasetConfig(
    data_dir = "./embeddings/LFW",
)
metrics_config = EERMetricsConfig() 

# 实例化
method = method_config.setup()