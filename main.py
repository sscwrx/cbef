import tyro 
from experiment.base_experiment import ExperimentConfig 
from data.face_dataset import FaceDatasetConfig 
from method.bio_hash import BioHashConfig

from metrics.performance.performance_metrics import EERMetricsConfig,EERMetrics
import numpy as np 
from config.base_config import BaseConfig



# 建立配置类，确定参数 
method_config = BioHashConfig()
lfw_data_config =  FaceDatasetConfig(
                    dataset_name="LFW",)
metrics_config = EERMetricsConfig(measure="euclidean")  


# 实例化
method = method_config.setup()
data = lfw_data_config.setup()
metrics = metrics_config.setup()
# Expriment 

# method data metrics
