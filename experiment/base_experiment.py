
from pathlib import Path 
from datetime import datetime
from dataclasses import dataclass,field
from data.base_dataset import BaseDatasetConfig
from data.face_dataset import FaceDatasetConfig
from method.base_method import MethodConfig
from method.bio_hash import BioHashConfig
from metrics.performance.performance_metrics import EERMetrics, EERMetricsConfig
from typing import Literal, Tuple,Union,Type ,Literal,Optional,List

from tqdm import tqdm 
import scipy as sp 
import numpy as np
import time 

@dataclass 

@dataclass
class ExperimentConfig:
    """Configuration class for experiments."""

    _target: Type = field(default_factory=lambda: Experiment) 
    
    # 当 ExperimentConfig 实例化时,  这些配置类需要以实例对象的形式传递给它ExperimentConfig
    # 方法配置
    method_config: MethodConfig = field(default_factory=BioHashConfig)
    # 数据集配置 
    dataset_config: BaseDatasetConfig = field(default_factory=FaceDatasetConfig)
    # 评价指标配置
    metrics_config: EERMetricsConfig = field(default_factory=EERMetricsConfig)

    expriment_name: str = "Experiment"
    # method_name : Literal["BioHash", "AVET", "C_IOM"] = "BioHash"
    # dataset_name: Literal["LFW", "FEI","CASIA-WebFace","ColorFert"] = "LFW"
    timestamp:str = datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
    output_dir: Path = Path("./output")  
    """The output directory for the experiment this time."""

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
    

    @property 
    def _get_base_dir(self):
        """Get the base directory for the output."""

        output = self.output_dir / self.method_config.method_name / f"{self.dataset_config.dataset_name}" / f"{self.expriment_name}_{self.timestamp}" 
        output.mkdir(parents=True, exist_ok=True)
        return output

    def save_config(self):
        """Save the configuration to a file."""
        with open(self._get_base_dir/"config.txt", "w") as f:
            f.write(str(self))
    
    def setup(self):
        """Setup the experiment."""
        return self._target(self) # 调用Experiment类的__init__()
    
class Experiment:
    """Base class for all experiments."""
    config: ExperimentConfig

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.dataset = self.config.dataset_config.setup()
        self.method = self.config.method_config.setup()

        self.metrics:EERMetrics = self.config.metrics_config.setup()
        self.metrics.data_config = self.dataset.config
        self.metrics.config.protected_template_dir = self.config._get_base_dir / "protected_template" 
        Path(self.metrics.config.protected_template_dir).mkdir(parents=True, exist_ok=True)
    def perform_generating(self)->float:
        """Perform the experiment to generate hashcodes ."""


        # 加载数据

        data = self.dataset.load_data()
        # 初始化计时
        start_time = time.time()

        for key, embedding in data.items():
            identy_id, sample_id = key 
            protected_template = self.method.process_feature(embedding)
            save_dir = self.config._get_base_dir / f"protected_template"
            save_dir.mkdir(parents=True,exist_ok=True)
            np.save( save_dir/ f"{identy_id}_{sample_id}.npy", protected_template)
            
        # 计算平均时间
        end_time = time.time()
        mean_time = (end_time - start_time) / self.dataset.total_samples
        # print(f'生成{self.dataset.total_samples}个受保护模板的平均时间是：', mean_time)
        
        # 保存保护模板
        return mean_time
    
    def perform_matching(self,expr_times=5) -> Tuple[float, float, List[float], List[float]]:
        """执行模板匹配实验, 计算EER和最佳阈值。
        
        计算过程:
        1. 对同一用户的不同样本进行真匹配(genuine matching)
        2. 对不同用户的样本进行假匹配(impostor matching)
        3. 根据真假匹配的相似度分布计算EER和最佳阈值
        
        Returns:
            Tuple[float,float,List[float],List[float]]: 返回一个元组, 包含:
                - EER (float): 等错误率, 表示FAR=FRR时的错误率
                - threshold (float): 最佳判决阈值, 使FAR≈FRR
                - genuine_similarities (List[float]): 真匹配相似度列表
                - impostor_similarities (List[float]): 假匹配相似度列表
        """
        return self.metrics.perform_matching()

    def log_experiment_results(self,logger, eer_list, optimal_thr_list, dataset,mean_time_list,table_results):
        logger.info("#" * 100)
        logger.info(f"\nFinal results for {dataset}:")
        logger.info(f"Mean EER: {np.mean(eer_list):.2f}")
        logger.info(f"Mean Optimal Threshold: {np.mean(optimal_thr_list):.2f}")
        logger.info(f"Standard Deviation of EER: {np.std(eer_list):.2f}")
        logger.info(f"Standard Deviation of Optimal Threshold: {np.std(optimal_thr_list):.2f}")
        logger.info(f"Max EER: {np.max(eer_list):.2f}")
        logger.info(f"Min EER: {np.min(eer_list):.2f}")
        logger.info(f"Max Optimal Threshold: {np.max(optimal_thr_list):.2f}")
        logger.info(f"Min Optimal Threshold: {np.min(optimal_thr_list):.2f}")
        logger.info(f"Mean time for generating protected templates: {np.mean(mean_time_list)} s, max time: {np.max(mean_time_list)} s. min time: {np.min(mean_time_list)} s.",)
        table_results.append([dataset, np.mean(eer_list)])
        logger.info("#" * 100)

if __name__ == "__main__":

    method_config = BioHashConfig()
    dataset_config = FaceDatasetConfig(dataset_name="LFW") 
    metrics_config = EERMetricsConfig(measure="euclidean")

    config = ExperimentConfig(
        method_config=method_config,
        dataset_config=dataset_config,
        metrics_config=metrics_config
    )

    experiment:Experiment = config.setup()

    mean_time = experiment.perform_generating()
    result_tuple= experiment.perform_matching() 
