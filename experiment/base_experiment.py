
import logging
from pathlib import Path 
from datetime import datetime
from dataclasses import dataclass,field
from data.base_dataset import BaseDatasetConfig
from data.face_dataset import FaceDatasetConfig
from method.base_method import MethodConfig
from method.bio_hash import BioHashConfig
from metrics.performance_metrics import PerformanceMetrics, PerformanceMetricsConfig
from typing import Literal, Tuple,Union,Type ,Literal,Optional,List

from tqdm import tqdm 
import scipy as sp 
import numpy as np
import time 
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
console = Console()


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
    metrics_config: PerformanceMetricsConfig = field(default_factory=PerformanceMetricsConfig)

    expriment_name: str = "Experiment"
    # method_name : Literal["BioHash", "AVET", "C_IOM"] = "BioHash"
    # dataset_name: Literal["LFW", "FEI","CASIA-WebFace","ColorFert"] = "LFW"
    timestamp:str = datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
    output_dir: Path = Path("./output")  
    """The output directory for the experiment this time."""
    expriment_times:int =5 


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
        """Get the base directory for the specific output."""

        output = self.output_dir / self.method_config.method_name / f"{self.dataset_config.dataset_name}" / f"{self.timestamp }_{self.expriment_name}" 
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

        self.metrics:PerformanceMetrics = self.config.metrics_config.setup()
        self.metrics.data_config = self.dataset.config
        self.metrics.config.protected_template_dir = self.config._get_base_dir / "protected_template" 

        Path(self.metrics.config.protected_template_dir).mkdir(parents=True, exist_ok=True)
        self.config.save_config()

        console.print(Panel.fit(str(self.config),
                            title="[bold green]√ Experiment Configuration [/]",
                            border_style="green"))
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler( self.config._get_base_dir / f'results_{datetime.now().strftime("%Y%m%d-%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    def run(self)->None:
        """ run generating and matching """
        best_eer_list = []
        best_threshold_list = []
        mean_time_generate_protected_template_list = []

        for i in range(1,self.config.expriment_times+1):
            
            # 1. generate protected templates
            mean_time_generate_protected_template: float = self.perform_generating(seed=i)

            # 2. perform matching
            genuine_similarity_list, impostor_similarity_list, mean_time_genuine, mean_time_impostor = self.metrics.perform_matching()

            # 3. perform evaluation
            # 3.1 EER
            best_eer ,best_threshold ,far_list,gar_list= self.metrics.perform_evaluation(genuine_similarity_list, impostor_similarity_list)

            best_eer_list.append(best_eer)
            best_threshold_list.append(best_threshold)
            mean_time_generate_protected_template_list.append(mean_time_generate_protected_template)

            # 3.2 Decidability Index 
            mean_genuine_similarity = np.mean(genuine_similarity_list)
            std_genuine_simlarity = np.std(genuine_similarity_list)
            mean_imposter_similarity = np.mean(impostor_similarity_list)
            std_imposter_similarity = np.std(impostor_similarity_list)
            assert std_genuine_simlarity != 0 or std_imposter_similarity != 0, "std_genuine_simlarity or std_imposter_similarity is zero"
            DI = np.abs(mean_genuine_similarity - mean_imposter_similarity) / np.sqrt((std_genuine_simlarity+std_imposter_similarity)/2)
            self._print_result_table(i,
                                    mean_time_generate_protected_template,
                                    best_eer, best_threshold,
                                    DI,
                                    mean_time_genuine,
                                     mean_time_impostor)
        # 4. log the a full experiment result
        self._log_experiment_results(self.logger, 
                                     best_eer_list, 
                                     best_threshold_list, 
                                     self.config.dataset_config.dataset_name, 
                                     mean_time_generate_protected_template_list)
    def perform_generating(self,seed=1)->float:
        """Perform the experiment to generate hashcodes ."""

        # 重置方法种子,并且生成或更新参数，这样就不用在每次生成模板的时候都重新生成参数
        self.method.set_seed(seed)

        # 加载数据
        data = self.dataset.load_data()

        # 初始化计时
        start_time = time.time()
        # 令牌被盗场景
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
    
    def _log_experiment_results(self, logger, eer_list, optimal_thr_list, dataset, mean_time_list):
        separator = "#" * 100
        logger.info(separator)
        logger.info(f"\nFinal results for {dataset}:")
        
        # EER Statistics
        mean_eer = np.mean(eer_list)
        std_eer = np.std(eer_list)
        max_eer = np.max(eer_list)
        min_eer = np.min(eer_list)
        
        # Threshold Statistics
        mean_thr = np.mean(optimal_thr_list)
        std_thr = np.std(optimal_thr_list)
        max_thr = np.max(optimal_thr_list)
        min_thr = np.min(optimal_thr_list)
        
        # Time Statistics
        mean_time = np.mean(mean_time_list)
        max_time = np.max(mean_time_list)
        min_time = np.min(mean_time_list)
        
        # Output formatting
        logger.info("EER Statistics:")
        logger.info(f"  Mean: {mean_eer:.2f}%")
        logger.info(f"  Std Dev: {std_eer:.2f}%")
        logger.info(f"  Max: {max_eer:.2f}%")
        logger.info(f"  Min: {min_eer:.2f}%")
        
        logger.info("\nThreshold Statistics:")
        logger.info(f"  Mean: {mean_thr:.2f}")
        logger.info(f"  Std Dev: {std_thr:.2f}")
        logger.info(f"  Max: {max_thr:.2f}")
        logger.info(f"  Min: {min_thr:.2f}")
        
        logger.info("\nTemplate Generation Time:")
        logger.info(f"  Mean: {mean_time*1000:.2f} ms")
        logger.info(f"  Max: {max_time*1000:.2f} ms")
        logger.info(f"  Min: {min_time*1000:.2f} ms")
        
        # For reference, also log the raw lists
        logger.info(f"\nRaw Data:")
        logger.info(f"  EER values: {[f'{x:.2f}' for x in eer_list]}")
        logger.info(f"  Threshold values: {[f'{x:.2f}' for x in optimal_thr_list]}")
        
        logger.info(separator)

    def _print_result_table(self, i: int, mean_time_generate: float, eer: float, threshold: float, 
                        DI:float,mean_time_genuine: float =0.0, mean_time_impostor: float = 0.0) -> None:
        """打印实验结果表格
        
        Args:
            i (int): 实验次数
            mean_time_generate (float): 平均生成时间（秒）
            eer (float): 等错误率
            threshold (float): 最佳阈值
            DI (float): Decidability Index
            mean_time_genuine (float, optional): 真匹配平均时间（秒）
            mean_time_impostor (float, optional): 假匹配平均时间（秒）
        """
        method_name = self.config.method_config.method_name
        dataset_name = self.config.dataset_config.dataset_name
        # Create result table
        table = Table(title=f"{method_name} on {dataset_name} \n Experiment ({i}/{self.config.expriment_times}) result, ", 
                show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        
        table.add_row("Mean Generation Time", f"{mean_time_generate*1000:.3f}ms")  # mean_time is in seconds
        
        # Add matching times if provided
        if mean_time_genuine is not None:
            table.add_row("Mean Genuine Match Time", f"{mean_time_genuine*1000:.3f}ms")
        if mean_time_impostor is not None:
            table.add_row("Mean Impostor Match Time", f"{mean_time_impostor*1000:.3f}ms")
            
        table.add_row("Equal Error Rate (EER)", f"{eer:.2f}%")  # EER is already percentage
        table.add_row("Optimal Threshold", f"{threshold:.4f}")
        table.add_row("Decidability Index (DI)", f"{DI:.4f}")
        console.print(table)

if __name__ == "__main__":

    method_config = BioHashConfig()
    dataset_config = FaceDatasetConfig(dataset_name="LFW") 
    metrics_config = PerformanceMetricsConfig(measure="euclidean")

    config = ExperimentConfig(
        method_config=method_config,
        dataset_config=dataset_config,
        metrics_config=metrics_config
    )

    experiment:Experiment = config.setup()

    mean_time = experiment.perform_generating()

