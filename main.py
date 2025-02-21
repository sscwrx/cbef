from typing import List, Tuple
import tyro 
from experiment.base_experiment import ExperimentConfig 
from data.face_dataset import FaceDatasetConfig 
from method.bio_hash import BioHashConfig
from experiment.base_experiment import ExperimentConfig ,Experiment
from metrics.performance.performance_metrics import EERMetricsConfig,EERMetrics
import numpy as np 
from config.base_config import BaseConfig
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
console = Console()

if __name__ == "__main__":

    method_config = BioHashConfig()
    dataset_config = FaceDatasetConfig(dataset_name="LFW") 
    metrics_config = EERMetricsConfig(measure="euclidean")

    config = ExperimentConfig(
        expriment_name="测试下整个流程是否能跑通",
        method_config=method_config,
        dataset_config=dataset_config,
        metrics_config=metrics_config
    )

    experiment:Experiment = config.setup()

    console.print(Panel.fit(
    str(experiment.config),
    title="[bold green]√ Experiment Configuration [/]",
    border_style="green"
    ))
    mean_time: float = experiment.perform_generating()
    result_tuple: Tuple[float, float, List[float], List[float]]= experiment.perform_matching() 
    EER, threshold, geniune_similarity, imposter_similarity = result_tuple 




    # 创建结果表格
    table = Table(title="Expriment result", show_header=True, header_style="bold magenta")
    table.add_column("指标", style="cyan")
    table.add_column("数值", justify="right", style="green")
    
    table.add_row("平均生成时间", f"{mean_time:.6f}s")
    table.add_row("等错误率(EER)", f"{EER:.4} %")
    table.add_row("最佳阈值", f"{threshold:.4f}")
    console.print(table)