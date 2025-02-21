from typing import List, Tuple
import tyro 
from data.fingerprint_dataset import FingerprintDataset, FingerprintDatasetConfig
from experiment.base_experiment import ExperimentConfig 
from data.face_dataset import FaceDatasetConfig 
from method.bio_hash import BioHashConfig
from method.swg import SWGConfig
from method.bi_avet import BiAVETConfig
from experiment.base_experiment import ExperimentConfig ,Experiment
from metrics.performance.performance_metrics import EERMetricsConfig,EERMetrics

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
console = Console()

if __name__ == "__main__":

    # swg_config = SWGConfig()
    bi_avet_config = BiAVETConfig()
    dataset_config = FaceDatasetConfig(dataset_name="LFW") 
    dataset2_config = FingerprintDatasetConfig(dataset_name="FVC2002/Db1_a")
    eer_config = EERMetricsConfig(measure="hamming")

    
    expr_config = ExperimentConfig(
        expriment_name="测试下指纹数据集",
        method_config=bi_avet_config,
        dataset_config=dataset2_config,
        metrics_config=eer_config,
        expriment_times=1
    )
    
    experiment:Experiment = expr_config.setup()

    experiment.run()