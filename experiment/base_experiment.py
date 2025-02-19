from config.base_config import BaseConfig
from pathlib import Path 
from datetime import datetime
from dataclasses import dataclass,field
from data.base_dataset import BaseDataset,BaseDatasetConfig
from method.base_method import BaseMethod,MethodConfig
from metrics.performance.performance_metrics import MetricsConfig
from typing import Tuple

@dataclass
class ExperimentConfig:
    """Configuration class for experiments."""

    timestamp:str = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    output_dir: Path = Path("./output")

    # metrics:
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
    dataset:BaseDatasetConfig
    method:MethodConfig
    metrics:MetricsConfig

    def __init__(self, config: ExperimentConfig, **kwargs):
        self.config = config
        self.kwargs = kwargs

    def setup(self):
        """Set up the experiment modules."""
        self.dataset = self.dataset.setup()
        self.method = self.method.setup()
        self.metrics = self.metrics.setup()

