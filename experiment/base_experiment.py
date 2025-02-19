from config.base_config import BaseConfig
from pathlib import Path 
from datetime import datetime
from dataclasses import dataclass,field
from data.base_dataset import BaseDataset,BaseDatasetConfig
from method.base_method import BaseMethod,MethodConfig

@dataclass
class ExperimentConfig(BaseConfig):
    """Configuration class for experiments."""
    output_dir: str = Path("./output")
    timestamp:str = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    dataset:BaseDatasetConfig = field(default_factory=BaseDatasetConfig)
    method:MethodConfig = field(default_factory=MethodConfig)
    # metrics:
    # 



class Experiment:
    """Base class for all experiments."""
    config: ExperimentConfig

    def __init__(self, config: ExperimentConfig, **kwargs):
        self.config = config
        self.kwargs = kwargs

    def setup(self):
        """Set up the experiment modules."""
        

