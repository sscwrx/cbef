from numpy._typing._array_like import NDArray
from method.base_method import BaseMethod, MethodConfig

from dataclasses import dataclass, field
from typing import Type
import numpy as np
from numpy.typing import NDArray

@dataclass
class BaselineConfig(MethodConfig):
    """Configuration class for Baseline method."""
    
    _target: Type = field(default_factory=lambda: Baseline)
    method_name: str = "Baseline"

class Baseline(BaseMethod):
    """Baseline implementation that directly returns input features."""
    config: BaselineConfig

    def __init__(self, config):
        super().__init__(config)

    def process_feature(self, feature_vector: NDArray, seed: int = 1) -> NDArray:
        """Directly returns input feature vector without any processing.

        Args:
            feature_vector (NDArray): Input feature vector
            seed (int, optional): Random seed. Defaults to 1.

        Returns:
            NDArray: Original feature vector
        """
        return feature_vector
