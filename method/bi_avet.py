from numpy._typing._array_like import NDArray
from method.avet_base import AVETBase, AVETBaseConfig
from dataclasses import dataclass, field
from typing import Type
import numpy as np
from numpy.typing import NDArray

@dataclass
class BiAVETConfig(AVETBaseConfig):
    """Configuration class for Binary AVET method."""
    
    _target: Type = field(default_factory=lambda: BiAVET)
    method_name: str = "BiAVET"

class BiAVET(AVETBase):
    """Binary AVET implementation."""
    config: BiAVETConfig

    def __init__(self, config):
        super().__init__(config)

    def process_feature(self, feature_vector: np.ndarray, seed: int = 1) -> NDArray[np.int32]:
        """Converts input feature vector to binary vector using AVET transform.

        Args:
            feature_vector (np.ndarray): Input feature vector
            seed (int, optional): Random seed. Defaults to 1.

        Returns:
            NDArray[np.int32]: Binary vector of 0s and 1s
        """
        y, _, _, _ = self.absolute_value_equations_transform(feature_vector, seed)
        return np.where(y > 0, 1, 0)