from numpy._typing._array_like import NDArray
from method.modules.avet_base import AVETBase, AVETBaseConfig
from dataclasses import dataclass, field
from typing import Type
import numpy as np
from numpy.typing import NDArray

@dataclass
class InAVETConfig(AVETBaseConfig):
    """Configuration class for Integer AVET method."""
    
    _target: Type = field(default_factory=lambda: InAVET)
    method_name: str = "InAVET"
    k: int = 300  # Number of random matrices
    g: int = 16   # Size of output vector for each transformation

class InAVET(AVETBase):
    """Integer AVET implementation."""
    config: InAVETConfig

    def __init__(self, config):
        super().__init__(config)

    def process_feature(self, feature_vector: NDArray ,seed: int = 1) -> NDArray[np.int32]:
        """Converts input feature vector to integer vector using AVET transform.

        Args:
            feature_vector (NDArray): Input feature vector
            seed (int, optional): Random seed. Defaults to 1.

        Returns:
            NDArray[np.int32]: Integer vector
        """
        assert len(feature_vector.shape) == 1, "Input must be a 1D array, please use np.squeeze() to convert it." 

        n = np.floor(len(feature_vector)/2).astype(int)
        
        u = feature_vector[:n].copy()
        v = feature_vector[n:2*n].copy()

        assert u.shape == v.shape, f"u v must have the same shape. u.shape = {u.shape}, v.shape = {v.shape}"
        if np.all(np.equal(u,v)): v=v + 1e-6

        rng = np.random.default_rng(seed)
        output = np.zeros(self.config.k,)
        
        R = rng.normal(loc=0, scale=(1/np.sqrt(n)), size=(self.config.k,n,n))
        A = rng.normal(loc=0, scale=(1/np.sqrt(n)), size=(self.config.k,self.config.g,n))
        B = rng.normal(loc=0, scale=(1/np.sqrt(n)), size=(self.config.k,self.config.g,n))  

        for i in range(self.config.k):
            y = A[i] @ u + B[i] @ np.abs(R[i] @ v)
            output[i] = np.argmax(y)

        return output.astype(np.int32)