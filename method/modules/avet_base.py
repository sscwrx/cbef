from numpy._typing._array_like import NDArray
from method.base_method import BaseMethod, MethodConfig
from dataclasses import dataclass, field
from typing import Type, Tuple
import numpy as np
from numpy.typing import NDArray
from abc import abstractmethod

@dataclass
class AVETBaseConfig(MethodConfig):
    """Configuration class for base AVET method."""
    
    _target: Type = field(default_factory=lambda: AVETBase)
    method_name: str = "AVETBase"

class AVETBase(BaseMethod):
    """Abstract base class for AVET implementations."""
    config: AVETBaseConfig

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def process_feature(self, feature_vector: np.ndarray, seed: int = 1) -> NDArray:
        """Process the input feature vector.
        
        This method must be implemented by subclasses.
        
        Args:
            feature_vector (np.ndarray): Input feature vector
            seed (int, optional): Random seed. Defaults to 1.
            
        Returns:
            NDArray: Transformed feature vector
        """
        pass

    def absolute_value_equations_transform(self, x: NDArray, seed: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Performs the absolute value equations transform.

        Args:
            x (np.ndarray): Input feature vector
            seed (int, optional): Random seed. Defaults to 1.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Transformed vector and matrices
        """
        assert len(x.shape) == 1, "Input must be a 1D array, please use np.squeeze() to convert it."

        n = np.floor(len(x)/2).astype(int)
        
        u = x[:n].copy()
        v = x[n:2*n].copy()

        assert u.shape == v.shape, f"u v must have the same shape. u.shape = {u.shape}, v.shape = {v.shape}"
        if np.all(np.equal(u,v)): v=v + 1e-6

        rng = np.random.default_rng(seed)
        R = rng.normal(loc=0, scale=(1/np.sqrt(n)), size=(n,n))
        A = rng.normal(loc=0, scale=(1/np.sqrt(n)), size=(n,n))
        B = rng.normal(loc=0, scale=(1/np.sqrt(n)), size=(n,n))
        y = A @ u + B @ np.abs(R @ v)

        return y, R, A, B