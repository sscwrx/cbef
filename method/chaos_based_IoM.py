from numpy._typing._array_like import NDArray
from method.base_method import BaseMethod, MethodConfig
from dataclasses import dataclass, field
from typing import Type, Tuple
import numpy as np 
from pathlib import Path
from tqdm import tqdm
from method.modules.sliding_window import SlidingWindowMechanism


@dataclass
class ChaosBasedIoMConfig(MethodConfig):
    """Configuration class for ChaosBasedIoM method."""
    _target: Type = field(default_factory=lambda: ChaosBasedIoM)
    method_name: str = "C_IOM"
    dimension: int = 512
    window_size: int = 3
    initial_conditions: Tuple[float, float, float] = (0.98, 0.21, 0.46)
    sequence_length: int = 50000
    dt: float = 0.001
    # memristor parameters
    R_off: float = 20e3
    R_on: float = 1e2
    M0: float = 16e3
    u_v: float = 1e-14  # the average mobility of oxygen vacancies
    D: float = 1e-8  # thickness of the film. unit: m
    a: float = 35.0
    b: float = 3.0
    c: float = 20.0
    alpha: float = 2000.0
    IoM_type: str = "min"  # or "max"
    generate_chaos_first: bool = True
    
    def __post_init__(self):
        # Calculate derived parameters
        self.beta = ((self.R_on - self.R_off) * self.u_v * self.R_on) / self.D**2
        self.N_3 = -(self.R_off - self.M0)**2 / (2 * self.beta)
        self.N_4 = -(self.R_on - self.M0)**2 / (2 * self.beta)
        self.N_5 = (self.R_off**2 - self.M0**2) / (2 * self.beta)
        self.N_6 = (self.R_on**2 - self.M0**2) / (2 * self.beta)

class ChaosBasedIoM(BaseMethod):
    """Implementation of Chaos-based IoM method."""
    config: ChaosBasedIoMConfig

    def __init__(self, config):
        super().__init__(config)
        self.sliding_window = SlidingWindowMechanism(config.window_size)
        
        # Initialize chaos sequences
        self._initialize_chaos_sequences()
        
        # Generate chaos seed
        self.Pi_n = self._find_dimension_indices(self.x_I)
        self.C_s = self._reset_C_s(seed_index=1)

    def set_seed(self,seed:int):
        self.C_s = self._reset_C_s(seed_index=seed)
    def _memristor_nonlinear_function(self, epsilon):
        if epsilon < self.config.N_5:
            return (epsilon - self.config.N_3) / self.config.R_off
        elif self.config.N_5 <= epsilon <= self.config.N_6:
            return (np.sqrt(2 * self.config.beta * epsilon + self.config.M0**2) - self.config.M0) / self.config.beta
        elif self.config.N_6 < epsilon:
            return (epsilon - self.config.N_4) / self.config.R_on
        else:
            raise Exception(f"The input value of memristor_nonlinear_function is invalid")

    def _memristor_chaotic_map(self, last_u, last_v, last_w):
        u_dot = self.config.a * (last_v - last_u)
        h = self._memristor_nonlinear_function(-np.abs(last_u))
        v_dot = (self.config.c - self.config.a) * self.config.alpha * h - last_u * last_w + self.config.c * last_v
        w_dot = last_u * last_v - self.config.b * last_w
        return u_dot, v_dot, w_dot

    def _generate_chaos_sequences(self, length):
        u_seq = np.zeros(length)
        v_seq = np.zeros(length)
        w_seq = np.zeros(length)
        
        u_seq[0], v_seq[0], w_seq[0] = self.config.initial_conditions

        for i in tqdm(range(1, length), desc="Generating chaos sequences"):
            u_dot, v_dot, w_dot = self._memristor_chaotic_map(u_seq[i-1], v_seq[i-1], w_seq[i-1])
            u_seq[i] = u_seq[i-1] + u_dot * self.config.dt
            v_seq[i] = v_seq[i-1] + v_dot * self.config.dt
            w_seq[i] = w_seq[i-1] + w_dot * self.config.dt

        x_I = np.floor(((u_seq + v_seq + 100) % 1) * 10**16) % (self.config.dimension + 1)
        y_I = np.floor(((u_seq + w_seq + 100) % 1) * 10**16) % (self.config.dimension + 1)
        z_I = np.floor(((v_seq + w_seq + 100) % 1) * 10**16) % (self.config.dimension + 1)
        
        return x_I.astype(int), y_I.astype(int), z_I.astype(int)

    def _initialize_chaos_sequences(self):
        try:
            if self.config.generate_chaos_first:
                raise Exception()
            sequences = np.load("sequence.npz")
            self.x_I, self.y_I, self.z_I = sequences['u'], sequences['v'], sequences['w']
            print("Load sequence from file.")
        except Exception as e:
            print("Generate sequence.")
            self.x_I, self.y_I, self.z_I = self._generate_chaos_sequences(self.config.sequence_length)
            np.savez("sequence.npz", u=self.x_I, v=self.y_I, w=self.z_I)

    def _find_dimension_indices(self, x_I):
        indices = np.where(x_I == self.config.dimension)[0]
        return indices
    

    def _reset_C_s(self, seed_index=1):
        Pi_n = self.Pi_n[7000*(seed_index-1):7000*seed_index]
        rng = np.random.default_rng(seed=Pi_n)
        C_s = rng.permutation(np.arange(0, self.config.dimension))

        return C_s

    def process_shuffle_indices(self, y_I):
        result = np.zeros(self.config.dimension, dtype=np.int64)
        mask = y_I != 0
        unique_indices = np.unique(y_I[mask], return_index=True)[1]
        unique_count = len(unique_indices)
        
        result[:unique_count] = y_I[np.sort(unique_indices)]
        existing_values = set(result[:unique_count])
        missing_values = np.array([x for x in range(1, self.config.dimension + 1)
                               if x not in existing_values], dtype=np.int64)
        
        remaining_space = self.config.dimension - unique_count
        result[unique_count:] = missing_values[:remaining_space]
        
        return result

    def _generate_Facecode(self, feature_vector, seed_index=1):
        if len(feature_vector) != self.config.dimension:
            raise ValueError(f"特征向量维度({len(feature_vector)})与预设维度({self.config.dimension})不匹配")

        y_I = self.y_I[7000*(seed_index-1):7000*seed_index]
        processed_y_I = self.process_shuffle_indices(y_I)
        
        i = np.arange(0, self.config.dimension)[:self.config.dimension//2]
        Feature_vector = np.copy(feature_vector)
        processed_y_I = np.subtract(processed_y_I, 1)
        t = feature_vector[processed_y_I[i]]
        feature_vector[processed_y_I[i]] = feature_vector[processed_y_I[self.config.dimension - i - 1]]
        feature_vector[processed_y_I[self.config.dimension - i - 1]] = t
        
        return Feature_vector

    def process_feature(self, feature_vector: np.ndarray, seed: int = 1) -> np.ndarray:
        """Process feature vector using Chaos-based IoM method."""
        Facecode = self._generate_Facecode(feature_vector, seed_index=seed)
        protected_template = self.sliding_window.process_feature(Facecode, self.C_s, IoM_type=self.config.IoM_type)
        return protected_template
