from ast import Dict, List
import itertools
import numpy as np
import time

from data.face_dataset import FaceDatasetConfig
from data.fingerprint_dataset import FingerprintDatasetConfig
import metrics.CalculateVerificationRate as CalculateVerificationRate

import scipy as sp
from scipy.spatial import distance
from config.base_config import BaseConfig
from data.base_dataset import BaseDataset
from data.base_dataset import BaseDatasetConfig
from pathlib import Path 
from typing import Literal, Type,List,Dict,Tuple,Union ,Optional
import math 
from dataclasses import dataclass,field
from numpy.typing import NDArray
from tqdm import tqdm
@dataclass
class PerformanceMetricsConfig(BaseConfig):
    """Configuration class for metrics."""

    _target: Type = field(default_factory= lambda: PerformanceMetrics) 
    """目标类型"""

    verbose: bool = False 
    """Whether to print verbose output.""" 

class PerformanceMetrics:
    """ Class for calculating Performance and threshold."""

    config: PerformanceMetricsConfig

    
    def __init__(self, config: PerformanceMetricsConfig):
        self.config:PerformanceMetricsConfig = config

    def perform_evaluation(self, genuine_similarity_list: List[float], impostor_similarity_list: List[float]) -> Tuple[float, float, List[int], List[int]]:
        """
        计算等错误率(EER)和最佳阈值
        
        Args:
            genuine_similarity_list (List[float]): 真匹配相似度列表
            impostor_similarity_list (List[float]): 假匹配相似度列表
        
        Returns:
            Tuple[float, float, float]: EER, 最佳阈值, 最佳阈值对应的FAR
        """
        assert len(genuine_similarity_list)>0 or len(impostor_similarity_list)>0, "No similarity data provided"
        best_eer, best_thrshold,far_list,gar_list = CalculateVerificationRate.computePerformance(genuine_similarity_list, impostor_similarity_list,step=0.001)
        return best_eer, best_thrshold,far_list,gar_list

    def calculate_DI(self, genuine_similarity_list:List[float], impostor_similarity_list:List[float])->float:
        """计算 Decidability Index (DI)
        
        输入:真匹配相似度列表,假匹配相似度列表
        输出:DI(flaot)
        """
        mean_genuine_similarity = np.mean(genuine_similarity_list)
        std_genuine_simlarity = np.var(genuine_similarity_list)
        mean_imposter_similarity = np.mean(impostor_similarity_list)
        std_imposter_similarity = np.var(impostor_similarity_list)
        assert std_genuine_simlarity != 0 or std_imposter_similarity != 0, "std_genuine_simlarity or std_imposter_similarity is zero"
        assert mean_imposter_similarity !=0 or mean_genuine_similarity != 0, "mean_imposter_similarity or mean_genuine_similarity is zero"
        DI = np.abs(mean_genuine_similarity - mean_imposter_similarity) / np.sqrt((std_genuine_simlarity+std_imposter_similarity)/2)
        return DI 