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

    measure: Optional[Literal["cosine", "euclidean", "hamming", "jaccard"]] = None 
    """Measure to use for calculating similarity."""

    verbose: bool = False 
    """Whether to print verbose output.""" 
    
    protected_template_dir: Path = Path("./")
    """Path to protected templates, set it in ExperimentConfig""" 

class PerformanceMetrics:
    """ Class for calculating Performance and threshold."""

    config: PerformanceMetricsConfig
    data_config: BaseDatasetConfig
    
    def __init__(self, config: PerformanceMetricsConfig):
        self.config:PerformanceMetricsConfig = config
        assert self.config.measure is not None, "Measure must be provided." 

        
    def _calculate_template_similarity(self, template1, template2)->float:
        """计算两个受保护模板之间的相似度"""
        if len(template1) != len(template2):
            raise ValueError(f"模板长度不匹配, {len(template1)} != {len(template2)}")
        if self.config.measure == "cosine":
            # cosine similarity
            similarity = np.dot(template1, template2) / (np.linalg.norm(template1) * np.linalg.norm(template2))
        elif self.config.measure == "euclidean":
            # eculidean
            similarity = - sp.spatial.distance.euclidean(template1, template2)
        elif self.config.measure == "hamming":
            # hamming similarity 
            similarity = 1 - sp.spatial.distance.hamming(template1, template2)
        elif self.config.measure == "jaccard":
            # distance = sp.spatial.distance.jaccard(template1, template2)
            # similarity = 1 - distance
            match = np.abs(template1 - template2)
            total_zero_num = np.count_nonzero(match == 0)
            similarity = total_zero_num / (template1.__len__() + template2.__len__() - total_zero_num)

        return similarity

    def perform_matching(self)->Tuple[List[float], List[float], float, float]:
        """
        执行模板匹配过程，计算真/假匹配相似度和匹配时间
        
        Returns:
            tuple: 真匹配相似度列表，假匹配相似度列表，真匹配平均时间，假匹配平均时间

        """
        # 生成真匹配和假匹配的组合
        if isinstance(self.data_config, FingerprintDatasetConfig):
            genuine_combinations = list(itertools.combinations(range(4, 4+self.data_config.samples_per_subject), 2))
            impostor_combinations = list(itertools.combinations(range(1, self.data_config.n_subjects+1), 2))
        else:
            genuine_combinations = list(itertools.combinations(range(1, self.data_config.samples_per_subject+1), 2))
            impostor_combinations = list(itertools.combinations(range(1, self.data_config.n_subjects+1), 2))
        
        genuine_similarity_list = []
        impostor_similarity_list = []
        
        # 执行真匹配（同一用户的不同样本）
        start_time1 = time.time()
        with tqdm(total=self.n_genuines_combinations, desc="真匹配") as pbar:
            for i in range(self.data_config.n_subjects):
                for comb in genuine_combinations: 
                    # 加载第一个模板
                    template1: NDArray = np.load(f"{self.config.protected_template_dir}/{i+1}_{comb[0]}.npy")
                    # 加载第二个模板
                    template2: NDArray = np.load(f"{self.config.protected_template_dir}/{i+1}_{comb[1]}.npy")
                    
                    template1 = np.squeeze(template1)
                    template2 = np.squeeze(template2)
                    # 计算相似度
                    genuine_similarity = self._calculate_template_similarity(template1, template2)
                    genuine_similarity_list.append(genuine_similarity)
                    pbar.update(1)

        end_time1 = time.time()
        mean_time_genuine = (end_time1 - start_time1) / self.n_genuines_combinations
        # print(f"\n {self.n_genuines_combinations}次真匹配的平均时间：{mean_time_genuine:.6f}秒")
            
        # 执行假匹配（不同用户之间的匹配）
        start_time2 = time.time()
        with tqdm(total=self.n_impostor_combinations, desc="假匹配") as pbar:
            for comb in impostor_combinations:
                # 加载第一个模板
                try:
                    template1: NDArray = np.load(f"{self.config.protected_template_dir}/{comb[0]}_1.npy")
                    # 加载第二个模板
                    template2: NDArray = np.load(f"{self.config.protected_template_dir}/{comb[1]}_1.npy")
                except Exception:
                    template1: NDArray = np.load(f"{self.config.protected_template_dir}/{comb[0]}_4.npy")
                    template2: NDArray = np.load(f"{self.config.protected_template_dir}/{comb[1]}_4.npy")
                
                # 计算相似度
                impostor_similarity = self._calculate_template_similarity(template1, template2)
                impostor_similarity_list.append(impostor_similarity)
                pbar.update(1)
                
        end_time2 = time.time()
        mean_time_impostor = (end_time2 - start_time2) / self.n_impostor_combinations
        # print(f"{self.n_impostor_combinations}次假匹配的平均时间: {mean_time_impostor:.6f}秒")
        
        return genuine_similarity_list, impostor_similarity_list, mean_time_genuine, mean_time_impostor

    def perform_evaluation(self, genuine_similarity_list: List[float], impostor_similarity_list: List[float]) -> Tuple[float, float, List[int], List[int]]:
        """
        计算等错误率(EER)和最佳阈值
        
        Args:
            genuine_similarity_list (List[float]): 真匹配相似度列表
            impostor_similarity_list (List[float]): 假匹配相似度列表
        
        Returns:
            Tuple[float, float, float]: EER, 最佳阈值, 最佳阈值对应的FAR
        """
        best_eer, best_thrshold,far_list,gar_list = CalculateVerificationRate.computePerformance(genuine_similarity_list, impostor_similarity_list,step=0.001)
        return best_eer, best_thrshold,far_list,gar_list
    
    @property
    def n_genuines_combinations(self):
        return self.data_config.n_subjects * math.comb(self.data_config.samples_per_subject, 2)
    
    @property
    def n_impostor_combinations(self):
        return math.comb(self.data_config.n_subjects, 2)