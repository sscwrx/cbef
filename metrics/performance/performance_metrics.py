from ast import Dict, List
import itertools
import numpy as np
import time

from sympy import O
import metrics.performance.CalculateVerificationRate as CalculateVerificationRate

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
class EERMetricsConfig(BaseConfig):
    """Configuration class for metrics."""

    _target: Type = field(default_factory= lambda: EERMetrics) 
    """目标类型"""

    measure: Optional[Literal["cosine", "euclidean", "hamming", "jaccard"]] = None 
    """Measure to use for calculating similarity."""

    verbose: bool = False 
    """Whether to print verbose output.""" 
    
    protected_template_dir: Path = Path("./")
    """Path to protected templates, set it in ExperimentConfig""" 

class EERMetrics:
    """ Class for calculating EER and threshold."""

    config: EERMetricsConfig
    data_config: BaseDatasetConfig
    
    def __init__(self, config: EERMetricsConfig):
        self.config:EERMetricsConfig = config
        assert self.config.measure is not None, "Measure must be provided." 

        
    def calculate_template_similarity(self, template1, template2)->float:
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

    def perform_matching(self)->Tuple[float, float,List[float],List[float]]:
        """执行模板匹配过程"""
        # 生成真匹配和假匹配的组合
        genuine_combinations = list(itertools.combinations(range(1, self.data_config.samples_per_subject+1), 2))
        impostor_combinations = list(itertools.combinations(range(1, self.data_config.n_subjects+1), 2))

        genuine_similarity_list = []
        impostor_similarity_list = []
        
        # 执行真匹配（同一用户的不同样本）
        start_time1 = time.time()
        with tqdm(total=self.n_genuines_combinations,desc="Genuine Matching") as pbar:
            for i in range(self.data_config.n_subjects):
                for comb in genuine_combinations: 
                    # 加载第一个模板
                    template1:NDArray = np.load(f"{self.config.protected_template_dir}/{i+1}_{comb[0]}.npy")
                    # 加载第二个模板
                    template2:NDArray = np.load(f"{self.config.protected_template_dir}/{i+1}_{comb[1]}.npy")
                    template1 = np.squeeze(template1)
                    template2 = np.squeeze(template2)
                    # 计算相似度

                    similarity = self.calculate_template_similarity(template1, template2)
                    genuine_similarity_list.append(similarity)
                    pbar.update(1)

        end_time1 = time.time()
        mean_time_genuine = (end_time1 - start_time1) / self.n_genuines_combinations
        print(f"\n {self.n_genuines_combinations}次真匹配的平均时间：", mean_time_genuine)
            
        # 执行假匹配（不同用户之间的匹配）
        start_time2 = time.time()
        for comb in tqdm(impostor_combinations,desc="Imposter Matching"):
            # 加载第一个模板
            template1:NDArray = np.load(f"{self.config.protected_template_dir}/{comb[0]}_1.npy")
            
            # 加载第二个模板
            template2:NDArray = np.load(f"{self.config.protected_template_dir}/{comb[1]}_1.npy")
            
            # 计算相似度
            similarity = self.calculate_template_similarity(template1, template2)
            impostor_similarity_list.append(similarity)
            
        end_time2 = time.time()
        mean_time_impostor = (end_time2 - start_time2) / self.n_impostor_combinations
        print(f"{self.n_impostor_combinations}次假匹配的平均时间: {mean_time_impostor}")
        
        # 计算EER和阈值
        EER, thr = CalculateVerificationRate.computePerformance(
            genuine_similarity_list, 
            impostor_similarity_list, 
            0.001,
            verbose=self.config.verbose
        )
        
        return EER, thr, genuine_similarity_list, impostor_similarity_list

    @property
    def n_genuines_combinations(self):
        return self.data_config.n_subjects * math.comb(self.data_config.samples_per_subject, 2)
    
    @property
    def n_impostor_combinations(self):
        return math.comb(self.data_config.n_subjects, 2)