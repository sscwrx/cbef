import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from metrics.performance_metrics import PerformanceMetrics, PerformanceMetricsConfig
from metrics.CalculateVerificationRate import computePerformance, caculateVerificationRate

@pytest.fixture
def mock_config():
    return PerformanceMetricsConfig(
        verbose=False
    )

def test_perform_evaluation(mock_config):
    genuine_sim = [0.9, 0.8, 0.85]
    impostor_sim = [0.1, 0.2, 0.15]
    pm = PerformanceMetrics(mock_config)
    eer, threshold, far_list, gar_list = pm.perform_evaluation(genuine_sim, impostor_sim)
    assert isinstance(eer, float)
    assert isinstance(threshold, float)
    assert isinstance(far_list, list)
    assert isinstance(gar_list, list)

def test_compute_performance():
    genuine = [0.9, 0.8, 0.85]
    impostor = [0.1, 0.2, 0.15]
    eer, threshold, far_list, gar_list = computePerformance(genuine, impostor, step=0.1)
    assert isinstance(eer, float)
    assert isinstance(threshold, float)
    assert isinstance(far_list, list)
    assert isinstance(gar_list, list)

def test_caculate_verification_rate():
    genuine = [0.9, 0.8, 0.85]
    impostor = [0.1, 0.2, 0.15]
    tsr, far, frr, gar = caculateVerificationRate(0.5, genuine, impostor)
    assert isinstance(tsr, float)
    assert isinstance(far, float)
    assert isinstance(frr, float)
    assert isinstance(gar, float)

def test_calculate_DI_normal(mock_config):
    """测试 calculate_DI 函数在正常情况下的行为"""
    genuine_sim = [0.9, 0.8, 0.85]  # 均值约0.85，方差不为0
    impostor_sim = [0.1, 0.2, 0.15]  # 均值约0.15，方差不为0
    pm = PerformanceMetrics(mock_config)
    di = pm.calculate_DI(genuine_sim, impostor_sim)
    assert isinstance(di, float)
    assert di > 0  # DI应该是正数，因为是差值的绝对值除以标准差

def test_calculate_DI_zero_mean():
    """测试当两个分布均值都为0时的情况"""
    genuine_sim = [0.0, 0.0, 0.0]  # 均值为0
    impostor_sim = [0.0, 0.0, 0.0]  # 均值为0
    pm = PerformanceMetrics(PerformanceMetricsConfig(verbose=False))
    with pytest.raises(AssertionError):
        pm.calculate_DI(genuine_sim, impostor_sim)

def test_calculate_DI_edge_case():
    """测试边缘情况：genuine和impostor分布非常接近"""
    genuine_sim = [0.5, 0.51, 0.49]
    impostor_sim = [0.49, 0.5, 0.51]
    pm = PerformanceMetrics(PerformanceMetricsConfig(verbose=False))
    di = pm.calculate_DI(genuine_sim, impostor_sim)
    assert isinstance(di, float)
    assert di >= 0  # DI应该是非负数
    assert di < 1  # 由于分布非常接近，DI应该很小