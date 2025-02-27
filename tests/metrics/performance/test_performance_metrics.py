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