import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from metrics.performance.eer_metrics import EERMetrics, EERMetricsConfig
from metrics.performance.CalculateVerificationRate import computePerformance, caculateVerificationRate

@pytest.fixture
def mock_config():
    return EERMetricsConfig(
        measure="cosine",
        verbose=False,
        protected_template_dir=Path("./test_data")
    )

@pytest.fixture
def mock_data_config():
    class MockDataConfig:
        n_subjects = 10
        samples_per_subject = 4
    return MockDataConfig()

def test_calculate_template_similarity_cosine(mock_config):
    eer_metrics = EERMetrics(mock_config)
    template1 = np.array([1, 0, 0])
    template2 = np.array([1, 0, 0])
    similarity = eer_metrics._calculate_template_similarity(template1, template2)
    assert np.isclose(similarity, 1.0)

def test_calculate_template_similarity_euclidean(mock_config):
    mock_config.measure = "euclidean"
    eer_metrics = EERMetrics(mock_config)
    template1 = np.array([1, 0, 0])
    template2 = np.array([0, 1, 0])
    similarity = eer_metrics._calculate_template_similarity(template1, template2)
    assert np.isclose(similarity, -np.sqrt(2))

def test_calculate_template_similarity_hamming(mock_config):
    mock_config.measure = "hamming"
    eer_metrics = EERMetrics(mock_config)
    template1 = np.array([1, 0, 1])
    template2 = np.array([1, 1, 0])
    similarity = eer_metrics._calculate_template_similarity(template1, template2)
    assert np.isclose(similarity, 1/3)

def test_calculate_template_similarity_jaccard(mock_config):
    mock_config.measure = "jaccard"
    eer_metrics = EERMetrics(mock_config)
    template1 = np.array([1, 0, 1])
    template2 = np.array([1, 1, 0])
    similarity = eer_metrics._calculate_template_similarity(template1, template2)
    assert np.isclose(similarity, 1/5)

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

@patch("numpy.load")
def test_perform_matching(mock_load, mock_config, mock_data_config):
    # 准备模拟数据
    genuine_template = np.array([1, 0, 0])
    impostor_template = np.array([0, 1, 0])
    
    # 计算需要的模拟数据数量
    n_genuine = mock_data_config.n_subjects * 6 * 2  # 10 subjects * C(4,2) * 2 templates
    n_impostor = 45 * 2  # C(10,2) * 2 templates
    
    # 设置 side_effect
    mock_load.side_effect = [genuine_template] * n_genuine + [impostor_template] * n_impostor
    
    eer_metrics = EERMetrics(mock_config)
    eer_metrics.data_config = mock_data_config
    
    genuine_sim, impostor_sim, _, _ = eer_metrics.perform_matching()
    assert len(genuine_sim) == mock_data_config.n_subjects * 6
    assert len(impostor_sim) == 45

def test_perform_evaluation():
    genuine_sim = [0.9, 0.8, 0.85]
    impostor_sim = [0.1, 0.2, 0.15]
    eer, threshold, far_list, gar_list = computePerformance(genuine_sim, impostor_sim, step=0.1)
    assert isinstance(eer, float)
    assert isinstance(threshold, float)
    assert isinstance(far_list, list)
    assert isinstance(gar_list, list)