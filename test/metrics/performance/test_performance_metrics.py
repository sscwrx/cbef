import pytest
import numpy as np
import math
from pathlib import Path
from unittest.mock import MagicMock, patch
from metrics.performance.performance_metrics import MetricsConfig, EERMetrics
from data.base_dataset import FaceDatasetConfig

@pytest.fixture
def output_path():
    """测试输出目录的固定路径"""
    return Path("./output/test")

@pytest.fixture
def template_path():
    """模板文件路径"""
    return Path("./dummy/path")

@pytest.fixture
def data_path():
    """数据目录路径"""
    return Path("./data")

def test_calculate_template_similarity_cosine(output_path, template_path, data_path):
    """测试余弦相似度计算"""
    config = MetricsConfig(
        _target=EERMetrics,
        measure="cosine",
        verbose=False,
        protected_template_dir=template_path,
        output_dir=output_path
    )
    data_config = FaceDatasetConfig(
        _target=None,
        data_type="face",
        dataset_name="test",
        output_dir=output_path,
        data_dir=data_path
    )
    metrics = EERMetrics(config=config, data_config=data_config)
    
    template1 = np.array([1, 0, 1, 0])
    template2 = np.array([1, 0, 1, 0])
    similarity = metrics.calculate_template_similarity(template1, template2)
    assert similarity == pytest.approx(1.0)

    template2 = np.array([0, 1, 0, 1])
    similarity = metrics.calculate_template_similarity(template1, template2)
    assert similarity == pytest.approx(0.0)

def test_calculate_template_similarity_euclidean(output_path, template_path, data_path):
    """测试欧氏距离相似度计算"""
    config = MetricsConfig(
        _target=EERMetrics,
        measure="euclidean",
        verbose=False,
        protected_template_dir=template_path,
        output_dir=output_path
    )
    data_config = FaceDatasetConfig(
        _target=None,
        data_type="face",
        dataset_name="test",
        output_dir=output_path,
        data_dir=data_path
    )
    metrics = EERMetrics(config=config, data_config=data_config)
    
    template1 = np.array([1, 0, 1, 0])
    template2 = np.array([1, 0, 1, 0])
    similarity = metrics.calculate_template_similarity(template1, template2)
    assert similarity == pytest.approx(0.0)

    template2 = np.array([0, 1, 0, 1])
    similarity = metrics.calculate_template_similarity(template1, template2)
    assert similarity < 0  # 欧氏距离为负数表示距离较远

def test_calculate_template_similarity_hamming(output_path, template_path, data_path):
    """测试汉明距离相似度计算"""
    config = MetricsConfig(
        _target=EERMetrics,
        measure="hamming",
        verbose=False,
        protected_template_dir=template_path,
        output_dir=output_path
    )
    data_config = FaceDatasetConfig(
        _target=None,
        data_type="face",
        dataset_name="test",
        output_dir=output_path,
        data_dir=data_path
    )
    metrics = EERMetrics(config=config, data_config=data_config)
    
    template1 = np.array([1, 0, 1, 0])
    template2 = np.array([1, 0, 1, 0])
    similarity = metrics.calculate_template_similarity(template1, template2)
    assert similarity == pytest.approx(1.0)

    template2 = np.array([0, 1, 0, 1])
    similarity = metrics.calculate_template_similarity(template1, template2)
    assert similarity == pytest.approx(0.0)

def test_calculate_template_similarity_jaccard(output_path, template_path, data_path):
    """测试Jaccard相似度计算"""
    config = MetricsConfig(
        _target=EERMetrics,
        measure="jaccard",
        verbose=False,
        protected_template_dir=template_path,
        output_dir=output_path
    )
    data_config = FaceDatasetConfig(
        _target=None,
        data_type="face",
        dataset_name="test",
        output_dir=output_path,
        data_dir=data_path
    )
    metrics = EERMetrics(config=config, data_config=data_config)
    
    template1 = np.array([1, 0, 1, 0])
    template2 = np.array([1, 0, 1, 0])
    similarity = metrics.calculate_template_similarity(template1, template2)
    assert similarity == pytest.approx(1.0)

    template2 = np.array([0, 1, 0, 1])
    similarity = metrics.calculate_template_similarity(template1, template2)
    assert similarity == pytest.approx(0.0)

def test_template_length_mismatch(output_path, template_path, data_path):
    """测试模板长度不匹配的情况"""
    config = MetricsConfig(
        _target=EERMetrics,
        measure="cosine",
        verbose=False,
        protected_template_dir=template_path,
        output_dir=output_path
    )
    data_config = FaceDatasetConfig(
        _target=None,
        data_type="face",
        dataset_name="test",
        output_dir=output_path,
        data_dir=data_path
    )
    metrics = EERMetrics(config=config, data_config=data_config)
    
    template1 = np.array([1, 0, 1])
    template2 = np.array([1, 0, 1, 0])
    
    with pytest.raises(ValueError) as exc_info:
        metrics.calculate_template_similarity(template1, template2)
    assert "模板长度不匹配" in str(exc_info.value)

def test_n_genuines_combinations(output_path, template_path, data_path):
    """测试真实匹配组合数计算"""
    config = MetricsConfig(
        _target=EERMetrics,
        measure="cosine",
        verbose=False,
        protected_template_dir=template_path,
        output_dir=output_path
    )
    data_config = FaceDatasetConfig(
        _target=None,
        data_type="face",
        dataset_name="test",
        output_dir=output_path,
        data_dir=data_path
    )
    metrics = EERMetrics(config=config, data_config=data_config)

    # 对于每个主体(n_subjects)，有C(samples_per_subject,2)个组合
    expected = data_config.n_subjects * math.comb(data_config.samples_per_subject, 2)
    assert metrics.n_genuines_combinations == expected

def test_n_impostor_combinations(output_path, template_path, data_path):
    """测试虚假匹配组合数计算"""
    config = MetricsConfig(
        _target=EERMetrics,
        measure="cosine",
        verbose=False,
        protected_template_dir=template_path,
        output_dir=output_path
    )
    data_config = FaceDatasetConfig(
        _target=None,
        data_type="face",
        dataset_name="test",
        output_dir=output_path,
        data_dir=data_path
    )
    metrics = EERMetrics(config=config, data_config=data_config)
    
    # 虚假匹配的组合数是C(samples_per_subject,2)
    expected = math.comb(data_config.samples_per_subject, 2)
    assert metrics.n_impostor_combinations == expected

@patch('numpy.load')
def test_perform_matching(mock_load, output_path, template_path, data_path):
    """测试perform_matching方法"""
    config = MetricsConfig(
        _target=EERMetrics,
        measure="cosine",
        verbose=False,
        protected_template_dir=template_path,
        output_dir=output_path
    )
    data_config = FaceDatasetConfig(
        _target=None,
        data_type="face",
        dataset_name="test",
        output_dir=output_path,
        data_dir=data_path
    )
    metrics = EERMetrics(config=config, data_config=data_config)
    
    # 模拟numpy.load的返回值
    mock_data = {
        'protected_template': np.array([1, 0, 1, 0])
    }
    mock_load.return_value = mock_data
    
    with patch('metrics.performance.CalculateVerificationRate.computePerformance') as mock_compute:
        mock_compute.return_value = (0.1, 0.5)  # 模拟EER和阈值
        
        EER, thr = metrics.perform_matching()
        
        assert EER == 0.1
        assert thr == 0.5
        
        # 验证compute_performance被正确调用
        mock_compute.assert_called_once()