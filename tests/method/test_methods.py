import pytest
import numpy as np
from method.bi_avet import BiAVET, BiAVETConfig
from method.in_avet import InAVET, InAVETConfig
from method.bio_hash import BioHash, BioHashConfig

@pytest.fixture
def sample_feature():
    """生成测试用的特征向量"""
    rng = np.random.default_rng(42)
    return rng.normal(size=100)  # 生成100维的特征向量

def test_bi_avet_process(sample_feature):
    """测试二值化AVET"""
    config = BiAVETConfig()
    bi_avet = BiAVET(config)
    
    # 测试输出是二值的
    result = bi_avet.process_feature(sample_feature)
    assert set(np.unique(result)).issubset({0, 1})
    
    # 测试输出维度
    assert result.shape == (len(sample_feature) // 2,)
    
    # 测试结果的一致性
    result2 = bi_avet.process_feature(sample_feature, seed=1)
    np.testing.assert_array_equal(result, result2)

def test_in_avet_process(sample_feature):
    """测试整数AVET"""
    config = InAVETConfig(k=50, g=8)  # 使用较小的k和g加快测试
    in_avet = InAVET(config)
    
    # 测试输出形状
    result = in_avet.process_feature(sample_feature)
    assert result.shape == (config.k,)
    
    # 测试输出是整数
    assert result.dtype == np.int32
    
    # 测试输出范围在[0, g)之间
    assert np.all(result >= 0) and np.all(result < config.g)
    
    # 测试结果的一致性
    result2 = in_avet.process_feature(sample_feature, seed=1)
    np.testing.assert_array_equal(result, result2)
    
    # 测试不同配置
    config2 = InAVETConfig(k=30, g=4)
    in_avet2 = InAVET(config2)
    result3 = in_avet2.process_feature(sample_feature)
    assert result3.shape == (30,)
    assert np.all(result3 >= 0) and np.all(result3 < 4)

def test_bio_hash_process(sample_feature):
    """测试BioHash"""
    config = BioHashConfig(bh_len=64)
    bio_hash = BioHash(config)
    
    # 测试输出是二值的
    result = bio_hash.process_feature(sample_feature)
    assert set(np.unique(result)).issubset({0, 1})
    
    # 测试输出维度
    assert result.shape == (config.bh_len,)
    
    # 测试结果的一致性
    result2 = bio_hash.process_feature(sample_feature, seed=1)
    np.testing.assert_array_equal(result, result2)
    
    # 测试不同配置
    config2 = BioHashConfig(bh_len=32)
    bio_hash2 = BioHash(config2)
    result3 = bio_hash2.process_feature(sample_feature)
    assert result3.shape == (32,)

def test_methods_stability():
    """测试方法在相同输入和seed下的稳定性"""
    rng = np.random.default_rng(42)
    features = [rng.normal(size=100) for _ in range(5)]
    
    methods = [
        (BiAVET(BiAVETConfig()), "BiAVET"),
        (InAVET(InAVETConfig()), "InAVET"),
        (BioHash(BioHashConfig()), "BioHash")
    ]
    
    for method, name in methods:
        # 对同一个特征进行多次转换
        for feat in features:
            result1 = method.process_feature(feat, seed=1)
            result2 = method.process_feature(feat, seed=1)
            np.testing.assert_array_equal(
                result1, 
                result2, 
                err_msg=f"{name} failed stability test"
            )