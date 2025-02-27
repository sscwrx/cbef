import pytest
import numpy as np
from method.bi_avet import BiAVET, BiAVETConfig
from method.in_avet import InAVET, InAVETConfig
from method.bio_hash import BioHash, BioHashConfig
from method.chaos_based_IoM import ChaosBasedIoM, ChaosBasedIoMConfig

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
        (BiAVET(BiAVETConfig()), "BiAVET", False),
        (InAVET(InAVETConfig()), "InAVET", False),
        (BioHash(BioHashConfig()), "BioHash", False)
    ]

    for method, name, needs_seed_reset in methods:
        # 对同一个特征进行多次转换
        test_features = features
        
        # 对同一个特征进行多次转换
        for feat in test_features:
            if needs_seed_reset:
                method.set_seed(1)
            result1 = method.process_feature(feat, seed=1)
            if needs_seed_reset:
                method.set_seed(1)
            result2 = method.process_feature(feat, seed=1)
            np.testing.assert_array_equal(
                result1,
                result2,
                err_msg=f"{name} failed stability test"
            )

def test_chaos_based_iom_config():
    """测试混沌IoM配置参数计算"""
    config = ChaosBasedIoMConfig()
    
    # 测试派生参数计算
    expected_beta = ((config.R_on - config.R_off) * config.u_v * config.R_on) / config.D**2
    np.testing.assert_allclose(config.beta, expected_beta)
    
    # 验证N_3到N_6的计算
    expected_N3 = -(config.R_off - config.M0)**2 / (2 * config.beta)
    expected_N4 = -(config.R_on - config.M0)**2 / (2 * config.beta)
    expected_N5 = (config.R_off**2 - config.M0**2) / (2 * config.beta)
    expected_N6 = (config.R_on**2 - config.M0**2) / (2 * config.beta)
    
    np.testing.assert_allclose(config.N_3, expected_N3)
    np.testing.assert_allclose(config.N_4, expected_N4)
    np.testing.assert_allclose(config.N_5, expected_N5)
    np.testing.assert_allclose(config.N_6, expected_N6)

def test_chaos_based_iom_memristor_function():
    """测试混沌IoM的memristor非线性函数"""
    config = ChaosBasedIoMConfig()
    iom = ChaosBasedIoM(config)
    
    # 测试三个不同区间的函数行为
    epsilon_less = config.N_5 - 1
    epsilon_middle = (config.N_5 + config.N_6) / 2
    epsilon_greater = config.N_6 + 1
    
    # 测试 epsilon < N_5 区间
    result1 = iom._memristor_nonlinear_function(epsilon_less)
    expected1 = (epsilon_less - config.N_3) / config.R_off
    np.testing.assert_allclose(result1, expected1)
    
    # 测试 N_5 <= epsilon <= N_6 区间
    result2 = iom._memristor_nonlinear_function(epsilon_middle)
    expected2 = (np.sqrt(2 * config.beta * epsilon_middle + config.M0**2) - config.M0) / config.beta
    np.testing.assert_allclose(result2, expected2)
    
    # 测试 epsilon > N_6 区间
    result3 = iom._memristor_nonlinear_function(epsilon_greater)
    expected3 = (epsilon_greater - config.N_4) / config.R_on
    np.testing.assert_allclose(result3, expected3)

def test_chaos_based_iom_process(sample_feature):
    """测试混沌IoM特征处理"""
    # 创建512维的测试特征
    rng = np.random.default_rng(42)
    test_feature = rng.normal(size=512)

    config = ChaosBasedIoMConfig()
    iom = ChaosBasedIoM(config)

    # 测试维度
    result = iom.process_feature(test_feature)
    expected_dimension = config.dimension - config.window_size + 1
    assert result.shape == (expected_dimension,)

    # 测试不同的IoM类型
    config_max = ChaosBasedIoMConfig(IoM_type="max")
    iom_max = ChaosBasedIoM(config_max)
    result_max = iom_max.process_feature(test_feature)
    assert result_max.shape == (expected_dimension,)
