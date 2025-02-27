import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Type

from method.base_method import BaseMethod, MethodConfig
from method.baseline import Baseline, BaselineConfig

class TestBaseMethod:
    def test_set_seed(self):
        # 创建一个具体的BaseMethod实现用于测试
        class ConcreteMethod(BaseMethod):
            def process_feature(self, feature_vector, seed=1):
                return feature_vector
        
        # 使用Mock创建配置
        mock_config = Mock()
        mock_config.seed = 1
        
        method = ConcreteMethod(mock_config)
        method.set_seed(42)
        assert method.config.seed == 42

class TestBaseline:
    @pytest.fixture
    def baseline_config(self):
        return BaselineConfig(_target=Baseline)

    def test_init(self, baseline_config):
        method = Baseline(baseline_config)
        assert method.config.method_name == "Baseline"
        assert hasattr(method.config, 'seed')

    def test_process_feature(self, baseline_config):
        method = Baseline(baseline_config)
        feature = np.array([1.0, 2.0, 3.0])
        processed = method.process_feature(feature)
        assert np.array_equal(processed, feature)
        
    def test_process_feature_with_different_seed(self, baseline_config):
        method = Baseline(baseline_config)
        feature = np.array([1.0, 2.0, 3.0])
        processed1 = method.process_feature(feature, seed=1)
        processed2 = method.process_feature(feature, seed=42)
        # Baseline应该不管seed是什么值都返回相同的结果
        assert np.array_equal(processed1, processed2)