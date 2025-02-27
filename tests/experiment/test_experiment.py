import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime

from experiment.base_experiment import Experiment, ExperimentConfig
from method.bio_hash import BioHashConfig
from data.face_dataset import FaceDatasetConfig
from verification.verify import VerifierConfig
from metrics.performance_metrics import PerformanceMetricsConfig

# ============= Fixtures =============

@pytest.fixture
def mock_configs():
    """提供基础配置对象"""
    method_config = BioHashConfig()
    dataset_config = FaceDatasetConfig(dataset_name="LFW")
    verifier_config = VerifierConfig(measure="cosine")
    metrics_config = PerformanceMetricsConfig()
    
    return {
        'method_config': method_config,
        'dataset_config': dataset_config,
        'verifier_config': verifier_config,
        'metrics_config': metrics_config
    }

@pytest.fixture
def experiment_config(mock_configs):
    """提供实验配置对象"""
    return ExperimentConfig(
        method_config=mock_configs['method_config'],
        dataset_config=mock_configs['dataset_config'],
        verifier_config=mock_configs['verifier_config'],
        metrics_config=mock_configs['metrics_config'],
        expriment_name="TestExperiment",
        output_dir=Path("./test_output"),
        expriment_times=2
    )

@pytest.fixture
def mock_dependencies():
    """提供所有mock依赖"""
    mock_dataset = MagicMock()
    mock_dataset.load_data.return_value = {(1, 1): np.array([1, 2, 3])}
    mock_dataset.total_samples = 1
    mock_dataset.config = MagicMock()

    mock_method = MagicMock()
    mock_method.process_feature.return_value = np.array([4, 5, 6])

    mock_verifier = MagicMock()
    mock_verifier.perform_matching.return_value = (
        [0.8, 0.85, 0.9, 0.95],  # genuine similarities
        [0.1, 0.15, 0.2, 0.25],  # impostor similarities
        0.001,  # mean_time_genuine
        0.001   # mean_time_impostor
    )

    mock_metrics = MagicMock()
    mock_metrics.perform_evaluation.return_value = (
        0.1,  # EER
        0.5,  # threshold
        [0.0, 0.1, 0.2, 0.3, 0.4],  # FAR list
        [1.0, 0.9, 0.8, 0.7, 0.6]   # GAR list
    )
    mock_metrics.calculate_DI.return_value = 2.5
    
    return {
        'dataset': mock_dataset,
        'method': mock_method,
        'verifier': mock_verifier,
        'metrics': mock_metrics
    }

# ============= ExperimentConfig Tests =============

class TestExperimentConfig:
    def test_init(self, experiment_config):
        """测试ExperimentConfig初始化"""
        assert experiment_config.expriment_name == "TestExperiment"
        assert experiment_config.output_dir == Path("./test_output")
        assert experiment_config.expriment_times == 2
        assert experiment_config._target == Experiment

    def test_get_base_dir(self, experiment_config):
        """测试_get_base_dir属性"""
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            base_dir = experiment_config._get_base_dir
            assert isinstance(base_dir, Path)
            assert str(base_dir).startswith(str(experiment_config.output_dir))
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_save_config(self, experiment_config):
        """测试配置保存功能"""
        with patch('builtins.open', create=True) as mock_open:
            experiment_config.save_config()
            mock_open.assert_called_once()
            mock_open.return_value.__enter__().write.assert_called_once_with(str(experiment_config))

    def test_setup(self, experiment_config):
        """测试setup方法"""
        mock_experiment = MagicMock()
        experiment_config._target = mock_experiment
        experiment_config.setup()
        mock_experiment.assert_called_once_with(experiment_config)

# ============= Experiment Tests =============

class TestExperiment:
    @patch('pathlib.Path.mkdir')
    @patch('experiment.base_experiment.print_config')
    @patch('logging.basicConfig')
    def test_init(self, mock_logging, mock_print_config, mock_mkdir, experiment_config, mock_dependencies):
        """测试Experiment初始化"""
        with patch.object(experiment_config.dataset_config, 'setup') as mock_dataset_setup, \
             patch.object(experiment_config.method_config, 'setup') as mock_method_setup, \
             patch.object(experiment_config.verifier_config, 'setup') as mock_verifier_setup, \
             patch.object(experiment_config.metrics_config, 'setup') as mock_metrics_setup, \
             patch.object(experiment_config, 'save_config') as mock_save_config:

            mock_dataset_setup.return_value = mock_dependencies['dataset']
            mock_method_setup.return_value = mock_dependencies['method']
            mock_verifier_setup.return_value = mock_dependencies['verifier']
            mock_metrics_setup.return_value = mock_dependencies['metrics']
            
            experiment = Experiment(experiment_config)
            
            # 验证初始化逻辑
            assert experiment.config == experiment_config
            mock_dataset_setup.assert_called_once()
            mock_method_setup.assert_called_once()
            mock_verifier_setup.assert_called_once()
            mock_metrics_setup.assert_called_once()
            mock_save_config.assert_called_once()
            mock_print_config.assert_called_once_with(experiment)
            mock_logging.assert_called_once()
            mock_mkdir.assert_called()

    def test_perform_generating(self, experiment_config, mock_dependencies):
        """测试模板生成功能"""
        with patch('numpy.save') as mock_save, \
             patch('time.time', side_effect=[0, 1]), \
             patch.object(experiment_config.dataset_config, 'setup') as mock_dataset_setup, \
             patch.object(experiment_config.method_config, 'setup') as mock_method_setup, \
             patch.object(experiment_config.verifier_config, 'setup') as mock_verifier_setup, \
             patch.object(experiment_config.metrics_config, 'setup') as mock_metrics_setup, \
             patch('pathlib.Path.mkdir'):
            
            mock_dataset_setup.return_value = mock_dependencies['dataset']
            mock_method_setup.return_value = mock_dependencies['method']
            mock_verifier_setup.return_value = mock_dependencies['verifier']
            mock_metrics_setup.return_value = mock_dependencies['metrics']
            
            experiment = Experiment(experiment_config)
            mean_time = experiment.perform_generating(seed=1)
            
            # 验证功能执行
            assert isinstance(mean_time, float)
            assert mean_time == 1.0  # (1 - 0) / 1 sample
            mock_dependencies['method'].set_seed.assert_called_once_with(1)
            mock_dependencies['dataset'].load_data.assert_called_once()
            
            # 使用np.array_equal来比较numpy数组
            actual_args = mock_dependencies['method'].process_feature.call_args[0][0]
            expected_args = np.array([1, 2, 3])
            assert np.array_equal(actual_args, expected_args)
            mock_save.assert_called_once()

    @patch('experiment.base_experiment.plot_score_distributions')
    @patch('experiment.base_experiment.print_result_table')
    @patch('experiment.base_experiment.log_experiment_results')
    def test_run(self, mock_log_results, mock_print_table, mock_plot, experiment_config, mock_dependencies):
        """测试完整实验流程"""
        with patch.object(experiment_config.dataset_config, 'setup') as mock_dataset_setup, \
             patch.object(experiment_config.method_config, 'setup') as mock_method_setup, \
             patch.object(experiment_config.verifier_config, 'setup') as mock_verifier_setup, \
             patch.object(experiment_config.metrics_config, 'setup') as mock_metrics_setup, \
             patch('pathlib.Path.mkdir'):
            
            mock_dataset_setup.return_value = mock_dependencies['dataset']
            mock_method_setup.return_value = mock_dependencies['method']
            mock_verifier_setup.return_value = mock_dependencies['verifier']
            mock_metrics_setup.return_value = mock_dependencies['metrics']
            
            experiment = Experiment(experiment_config)
            
            with patch.object(experiment, 'perform_generating', return_value=0.001):
                experiment.run()
                
                # 验证方法调用次数
                assert mock_dependencies['verifier'].perform_matching.call_count == experiment_config.expriment_times
                assert mock_dependencies['metrics'].perform_evaluation.call_count == experiment_config.expriment_times
                assert mock_dependencies['metrics'].calculate_DI.call_count == experiment_config.expriment_times
                assert mock_print_table.call_count == experiment_config.expriment_times
                assert mock_plot.call_count == 1

                # 验证最终日志记录的参数
                mock_log_results.assert_called_once()
                args = mock_log_results.call_args[0]
                assert len(args) == 7  # experiment, logger, eer_list, threshold_list, dataset_name, time_list, di_list
                assert isinstance(args[0], Experiment)  # experiment
                assert all(isinstance(x, list) for x in [args[2], args[3], args[5], args[6]])  # 验证列表参数
                assert isinstance(args[4], str)  # dataset_name

    @patch('experiment.base_experiment.plot_score_distributions')
    def test_edge_cases(self, mock_plot, experiment_config, mock_dependencies):
        """测试边界情况"""
        with patch.object(experiment_config.dataset_config, 'setup') as mock_dataset_setup, \
             patch.object(experiment_config.method_config, 'setup') as mock_method_setup, \
             patch.object(experiment_config.verifier_config, 'setup') as mock_verifier_setup, \
             patch.object(experiment_config.metrics_config, 'setup') as mock_metrics_setup, \
             patch('pathlib.Path.mkdir'), \
             patch('time.time', side_effect=[0, 1]):
            
            # 测试空数据集
            mock_dependencies['dataset'].load_data.return_value = {}
            mock_dependencies['dataset'].total_samples = 0
            mock_dataset_setup.return_value = mock_dependencies['dataset']
            mock_method_setup.return_value = mock_dependencies['method']
            mock_verifier_setup.return_value = mock_dependencies['verifier']
            mock_metrics_setup.return_value = mock_dependencies['metrics']
            
            experiment = Experiment(experiment_config)
            mean_time = experiment.perform_generating(seed=1)
            
            # 空数据集应该返回0而不是抛出异常
            assert mean_time == 0
            
            # 测试相似度值为空的情况
            mock_dependencies['verifier'].perform_matching.return_value = ([], [], 0, 0)
            mock_dependencies['metrics'].perform_evaluation.return_value = (0, 0, [], [])
            
            with patch.object(experiment, 'perform_generating', return_value=0):
                experiment.run()  # 不应抛出异常