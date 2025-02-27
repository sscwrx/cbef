import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from experiment.base_experiment import Experiment, ExperimentConfig
from method.bio_hash import BioHashConfig
from data.face_dataset import FaceDatasetConfig
from verification.verify import VerifierConfig
from metrics.performance_metrics import PerformanceMetricsConfig

@pytest.fixture
def mock_configs():
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
    return ExperimentConfig(
        method_config=mock_configs['method_config'],
        dataset_config=mock_configs['dataset_config'],
        verifier_config=mock_configs['verifier_config'],
        metrics_config=mock_configs['metrics_config'],
        expriment_name="TestExperiment",
        output_dir=Path("./test_output"),
        expriment_times=2
    )

class TestExperimentConfig:
    def test_init(self, experiment_config):
        assert experiment_config.expriment_name == "TestExperiment"
        assert experiment_config.output_dir == Path("./test_output")
        assert experiment_config.expriment_times == 2

    def test_get_base_dir(self, experiment_config):
        base_dir = experiment_config._get_base_dir
        assert isinstance(base_dir, Path)
        assert str(base_dir).startswith(str(experiment_config.output_dir))

    @patch('builtins.open', create=True)
    def test_save_config(self, mock_open, experiment_config):
        experiment_config.save_config()
        mock_open.assert_called_once()
        mock_open.return_value.__enter__().write.assert_called_once()

class TestExperiment:
    @patch('pathlib.Path.mkdir')
    def test_init(self, mock_mkdir, experiment_config):
        with patch.object(experiment_config.dataset_config, 'setup') as mock_dataset_setup, \
             patch.object(experiment_config.method_config, 'setup') as mock_method_setup, \
             patch.object(experiment_config.verifier_config, 'setup') as mock_verifier_setup, \
             patch.object(experiment_config.metrics_config, 'setup') as mock_metrics_setup:
            
            experiment = Experiment(experiment_config)
            
            assert experiment.config == experiment_config
            mock_dataset_setup.assert_called_once()
            mock_method_setup.assert_called_once()
            mock_verifier_setup.assert_called_once()
            mock_metrics_setup.assert_called_once()

    @patch('numpy.save')
    def test_perform_generating(self, mock_save, experiment_config):
        with patch.object(experiment_config.dataset_config, 'setup') as mock_dataset_setup, \
             patch.object(experiment_config.method_config, 'setup') as mock_method_setup, \
             patch.object(experiment_config.verifier_config, 'setup') as mock_verifier_setup, \
             patch.object(experiment_config.metrics_config, 'setup') as mock_metrics_setup:
            
            # 设置mock数据
            mock_dataset = MagicMock()
            mock_dataset.load_data.return_value = {(1, 1): np.array([1, 2, 3])}
            mock_dataset.total_samples = 1
            mock_dataset_setup.return_value = mock_dataset
            
            mock_method = MagicMock()
            mock_method.process_feature.return_value = np.array([4, 5, 6])
            mock_method_setup.return_value = mock_method
            
            experiment = Experiment(experiment_config)
            mean_time = experiment.perform_generating(seed=1)
            
            assert isinstance(mean_time, float)
            mock_method.set_seed.assert_called_once_with(1)
            mock_dataset.load_data.assert_called_once()
            mock_save.assert_called_once()

    def test_run(self, experiment_config):
        with patch.object(experiment_config.dataset_config, 'setup') as mock_dataset_setup, \
             patch.object(experiment_config.method_config, 'setup') as mock_method_setup, \
             patch.object(experiment_config.verifier_config, 'setup') as mock_verifier_setup, \
             patch.object(experiment_config.metrics_config, 'setup') as mock_metrics_setup:
            
            # 设置mock返回值，确保生成具有非零方差的相似度值
            genuine_similarities = [0.8, 0.85, 0.9, 0.95]  # 真匹配相似度
            impostor_similarities = [0.1, 0.15, 0.2, 0.25]  # 假匹配相似度
            
            mock_verifier = MagicMock()
            mock_verifier.perform_matching.return_value = (
                genuine_similarities,
                impostor_similarities,
                0.001,  # mean_time_genuine
                0.001   # mean_time_impostor
            )
            mock_verifier_setup.return_value = mock_verifier
            
            mock_metrics = MagicMock()
            mock_metrics.perform_evaluation.return_value = (0.1, 0.5, [0.1], [0.9])
            mock_metrics.calculate_DI.return_value = 2.5  # 添加 calculate_DI 的返回值
            mock_metrics_setup.return_value = mock_metrics
            
            experiment = Experiment(experiment_config)
            
            # Mock perform_generating方法
            with patch.object(experiment, 'perform_generating', return_value=0.001):
                experiment.run()
                
                assert mock_verifier.perform_matching.call_count == experiment_config.expriment_times
                assert mock_metrics.perform_evaluation.call_count == experiment_config.expriment_times
                assert mock_metrics.calculate_DI.call_count == experiment_config.expriment_times  # 验证 calculate_DI 被调用

    def test_log_experiment_results(self, experiment_config):
        with patch.object(experiment_config.dataset_config, 'setup'), \
             patch.object(experiment_config.method_config, 'setup'), \
             patch.object(experiment_config.verifier_config, 'setup'), \
             patch.object(experiment_config.metrics_config, 'setup'):
            
            experiment = Experiment(experiment_config)
            mock_logger = MagicMock()
            
            eer_list = [0.1, 0.2]
            threshold_list = [0.5, 0.6]
            mean_time_list = [0.001, 0.002]
            
            experiment._log_experiment_results(
                mock_logger,
                eer_list,
                threshold_list,
                "TestDataset",
                mean_time_list
            )
            
            # 验证logger被调用
            assert mock_logger.info.call_count > 0