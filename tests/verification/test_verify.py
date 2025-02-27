import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import math

from verification.verify import Verifier, VerifierConfig
from data.fingerprint_dataset import FingerprintDatasetConfig

@pytest.fixture
def verifier_config():
    return VerifierConfig(
        _target=Verifier,
        measure="cosine",
        protected_template_dir=Path("./test_templates")
    )

@pytest.fixture
def data_config():
    return FingerprintDatasetConfig(
        n_subjects=10,
        samples_per_subject=4,
        embeddings_dir=Path("./test_embeddings")
    )

class TestVerifier:
    def test_calculate_template_similarity_cosine(self, verifier_config):
        verifier = Verifier(verifier_config)
        template1 = np.array([1.0, 0.0])
        template2 = np.array([1.0, 0.0])
        similarity = verifier._calculate_template_similarity(template1, template2)
        assert np.isclose(similarity, 1.0)

    def test_calculate_template_similarity_euclidean(self, verifier_config):
        verifier_config.measure = "euclidean"
        verifier = Verifier(verifier_config)
        template1 = np.array([1.0, 0.0])
        template2 = np.array([0.0, 1.0])
        similarity = verifier._calculate_template_similarity(template1, template2)
        assert np.isclose(similarity, -np.sqrt(2))

    def test_calculate_template_similarity_hamming(self, verifier_config):
        verifier_config.measure = "hamming"
        verifier = Verifier(verifier_config)
        template1 = np.array([1, 0, 1])
        template2 = np.array([1, 1, 0])
        similarity = verifier._calculate_template_similarity(template1, template2)
        assert np.isclose(similarity, 1/3)

    def test_calculate_template_similarity_jaccard(self, verifier_config):
        verifier_config.measure = "jaccard"
        verifier = Verifier(verifier_config)
        template1 = np.array([1, 0, 1])
        template2 = np.array([1, 1, 0])
        similarity = verifier._calculate_template_similarity(template1, template2)
        assert np.isclose(similarity, 1/5)

    def test_calculate_template_similarity_length_mismatch(self, verifier_config):
        verifier = Verifier(verifier_config)
        template1 = np.array([1.0, 0.0])
        template2 = np.array([1.0])
        with pytest.raises(ValueError, match="模板长度不匹配"):
            verifier._calculate_template_similarity(template1, template2)

    @patch("numpy.load")
    def test_perform_matching(self, mock_load, verifier_config, data_config):
        verifier = Verifier(verifier_config)
        verifier.data_config = data_config
        
        # 设置mock返回值
        template1 = np.array([1.0, 0.0])
        template2 = np.array([1.0, 0.0])
        mock_load.return_value = template1
        
        genuine_sim, impostor_sim, time_gen, time_imp = verifier.perform_matching()
        
        assert len(genuine_sim) == verifier.n_genuines_combinations
        assert len(impostor_sim) == verifier.n_impostor_combinations
        assert isinstance(time_gen, float)
        assert isinstance(time_imp, float)

    def test_n_genuines_combinations(self, verifier_config, data_config):
        verifier = Verifier(verifier_config)
        verifier.data_config = data_config
        expected = data_config.n_subjects * math.comb(data_config.samples_per_subject, 2)
        assert verifier.n_genuines_combinations == expected

    def test_n_impostor_combinations(self, verifier_config, data_config):
        verifier = Verifier(verifier_config)
        verifier.data_config = data_config
        expected = math.comb(data_config.n_subjects, 2)
        assert verifier.n_impostor_combinations == expected