import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from data.base_dataset import BaseDataset, BaseDatasetConfig
from data.face_dataset import FaceDataset, FaceDatasetConfig

@pytest.fixture
def base_config():
    return BaseDatasetConfig(
        n_subjects=10,
        samples_per_subject=4,
        embeddings_dir=Path("./test_embeddings")
    )

@pytest.fixture
def face_config():
    return FaceDatasetConfig(
        dataset_name="LFW",
        n_subjects=10,
        samples_per_subject=4,
        embedding_dim=512,
        embeddings_dir=Path("./test_embeddings")
    )

class TestBaseDataset:
    def test_total_samples(self, base_config):
        dataset = BaseDataset(base_config)
        assert dataset.total_samples == 40  # 10 subjects * 4 samples

class TestFaceDataset:
    def test_init_without_dataset_name(self):
        config = FaceDatasetConfig(
            dataset_name=None,
            n_subjects=10,
            samples_per_subject=4,
            embeddings_dir=Path("./test_embeddings")
        )
        with pytest.raises(AssertionError, match="Dataset name must be provided"):
            FaceDataset(config)

    @patch("pathlib.Path.exists")
    def test_init_with_nonexistent_dir(self, mock_exists, face_config):
        mock_exists.return_value = False
        with pytest.raises(AssertionError, match="Embeddings directory does not exist"):
            FaceDataset(face_config)

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.glob")
    @patch("scipy.io.loadmat")
    def test_load_data(self, mock_loadmat, mock_glob, mock_exists, face_config):
        # 设置mock返回值
        mock_exists.return_value = True
        mock_glob.return_value = [Path("./test_embeddings/LFW/1_1.mat")]
        mock_loadmat.return_value = {"preOut": np.array([[[1, 2, 3]]])}

        dataset = FaceDataset(face_config)
        embeddings = dataset.load_data()

        assert isinstance(embeddings, dict)
        assert (1, 1) in embeddings
        assert embeddings[(1, 1)].shape == (3,)
        mock_loadmat.assert_called_once()

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.glob")
    def test_load_data_with_nonexistent_file(self, mock_glob, mock_exists, face_config):
        mock_exists.side_effect = [True, True, False]  # dir exists, but file doesn't
        mock_glob.return_value = [Path("./test_embeddings/LFW/1_1.mat")]

        dataset = FaceDataset(face_config)
        with pytest.raises(AssertionError, match=".*does not exist"):
            dataset.load_data()