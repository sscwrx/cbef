import pytest
import numpy as np
from pathlib import Path
import scipy as sp
from data.face_dataset import FaceDataset, FaceDatasetConfig

def test_face_dataset_load_data():
    """测试使用实际LFW数据集进行加载"""
    config = FaceDatasetConfig(
        dataset_name="LFW",
        embeddings_dir=Path("./embeddings"),
        n_subjects=127,
        samples_per_subject=12
    )
    
    dataset = FaceDataset(config)
    embeddings_dict = dataset.load_data()
    
    # 验证加载的数据
    assert len(embeddings_dict) > 0
    assert all(isinstance(k, tuple) and len(k) == 2 for k in embeddings_dict.keys())
    assert all(isinstance(v, np.ndarray) and v.shape == (512,) for v in embeddings_dict.values())


def test_face_dataset_invalid_path():
    """测试无效路径情况"""
    config = FaceDatasetConfig(
        dataset_name=None,
        embeddings_dir=Path("/nonexistent/path"),
        n_subjects=2,
        samples_per_subject=3
    )
    
    with pytest.raises(AssertionError):
        dataset = FaceDataset(config)

def test_face_dataset_no_dataset_name():
    """测试未提供数据集名称的情况"""
    config = FaceDatasetConfig(
        dataset_name=None,
        embeddings_dir=Path("/some/path"),
        n_subjects=2,
        samples_per_subject=3
    )
    
    with pytest.raises(AssertionError):
        dataset = FaceDataset(config)

if __name__ == "__main__":
    pytest.main()