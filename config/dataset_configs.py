from typing import Dict,List
from data.base_dataset import BaseDatasetConfig
from data.face_dataset import FaceDatasetConfig
from data.fingerprint_dataset import FingerprintDatasetConfig


dataset_names:List[str] = ["LFW","FEI","CASIA-WebFace","ColorFeret",
                           "FVC2002/Db1_a","FVC2002/Db2_a","FVC2002/Db3_a",
                           "FVC2004/Db1_a","FVC2004/Db2_a","FVC2004/Db3_a",]

DATASET_CONFIGS:Dict[str,BaseDatasetConfig] = {
    "LFW":FaceDatasetConfig(dataset_name="LFW"),
    "FEI":FaceDatasetConfig(dataset_name="FEI"),
    "CASIA-WebFace":FaceDatasetConfig(dataset_name="CASIA-WebFace"),
    "ColorFeret":FaceDatasetConfig(dataset_name="ColorFeret"),
    
    "FVC2002/Db1_a":FingerprintDatasetConfig(dataset_name="FVC2002/Db1_a"),
    "FVC2002/Db2_a":FingerprintDatasetConfig(dataset_name="FVC2002/Db2_a"),
    "FVC2002/Db3_a":FingerprintDatasetConfig(dataset_name="FVC2002/Db3_a"),
    "FVC2004/Db1_a":FingerprintDatasetConfig(dataset_name="FVC2004/Db1_a"),
    "FVC2004/Db2_a":FingerprintDatasetConfig(dataset_name="FVC2004/Db2_a"),
    "FVC2004/Db3_a":FingerprintDatasetConfig(dataset_name="FVC2004/Db3_a"),
}