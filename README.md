# Biometric Template Protection Method Evaluation Framework

This repository implements various biometric template protection methods and an evaluation framework.

## Quick Start

Clone the repository:

```bash
git clone https://gitee.com/futurecodes/avet.git
```

Install dependencies:

```bash
conda create -n env4cb python=3.11
conda activate env4cb
pip install -r requirements.txt
```

Run example experiment:

```python
from experiment.base_experiment import ExperimentConfig 
from data.face_dataset import FaceDatasetConfig 
from experiment.base_experiment import ExperimentConfig, Experiment
from config.method_configs import METHOD_CONFIGS

if __name__ == "__main__":
    dataset_config = FaceDatasetConfig(dataset_name="LFW") 
    method_config, verifier_config = METHOD_CONFIGS["SWG"]

    expr_config = ExperimentConfig(
        method_config=method_config,
        dataset_config=dataset_config,
        verifier_config=verifier_config,
        expriment_times=5
    )

    experiment: Experiment = expr_config.setup()
    experiment.run()
```

Note: The data has been preprocessed into 512-dimensional vectors using arcface feature extraction

## Features

- Supports multiple template protection methods:
  - BioHash
  - AVET
  - Bi-AVET
  - In-AVET
  - C-IOM
  - SWG
- Supports multiple similarity metrics:
  - Cosine similarity
  - Euclidean distance
  - Hamming distance
  - Jaccard distance
- Supports multiple datasets (e.g., LFW face dataset)
- Provides comprehensive performance evaluation metrics

## Implementation

### Experiment

- [x] Token Stolen Scenario (one experiment, one seed for **all** identity.)
- [ ] Normal Scenario (one experiment, one seed for **one** identity.)

### Method

- [x] SWG (Our method)
- [x] MLPHash
- [x] BioHash
- [x] AVET, LiAVET and BiAVET
- [x] LiAVET
- [x] C_IOM
- [ ] simHash
- [ ] DRH

### Matching  

- [x] Similarity (Score)
- [x] Bin plot (genuine, imposter)
- [ ] Bin plot (genuine, genuine-two seed) one sample with two seed for imposter matching examine Unlinkablity

### Metrics

#### Performance

- [x] FAR (False Accept Rate)
- [x] GAR (Genuine Accept Rate)
- [x] EER (Equal Error Rate) Used to evaluate the system's overall recognition accuracy and error rate
- [x] DI (Decidability Index)
- [ ] RI (Recognition Index)
- [ ] ROC curve (Receiver Operating Characteristic) Visually demonstrates system performance

#### Security

- [ ] Irreversibility
- [ ] Unlinkablity (Histogram analysis)
- [ ] Diversity
- [ ] Revocability

## Usage Guide

### 1. Data Preparation

Place the dataset embedding files in the correct directory structure:

```bash
embeddings/
  └── dataset_name/  # e.g., "LFW"
      └── user_id_sample_id.mat  # e.g., "1_1.mat"
```

### 2. Output Description

Experiment results will be saved in the output directory, including:

- Protected templates
- Performance evaluation results (EER, DI, etc.)
- Detailed log information

## Directory Structure

```bash
.
├── config/             # Configuration related code
├── data/              # Dataset processing related code
├── experiment/        # Experiment process related code
├── method/            # Template protection method implementations
├── metrics/           # Evaluation metrics calculation
├── tests/             # Test code
├── verification/      # Template matching related code
├── main.py           # Main program entry
```

## Notes

1. Ensure the dataset directory structure is correct, each sample file should be named in the format "{user_id}_{sample_id}.mat". (Starting from 1)
