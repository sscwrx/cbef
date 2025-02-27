# 生物特征模板保护方法评估框架

[English](README.md) | [中文](README_cn.md)

这个仓库实现了多种生物特征模板保护方法和评估框架。

## Quick Start

克隆仓库：

```bash
git clone https://gitee.com/futurecodes/avet.git
```

安装依赖：

```bash
conda create -n env4cb python=3.11
conda activate env4cb
pip install -r requirements.txt
```

运行示例实验：

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

注：数据已经过arcface特征提取成512维度向量

## 功能特性

- 支持多种模板保护方法：
  - BioHash
  - AVET
  - Bi-AVET
  - In-AVET
  - C-IOM
  - SWG
- 支持多种相似度度量：
  - 余弦相似度(cosine)
  - 欧氏距离(euclidean)
  - 汉明距离(hamming)
  - Jaccard距离
- 支持多个数据集（如LFW人脸数据集）
- 提供完整的性能评估指标

## Implementation

### Experiment

- [x] Token Stolen Scenario (one experiment, one seed for **all** identity.)
- [ ] Normal Scenario  (one experiment, one seed for **one** identity.)

### Method

- [x] SWG 本文方法
- [x] MLPHash
- [x] BioHash
- [x] AVET, LiAVET andBiAVET
- [x] LiAVET
- [x] C_IOM
- [ ] simHash
- [ ] DRH

### Matching  

- [x] Similarity (Score)
- [x] Bin plot (genuine, imposter)
- [ ] Bin plot (genuine,  genuine-two seed) one sample with two seed for imposter matching examine Unlinkablity

### Metrics

#### Performance

- [x] FAR (False Accept Rate)
- [x] GAR (Genuine Accept Rate)
- [x] EER (Equal Error Rate) 用于评估系统的整体识别准确性和错误率
- [x] DI (Decidability Index)
- [ ] RI (Recognition Index)
- [ ] ROC曲线（Receiver Operating Characteristic）直观展示系统的性能表现

#### Security

- [ ] 不可逆性 (Irreversibility)
- [ ] 不可链接性 (Unlinkablity) 柱状图分析
- [ ] 可区分性 (Diversity)
- [ ] 可撤销性 (Revocability)

## 使用说明

### 1. 数据准备

将数据集embedding文件放在正确的目录结构下：

```bash
embeddings/
  └── dataset_name/  # 例如 "LFW"
      └── user_id_sample_id.mat  # 例如 "1_1.mat"
```

### 2. 输出说明

实验结果会保存在output目录下，包括：

- 保护后的模板
- 性能评估结果（EER, DI等）
- 详细的日志信息

## 目录结构

```bash
.
├── config/             # 配置相关代码
├── data/              # 数据集处理相关代码
├── experiment/        # 实验流程相关代码
├── method/            # 模板保护方法实现
├── metrics/           # 评估指标计算
├── tests/             # 测试代码
├── verification/      # 模板匹配相关代码
├── main.py           # 主程序入口 
```

## 注意事项

1. 确保数据集目录结构正确，每个样本文件应按照"{user_id}_{sample_id}.mat"格式命名。(从1开始)
