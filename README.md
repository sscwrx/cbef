
# 快速开始

克隆仓库

```bash
git clone https://gitee.com/futurecodes/avet.git
```

安装依赖

```bash
conda create -n env4cb python=3.11
conda activate env4cb
pip install -r requirements.txt

```

对单个方法 单个数据集进行验证

```python
from method.swg import SWGConfig
from data.face_dataset import FaceDatasetConfig 
from metrics.performance.eer_metrics import EERMetricsConfig
from experiment.base_experiment import ExperimentConfig

# 1. 配置各组件
method_config = SWGConfig()  
dataset_config = FaceDatasetConfig(dataset_name="LFW") 
metrics_config = EERMetricsConfig(measure="hamming")  

# 2. 创建实验配置
expr_config = ExperimentConfig(
    expriment_name="SWG_LFW_Test",
    method_config=method_config,
    dataset_config=dataset_config,
    metrics_config=metrics_config,
    expriment_times=5  # 重复实验5次
)

# 3. 运行实验
experiment = expr_config.setup()
experiment.run()
```
`数据已经过arcface特征提取成512维度向量`