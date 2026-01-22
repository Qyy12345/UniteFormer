# 数据生成功能使用说明

## 概述

UniteFormer 的训练代码现在支持使用 **ML4CO-Kit 的数据生成器**来动态生成训练数据，无需预先准备数据集文件。

## 主要改进

### 1. **动态数据生成**
- 使用 `TSPDataGenerator` 或 `CVRPDataGenerator` 实时生成训练数据
- 支持多种数据分布（UNIFORM, GAUSSIAN, CLUSTER 等）
- 无需下载和管理大型数据集文件

### 2. **灵活配置**
可以轻松切换三种数据源模式：
- **数据生成模式**：使用 ML4CO-Kit 生成器（推荐用于训练）
- **数据集文件模式**：使用预先生成的数据集文件
- **原始模式**：使用 UniteFormer 原始的随机数据生成

## 使用方法

### TSP 训练

编辑 `UniteFormer/UF-TSP/train_tsp_ml4co.py`:

```python
# 方案 1: 使用数据生成器（推荐）
USE_DATA_GENERATOR = True
GENERATOR_CONFIG = {
    'distribution_type': TSP_TYPE.UNIFORM,  # 或 GAUSSIAN, CLUSTER
    'nodes_num': TSP_SIZE,
    'precision': 'float32',
}
USE_DATASET_FILE = False

# 方案 2: 使用数据集文件
USE_DATA_GENERATOR = False
USE_DATASET_FILE = True
DATASET_PATH = "../../ML4CO-Kit/test_dataset/routing/tsp/wrapper/tsp50_uniform_16ins.txt"

# 方案 3: 使用原始随机生成
USE_DATA_GENERATOR = False
USE_DATASET_FILE = False
```

运行训练：
```bash
cd UniteFormer/UF-TSP
python train_tsp_ml4co.py
```

### CVRP 训练

编辑 `UniteFormer/UF-CVRP/train_cvrp_ml4co.py`:

```python
# 使用数据生成器
USE_DATA_GENERATOR = True
GENERATOR_CONFIG = {
    'distribution_type': CVRP_TYPE.UNIFORM,
    'nodes_num': CVRP_SIZE,
    'capacity': 40.0,
    'precision': 'float32',
}
```

运行训练：
```bash
cd UniteFormer/UF-CVRP
python train_cvrp_ml4co.py
```

## 支持的数据分布类型

### TSP
- `TSP_TYPE.UNIFORM`: 均匀分布
- `TSP_TYPE.GAUSSIAN`: 高斯分布
- `TSP_TYPE.CLUSTER`: 聚类分布

### CVRP
- `CVRP_TYPE.UNIFORM`: 均匀分布
- `CVRP_TYPE.GAUSSIAN`: 高斯分布

## 环境变量

训练脚本需要添加 ML4CO-Kit 路径：
```python
sys.path.insert(0, "../../ML4CO-Kit")
```

## 优势

1. **无需预下载数据集**：训练时实时生成，节省存储空间
2. **灵活的数据分布**：轻松切换不同数据分布进行实验
3. **与 ML4CO-Bench-101 一致**：使用相同的数据生成工具，便于对比实验
4. **向后兼容**：仍可使用原始的数据加载方式

## 示例输出

```
================================================================================
Training UniteFormer on TSP50
================================================================================
Problem Size: 50
Data Generation: ENABLED
Generator: TSPDataGenerator
Distribution: TSP_TYPE.UNIFORM
Batch Size: 256
Epochs: 1010
================================================================================

✓ Created TSPDataGenerator with 50 nodes
✓ Data generator attached to environment
```

## 故障排除

### 问题：`ModuleNotFoundError: ml4co_kit`
**解决**：
```bash
pip install ml4co-kit==0.3.3
```

### 问题：数据生成失败
**解决**：脚本会自动回退到原始随机生成方式，检查日志中的警告信息

### 问题：内存不足
**解决**：减小 `train_batch_size` 的值

## 与 ML4CO-Bench-101 对比

现在可以使用与 ML4CO-Bench-101 相同的数据生成方式：

```python
# ML4CO-Bench-101 训练方式
from ml4co_kit import TSPDataGenerator, TSP_TYPE

generator = TSPDataGenerator(
    distribution_type=TSP_TYPE.UNIFORM,
    nodes_num=50
)
instances = generator.generate_only_instance_for_us(batch_size)
```

UniteFormer 现在也支持这种方式！
