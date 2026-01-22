# UniteFormer with ML4CO Dataset Integration

This document describes the modifications made to UniteFormer to enable training and testing with ML4CO datasets.

## Overview

The integration allows UniteFormer to:
1. Load and use ML4CO-Bench-101 datasets for TSP and CVRP problems
2. Leverage ML4CO-Kit's unified data management framework
3. Maintain compatibility with original UniteFormer training pipeline
4. Easily switch between original and ML4CO data formats

## File Structure

```
UniteFormer/
├─ utils/
│  └─ ml4co_data_loader.py          # Data loading utilities for ML4CO datasets
├─ UF-TSP/
│  ├─ TSPEnv_ML4CO.py               # Enhanced TSP environment with ML4CO support
│  ├─ train_tsp_ml4co.py            # Training script for TSP with ML4CO data
│  └─ test_tsp_ml4co.py             # Testing script for TSP with ML4CO data
└─ UF-CVRP/
   ├─ CVRPEnv_ML4CO.py              # Enhanced CVRP environment with ML4CO support
   ├─ train_cvrp_ml4co.py           # Training script for CVRP with ML4CO data
   └─ test_cvrp_ml4co.py            # Testing script for CVRP with ML4CO data
```

## Installation

### 1. Install ML4CO-Kit

First, install the ML4CO-Kit package:

```bash
pip install ml4co-kit==0.3.3
```

### 2. Download ML4CO Datasets

Download the ML4CO-Bench-101 datasets from Hugging Face:

**Training datasets:**
```bash
# Create data directory
mkdir -p UniteFormer/data/tsp
mkdir -p UniteFormer/data/cvrp

# Download from Hugging Face
# Visit: https://huggingface.co/datasets/ML4CO/ML4CO-Bench-101-SL/tree/main/train_dataset
```

**Testing datasets:**
```bash
# Visit: https://huggingface.co/datasets/ML4CO/ML4CO-Bench-101-SL/tree/main/test_dataset
```

Alternatively, you can use the datasets already available in `ML4CO-Bench-101/test_dataset/`.

## Usage

### TSP Training with ML4CO Dataset

1. **Edit the training script:**

```bash
cd UniteFormer/UF-TSP
nano train_tsp_ml4co.py
```

2. **Configure parameters:**

```python
# Problem Configuration
TSP_SIZE = 50  # Options: 20, 50, 100, 500, 1000

# Dataset Configuration
USE_ML4CO = True
ML4CO_DATASET_PATH = "../../ML4CO-Bench-101/test_dataset/tsp/tsp50_concorde_5.688.txt"

# Training Parameters
trainer_params = {
    'train_batch_size': 256,  # Adjust based on GPU memory
    'epochs': 1010,
    'train_episodes': 1000 * 100,
}
```

3. **Run training:**

```bash
python train_tsp_ml4co.py
```

### TSP Testing with ML4CO Dataset

1. **Edit the testing script:**

```bash
nano test_tsp_ml4co.py
```

2. **Configure parameters:**

```python
# Problem Configuration
TSP_SIZE = 50

# Dataset Configuration
USE_ML4CO = True
ML4CO_TEST_DATASET = "../../ML4CO-Bench-101/test_dataset/tsp/tsp50_concorde_5.688.txt"

# Model Configuration
MODEL_PATH = "./train_models/tsp50/epoch-1010.pkl"
```

3. **Run testing:**

```bash
python test_tsp_ml4co.py
```

### CVRP Training with ML4CO Dataset

1. **Prepare CVRP dataset:**

CVRP datasets in ML4CO may be in `.pkl` format. Ensure you have the correct dataset path.

2. **Edit training script:**

```bash
cd UniteFormer/UF-CVRP
nano train_cvrp_ml4co.py
```

3. **Configure and run:**

```python
CVRP_SIZE = 50
ML4CO_DATASET_PATH = "data/cvrp50_instances.pkl"
```

```bash
python train_cvrp_ml4co.py
```

## Key Features

### 1. ML4CODataLoader

The `ML4CODataLoader` class (`utils/ml4co_data_loader.py`) provides:

- **load_tsp_data()**: Load TSP datasets from ML4CO format
- **load_cvrp_data()**: Load CVRP datasets from ML4CO format
- **load_raw_tsp_for_uniteformer()**: Convert to UniteFormer's raw format
- **load_raw_cvrp_for_uniteformer()**: Convert CVRP data for UniteFormer

### 2. Enhanced Environments

**TSPEnvML4CO** and **CVRPEnvML4CO** extend the original environments with:

- `use_ml4co` parameter: Enable/disable ML4CO dataset support
- `ml4co_data_format` parameter: Choose loading method ('wrapper' or 'direct')
- Backward compatibility with original UniteFormer data format

### 3. Custom Trainers and Testers

The integration includes custom trainer/tester classes that:
- Inherit from original UniteFormer classes
- Use ML4CO-enabled environments
- Maintain all original functionality

## Data Format Compatibility

### Original UniteFormer Format

**TSP:**
```
x1 y1 x2 y2 ... xn yn output tour1 tour2 ... tourn
```

**CVRP:**
```
depot x1 y1 customer x2 y2 ... capacity C demand d1 d2 ... cost C node_flag f1 f2 ...
```

### ML4CO Format

**TSP:**
```
x1 y1 x2 y2 ... xn yn output tour1 tour2 ... tourn
```
(Compatible with UniteFormer)

**CVRP:**
```
depots [depots] points [points] demands [demands] capacity [capacity] output [sol]
```
(Automatically converted by ML4CODataLoader)

## Configuration Options

### Environment Parameters

```python
env_params = {
    'problem_size': 50,              # Number of nodes
    'pomo_size': 50,                 # POMO size (usually = problem_size)
    'num_neighbors': -1,             # -1 for dense, or K for sparse K-nearest
    'data_path': 'path/to/dataset',  # Path to ML4CO dataset
    'mode': 'train',                 # 'train' or 'test'
    'use_ml4co': True,               # Enable ML4CO dataset support
    'ml4co_data_format': 'wrapper',  # 'wrapper' or 'direct'
}
```

### Training Parameters

```python
trainer_params = {
    'train_batch_size': 256,         # Batch size (adjust based on GPU)
    'epochs': 1010,                   # Number of training epochs
    'train_episodes': 100000,         # Total training episodes
    'use_cuda': True,
    'cuda_device_num': 0,
}
```

### Testing Parameters

```python
tester_params = {
    'test_batch_size': 1,
    'test_episodes': 100,
    'augmentation_enable': True,      # Enable 8-fold augmentation
    'use_cuda': True,
    'cuda_device_num': 0,
}
```

## Advantages of ML4CO Integration

1. **Unified Framework**: Use ML4CO's standardized data format and evaluation protocols
2. **Benchmark Compatibility**: Compare results with ML4CO-Bench-101 baselines
3. **Multiple Datasets**: Access to 34 datasets for 7 CO problems
4. **Standardized Evaluation**: Consistent evaluation metrics across different methods
5. **Easy Switching**: Toggle between original and ML4CO formats with a single parameter

## Troubleshooting

### Issue: ML4CO-Kit not found

**Solution:**
```bash
pip install ml4co-kit==0.3.3
```

### Issue: Dataset file not found

**Solution:**
- Check that the dataset path is correct (relative to the script location)
- Ensure datasets are downloaded from Hugging Face
- Verify file permissions

### Issue: CUDA out of memory

**Solution:**
- Reduce `train_batch_size` in training scripts
- Reduce `problem_size` for testing
- Use a smaller model configuration

### Issue: Data format mismatch

**Solution:**
- Set `use_ml4co=False` for original UniteFormer format
- Set `ml4co_data_format='direct'` for direct file reading
- Check that your dataset matches the expected format

## Citation

If you use this integration, please cite both papers:

```bibtex
@inproceedings{
    ma2025mlcobench,
    title={ML4CO-Bench-101: Benchmark Machine Learning for Classic Combinatorial Problems on Graphs},
    author={Jiale Ma and Wenzheng Pan and Yang Li and Junchi Yan},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2025},
    url={https://openreview.net/forum?id=ye4ntB1Kzi}
}

@inproceedings{li2025unify,
  title={UniteFormer: Unifying Node and Edge Modalities in Transformers for Vehicle Routing Problems},
  booktitle={NeurIPS},
  year={2025}
}
```

## Contact

For questions or issues:
- ML4CO-Bench-101: https://github.com/Thinklab-SJTU/ML4CO-Bench-101
- ML4CO-Kit: https://github.com/Thinklab-SJTU/ML4CO-Kit
- UniteFormer: Check original repository

## License

This integration maintains the licenses of both original projects:
- ML4CO-Bench-101 and ML4CO-Kit: See original repository licenses
- UniteFormer: See original repository license
