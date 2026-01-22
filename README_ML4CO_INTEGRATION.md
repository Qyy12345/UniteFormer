# UniteFormer with ML4CO Dataset Integration

This document describes the modifications made to UniteFormer to enable training and testing with ML4CO datasets and data generation capabilities.

## 🎯 Overview

The integration allows UniteFormer to:
1. **Generate training data dynamically** using ML4CO-Kit's data generators (NEW!)
2. Load and use ML4CO-Bench-101 datasets for TSP and CVRP problems
3. Leverage ML4CO-Kit's unified data management framework
4. Maintain compatibility with original UniteFormer training pipeline
5. Easily switch between data sources (generation, files, or original)

## ✨ Key Features

### 1. Dynamic Data Generation (New!)
- **No dataset files required** - Generate data on-the-fly
- **Multiple distributions** - UNIFORM, GAUSSIAN, CLUSTER
- **Same as ML4CO-Bench-101** - Use identical data generation methods
- **Memory efficient** - No need to store large datasets

### 2. Dataset File Support
- Load ML4CO-Bench-101 datasets
- Load ML4CO-Kit test datasets
- Automatic format conversion

### 3. Backward Compatibility
- Original UniteFormer code still works
- Switch between modes with simple parameters
- No breaking changes

## 📁 File Structure

```
UniteFormer/
├─ utils/
│  └─ ml4co_data_loader.py          # Data loading utilities
├─ examples/
│  └─ example_data_loading.py       # Usage examples
├─ UF-TSP/
│  ├─ TSPEnv_ML4CO.py               # Enhanced TSP environment
│  ├─ train_tsp_ml4co.py            # Training script (with data generation!)
│  └─ test_tsp_ml4co.py             # Testing script
├─ UF-CVRP/
│  ├─ CVRPEnv_ML4CO.py              # Enhanced CVRP environment
│  ├─ train_cvrp_ml4co.py           # Training script (with data generation!)
│  └─ test_cvrp_ml4co.py            # Testing script
├─ README_ML4CO_INTEGRATION.md      # This file
├─ QUICKSTART_ML4CO.md              # Quick start guide
├─ DATA_GENERATION_GUIDE.md         # Data generation guide
└─ MODIFICATIONS_SUMMARY.md         # Complete modification summary
```

## 🚀 Installation

### 1. Install ML4CO-Kit

```bash
pip install ml4co-kit==0.3.3
```

### 2. (Optional) Download ML4CO Datasets

**For testing with dataset files:**

```bash
# ML4CO-Kit test datasets are already available at:
ls ../../ML4CO-Kit/test_dataset/routing/tsp/wrapper/

# For ML4CO-Bench-101 datasets, visit:
https://huggingface.co/datasets/ML4CO/ML4CO-Bench-101-SL
```

**Note:** With data generation, you don't need to download datasets!

## 💡 Usage

### Training with Data Generation (Recommended)

This is the **easiest way** to train UniteFormer!

#### TSP

```bash
cd UniteFormer/UF-TSP
```

Edit `train_tsp_ml4co.py`:

```python
# Configuration
TSP_SIZE = 50

# Enable data generation
USE_DATA_GENERATOR = True

# Generator settings
GENERATOR_CONFIG = {
    'distribution_type': TSP_TYPE.UNIFORM,  # Options: UNIFORM, GAUSSIAN, CLUSTER
    'nodes_num': TSP_SIZE,
    'precision': 'float32',
}

# No dataset file needed
USE_DATASET_FILE = False
```

Run training:

```bash
python train_tsp_ml4co.py
```

#### CVRP

```bash
cd UniteFormer/UF-CVRP
```

Edit `train_cvrp_ml4co.py`:

```python
# Configuration
CVRP_SIZE = 50

# Enable data generation
USE_DATA_GENERATOR = True

# Generator settings
GENERATOR_CONFIG = {
    'distribution_type': CVRP_TYPE.UNIFORM,
    'nodes_num': CVRP_SIZE,
    'capacity': 40.0,
    'precision': 'float32',
}
```

Run training:

```bash
python train_cvrp_ml4co.py
```

### Training with Dataset Files

If you prefer to use dataset files:

```python
# Disable data generation
USE_DATA_GENERATOR = False

# Enable dataset file loading
USE_DATASET_FILE = True
DATASET_PATH = "../../ML4CO-Kit/test_dataset/routing/tsp/wrapper/tsp50_uniform_16ins.txt"
```

### Testing

```python
# Test with ML4CO-Kit datasets
USE_ML4CO = True
ML4CO_TEST_DATASET = "../../ML4CO-Kit/test_dataset/routing/tsp/wrapper/tsp50_uniform_16ins.txt"
```

```bash
python test_tsp_ml4co.py
```

## 🔧 Configuration Options

### Data Source Selection

```python
env_params = {
    # Data generation (NEW!)
    'use_data_generator': True,  # Enable ML4CO data generation
    'generator_config': {
        'distribution_type': TSP_TYPE.UNIFORM,
        'nodes_num': 50,
        'precision': 'float32',
    },

    # Dataset file (alternative)
    'use_ml4co': False,  # Enable for dataset file loading
    'data_path': 'path/to/dataset.txt',

    # Original method (fallback)
    # If both above are False, uses original random generation
}
```

### Training Parameters

```python
trainer_params = {
    'train_batch_size': 256,  # Adjust based on GPU memory
    'epochs': 1010,
    'train_episodes': 100000,
}
```

### Data Distribution Options

#### TSP
- `TSP_TYPE.UNIFORM` - Random uniform distribution (default)
- `TSP_TYPE.GAUSSIAN` - Gaussian/Normal distribution
- `TSP_TYPE.CLUSTER` - Clustered distribution

#### CVRP
- `CVRP_TYPE.UNIFORM` - Random uniform distribution (default)
- `CVRP_TYPE.GAUSSIAN` - Gaussian distribution

## 📊 Data Format Compatibility

### Original UniteFormer Format
```
TSP: x1 y1 x2 y2 ... xn yn output t1 t2 ... tn
CVRP: depot x1 y1 ... capacity C demand d1 ... output tour
```

### ML4CO Format
```
TSP: x1 y1 x2 y2 ... xn yn output t1 t2 ... tn (compatible!)
CVRP: depots [...] points [...] demands [...] capacity [...] output [...]
```

**Automatic Conversion:** The `ML4CODataLoader` handles format conversion automatically.

## 🎓 Key Components

### 1. ML4CODataLoader
Location: `utils/ml4co_data_loader.py`

Provides:
- `load_tsp_data()` - Load TSP datasets
- `load_cvrp_data()` - Load CVRP datasets
- `load_raw_tsp_for_uniteformer()` - Convert to UniteFormer format
- `load_raw_cvrp_for_uniteformer()` - Convert CVRP data

### 2. Enhanced Environments
- `TSPEnv_ML4CO` - TSP environment with data generation support
- `CVRPEnv_ML4CO` - CVRP environment with data generation support

New parameters:
```python
env_params = {
    'use_data_generator': True,  # Use data generator
    'generator_config': {...},   # Generator configuration
}
```

### 3. Custom Trainers/Testers
- `TSPTrainerML4CO` - TSP trainer with ML4CO support
- `TSPTesterML4CO` - TSP tester with ML4CO support
- `CVRPTrainerML4CO` - CVRP trainer with ML4CO support
- `CVRPTesterML4CO` - CVRP tester with ML4CO support

## 🔍 How It Works

### Data Generation Flow

```
Training Start
    ↓
Check USE_DATA_GENERATOR
    ↓
[True] → Create TSPDataGenerator/CVRPDataGenerator
    ↓
    For each batch:
        generator.generate_only_instance_for_us(batch_size)
        ↓
        Convert to PyTorch tensors
        ↓
        Forward pass
    ↓
[False] → Check USE_DATASET_FILE
    ↓
    [True] → Load from file
    [False] → Use original random generation
```

### Environment Integration

```python
# Environment initialization
env = TSPEnv_ML4CO(**env_params)

# Attach data generator (if using)
if USE_DATA_GENERATOR:
    env.data_generator = data_generator

# Load problems (automatically chooses data source)
env.load_problems(episode=0, batch_size=32)
```

## ✅ Advantages of This Integration

### 1. Data Generation Benefits
| Feature | Data Generation | Dataset Files |
|---------|----------------|---------------|
| **Setup** | ✅ No download needed | ❌ Must download |
| **Storage** | ✅ Minimal | ❌ Large files |
| **Flexibility** | ✅ Multiple distributions | ❌ Fixed data |
| **ML4CO-Bench-101 compatible** | ✅ Same method | ✅ If using ML4CO data |
| **Reproducibility** | ✅ Set seed | ✅ Deterministic |

### 2. Unified Framework
- Use ML4CO's standardized data format and evaluation
- Compare with ML4CO-Bench-101 baselines
- Access to 34 datasets across 7 problems

### 3. Backward Compatibility
- Original UniteFormer code unchanged
- Easy to switch between modes
- No breaking changes

## 🛠️ Troubleshooting

### Issue: ModuleNotFoundError: ml4co_kit
**Solution:**
```bash
pip install ml4co-kit==0.3.3
```

### Issue: CUDA out of memory
**Solution:**
```python
'train_batch_size': 64  # Reduce batch size
```

### Issue: Data generation fails
**Solution:** Script automatically falls back to random generation. Check logs:
```
Warning: Data generation failed (...), falling back to random generation
```

### Issue: Dataset file not found
**Solution:**
- Use ML4CO-Kit datasets (already available)
- Or download ML4CO-Bench-101 datasets
- Check file path is correct (relative to script location)

## 📈 Performance Expectations

Based on ML4CO-Bench-101 with **UNIFORM distribution** and data generation:

| Problem | Size | Optimal | Expected |
|---------|------|---------|----------|
| TSP     | 20   | ~3.84   | ~3.85    |
| TSP     | 50   | ~5.69   | ~5.70    |
| TSP     | 100  | ~7.76   | ~7.78    |
| TSP     | 500  | ~16.55  | ~16.60   |
| TSP     | 1000 | ~23.12  | ~23.25   |

## 📚 Documentation

### Main Documents
- **QUICKSTART_ML4CO.md** - Quick start guide (start here!)
- **DATA_GENERATION_GUIDE.md** - Data generation usage
- **MODIFICATIONS_SUMMARY.md** - Complete modification list

### Examples
- **examples/example_data_loading.py** - Data loading examples

### Original Projects
- **ML4CO-Bench-101:** https://github.com/Thinklab-SJTU/ML4CO-Bench-101
- **ML4CO-Kit:** https://github.com/Thinklab-SJTU/ML4CO-Kit
- **UniteFormer:** Original repository

## 🎯 Use Cases

### 1. Quick Experimentation
```bash
# Train with data generation - no setup needed!
python train_tsp_ml4co.py
```

### 2. Comparing with ML4CO-Bench-101
```python
# Use same data generation as ML4CO-Bench-101
GENERATOR_CONFIG = {
    'distribution_type': TSP_TYPE.UNIFORM,
    'nodes_num': 50,
}
```

### 3. Testing on Fixed Datasets
```python
# Use ML4CO-Kit test datasets
USE_DATASET_FILE = True
DATASET_PATH = "../../ML4CO-Kit/test_dataset/routing/tsp/wrapper/tsp50_uniform_16ins.txt"
```

### 4. Distribution Ablation Study
```python
# Test different distributions
'distribution_type': TSP_TYPE.UNIFORM  # vs GAUSSIAN, CLUSTER
```

## 🔄 Migration Guide

### From Original UniteFormer

**Before:**
```python
env_params = {
    'problem_size': 50,
    'mode': 'train',
}
```

**After (with data generation):**
```python
env_params = {
    'problem_size': 50,
    'mode': 'train',
    'use_data_generator': True,  # Just add this!
    'generator_config': {...},
}
```

**No other changes needed!**

### From ML4CO-Bench-101 Training

UniteFormer now uses the same data generation approach:

```python
# ML4CO-Bench-101
from ml4co_kit import TSPDataGenerator
generator = TSPDataGenerator(nodes_num=50)
instances = generator.generate_only_instance_for_us(batch_size)

# UniteFormer (now identical!)
from ml4co_kit import TSPDataGenerator
generator = TSPDataGenerator(nodes_num=50)
instances = generator.generate_only_instance_for_us(batch_size)
```

## 📝 Citation

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

## 🤝 Contributing

Future enhancements:
- Support for more CO problems (MIS, MVC, MCut, etc.)
- Integration with ML4CO-Bench-101 training framework
- Additional data distributions
- Multi-GPU data generation

## 📧 Contact

For questions or issues:
- **ML4CO-Bench-101:** https://github.com/Thinklab-SJTU/ML4CO-Bench-101
- **ML4CO-Kit:** https://github.com/Thinklab-SJTU/ML4CO-Kit
- **UniteFormer:** Check original repository

## 📄 License

This integration maintains the licenses of both original projects:
- ML4CO-Bench-101 and ML4CO-Kit: See original repository licenses
- UniteFormer: See original repository license

---

## 🎉 Summary

This integration provides:

✅ **Easy data generation** - No dataset downloads needed
✅ **ML4CO compatibility** - Same tools as ML4CO-Bench-101
✅ **Backward compatible** - Original code still works
✅ **Well documented** - Comprehensive guides and examples
✅ **Flexible** - Switch between data sources easily

**Get started now:**
```bash
cd UniteFormer/UF-TSP
python train_tsp_ml4co.py
```

**No datasets required!** 🚀
