# Quick Start Guide: UniteFormer with ML4CO Data Generation

This guide will help you quickly set up and run UniteFormer with ML4CO data generation capabilities.

## Prerequisites

- Python >= 3.8
- PyTorch >= 2.0.1
- CUDA-enabled GPU (recommended)

## Step 1: Install Dependencies

```bash
# Install ML4CO-Kit (required for data generation)
pip install ml4co-kit==0.3.3

# Install other dependencies (if not already installed)
pip install torch>=2.0.1 numpy==1.24.4 matplotlib==3.5.2 tqdm==4.67.1
```

## Step 2: Choose Your Data Source

UniteFormer now supports **three** ways to get training data:

### Option A: Data Generation (Recommended for Training) ⭐
- ✅ **No dataset files needed**
- ✅ **Dynamic data generation**
- ✅ **Multiple distributions available**
- ✅ **Same as ML4CO-Bench-101**

### Option B: Existing Dataset Files
- Use ML4CO-Kit test datasets
- Use ML4CO-Bench-101 datasets (need to download)

### Option C: Original Random Generation
- UniteFormer's original method
- Good for quick testing

## Step 3: Quick Test Run (Recommended for Testing)

If you have pretrained models, you can test immediately:

```bash
cd UniteFormer/UF-TSP

# Edit test script
nano test_tsp_ml4co.py
```

Update these lines:

```python
# Line ~33: Set problem size
TSP_SIZE = 50

# Line ~40: Use existing dataset from ML4CO-Kit
USE_ML4CO = True
ML4CO_TEST_DATASET = "../../ML4CO-Kit/test_dataset/routing/tsp/wrapper/tsp50_uniform_16ins.txt"

# Line ~46: Set model path (need pretrained model)
MODEL_PATH = "./train_models/tsp50/epoch-1010.pkl"
```

Run test:
```bash
python test_tsp_ml4co.py
```

**Note:** If you don't have a pretrained model, go to Step 4 to train one.

## Step 4: Train with Data Generation (Recommended)

This is the **easiest way** to train UniteFormer!

### TSP Training

```bash
cd UniteFormer/UF-TSP

# The script is already configured to use data generation
# Just verify these settings in train_tsp_ml4co.py:
```

Configuration in `train_tsp_ml4co.py`:

```python
# Line ~42: Problem size
TSP_SIZE = 50  # Options: 20, 50, 100, 500, 1000

# Line ~46: Enable data generation (already set!)
USE_DATA_GENERATOR = True

# Line ~47-51: Generator config
GENERATOR_CONFIG = {
    'distribution_type': TSP_TYPE.UNIFORM,  # UNIFORM, GAUSSIAN, CLUSTER
    'nodes_num': TSP_SIZE,
    'precision': 'float32',
}

# Line ~55: Dataset file not needed
USE_DATASET_FILE = False
```

Start training:
```bash
python train_tsp_ml4co.py
```

### CVRP Training

```bash
cd UniteFormer/UF-CVRP

# Verify settings in train_cvrp_ml4co.py
```

Configuration:

```python
# Line ~42: Problem size
CVRP_SIZE = 50

# Line ~46: Enable data generation
USE_DATA_GENERATOR = True

# Line ~47-52: Generator config
GENERATOR_CONFIG = {
    'distribution_type': CVRP_TYPE.UNIFORM,
    'nodes_num': CVRP_SIZE,
    'capacity': 40.0,  # Vehicle capacity
    'precision': 'float32',
}
```

Start training:
```bash
python train_cvrp_ml4co.py
```

## Step 5: Train with Dataset Files (Alternative)

If you prefer to use dataset files:

### 1. Use ML4CO-Kit Test Datasets

```python
# In train_tsp_ml4co.py:
USE_DATA_GENERATOR = False
USE_DATASET_FILE = True
DATASET_PATH = "../../ML4CO-Kit/test_dataset/routing/tsp/wrapper/tsp50_uniform_16ins.txt"
```

### 2. Download ML4CO-Bench-101 Datasets

Visit: https://huggingface.co/datasets/ML4CO/ML4CO-Bench-101-SL

Download test datasets and place in:
```
ML4CO-Bench-101/test_dataset/tsp/
```

Then update path:
```python
DATASET_PATH = "../../ML4CO-Bench-101/test_dataset/tsp/tsp50_concorde_5.688.txt"
```

## Step 6: Test Your Trained Model

Once training is complete:

```bash
cd UniteFormer/UF-TSP

# Update model path in test script
MODEL_PATH = "./train_models/tsp_ml4co/epoch-1010.pkl"

# Run test
python test_tsp_ml4co.py
```

## Configuration Quick Reference

### Minimal Working Configuration

**For training TSP50 with data generation (Recommended):**

```python
TSP_SIZE = 50
USE_DATA_GENERATOR = True
GENERATOR_CONFIG = {
    'distribution_type': TSP_TYPE.UNIFORM,
    'nodes_num': TSP_SIZE,
}
train_batch_size = 128  # Adjust based on GPU memory
```

**For training TSP50 with dataset file:**

```python
TSP_SIZE = 50
USE_DATA_GENERATOR = False
USE_DATASET_FILE = True
DATASET_PATH = "../../ML4CO-Kit/test_dataset/routing/tsp/wrapper/tsp50_uniform_16ins.txt"
train_batch_size = 128
```

**For quick testing (few epochs):**

```python
TSP_SIZE = 50
USE_DATA_GENERATOR = True
train_batch_size = 64
epochs = 100  # Reduced for quick testing
train_episodes = 1000
```

## Data Distribution Options

### TSP
- `TSP_TYPE.UNIFORM`: Random uniform distribution (default)
- `TSP_TYPE.GAUSSIAN`: Gaussian/Normal distribution
- `TSP_TYPE.CLUSTER`: Clustered distribution

### CVRP
- `CVRP_TYPE.UNIFORM`: Random uniform distribution (default)
- `CVRP_TYPE.GAUSSIAN`: Gaussian distribution

## Advantages of Data Generation

| Feature | Data Generation | Dataset Files |
|---------|----------------|---------------|
| **Setup** | ✅ No download needed | ❌ Need to download datasets |
| **Storage** | ✅ Minimal storage | ❌ Large files required |
| **Flexibility** | ✅ Change distributions easily | ❌ Fixed distribution |
| **Consistency** | ✅ Same as ML4CO-Bench-101 | ✅ If using ML4CO datasets |
| **Reproducibility** | ✅ Set random seed | ✅ Deterministic files |

## Common Issues and Solutions

### Issue 1: ModuleNotFoundError: ml4co_kit

```bash
# Solution:
pip install ml4co-kit==0.3.3
```

### Issue 2: CUDA out of memory

```python
# Reduce batch size in training scripts:
'train_batch_size': 64  # or 32, 16, etc.
```

### Issue 3: Data generation fails

**Solution:** Script automatically falls back to random generation. Check logs for:
```
Warning: Data generation failed (...), falling back to random generation
```

### Issue 4: File not found

```bash
# Check dataset path is correct
# Use relative path from UF-TSP or UF-CVRP folder
# Example: "../../ML4CO-Kit/test_dataset/routing/tsp/wrapper/tsp50_uniform_16ins.txt"
```

### Issue 5: No pretrained model available

```bash
# Train your own model first using train_tsp_ml4co.py
# With data generation, this is very easy!
python train_tsp_ml4co.py
```

## Next Steps

1. **Experiment with different distributions:**
   ```python
   # Try GAUSSIAN instead of UNIFORM
   'distribution_type': TSP_TYPE.GAUSSIAN
   ```

2. **Try different problem sizes:**
   - TSP: 20, 50, 100, 500, 1000
   - CVRP: 20, 50, 100

3. **Compare with baselines:**
   - Use same data generation as ML4CO-Bench-101
   - Compare performance metrics

4. **Hyperparameter tuning:**
   - Learning rate
   - Batch size
   - Model architecture

## Example Workflow

```bash
# 1. Install dependencies
pip install ml4co-kit==0.3.3

# 2. Navigate to TSP folder
cd UniteFormer/UF-TSP

# 3. Train with data generation (easiest!)
python train_tsp_ml4co.py

# 4. Test trained model
python test_tsp_ml4co.py

# 5. Check results
ls -l result_folder/
```

## Performance Expectations

Based on ML4CO-Bench-101 with **UNIFORM distribution**:

| Problem | Size | Optimal | UniteFormer Target |
|---------|------|---------|-------------------|
| TSP     | 20   | ~3.84   | ~3.85             |
| TSP     | 50   | ~5.69   | ~5.70             |
| TSP     | 100  | ~7.76   | ~7.78             |
| TSP     | 500  | ~16.55  | ~16.60            |
| TSP     | 1000 | ~23.12  | ~23.25            |

*Values are average tour lengths. Lower is better.*

**Tip:** Use data generation to match ML4CO-Bench-101 training setup!

## Advanced: Mixing Data Sources

You can even mix data sources during training:

```python
# Epoch 1-500: Use UNIFORM distribution
GENERATOR_CONFIG = {'distribution_type': TSP_TYPE.UNIFORM, ...}

# Epoch 501-1000: Switch to GAUSSIAN
GENERATOR_CONFIG = {'distribution_type': TSP_TYPE.GAUSSIAN, ...}
```

## Additional Resources

- **Full Documentation:** See `README_ML4CO_INTEGRATION.md`
- **Data Generation Guide:** See `DATA_GENERATION_GUIDE.md`
- **ML4CO-Bench-101:** https://github.com/Thinklab-SJTU/ML4CO-Bench-101
- **ML4CO-Kit:** https://github.com/Thinklab-SJTU/ML4CO-Kit

## Support

For issues specific to:
- **Data generation:** Check ML4CO-Kit documentation
- **UniteFormer model:** Check original repository
- **This integration:** Check `MODIFICATIONS_SUMMARY.md`

---

**Ready to start? Run this:**

```bash
cd UniteFormer/UF-TSP
python train_tsp_ml4co.py
```

**No datasets needed!** 🚀

Good luck with your experiments!
