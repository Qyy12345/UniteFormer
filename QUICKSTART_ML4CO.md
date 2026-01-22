# Quick Start Guide: UniteFormer with ML4CO Datasets

This guide will help you quickly set up and run UniteFormer with ML4CO datasets.

## Prerequisites

- Python >= 3.8
- PyTorch >= 2.0.1
- CUDA-enabled GPU (recommended)

## Step 1: Install Dependencies

```bash
# Install ML4CO-Kit
pip install ml4co-kit==0.3.3

# Install other dependencies (if not already installed)
pip install torch>=2.0.1 numpy==1.24.4 matplotlib==3.5.2 tqdm==4.67.1
```

## Step 2: Prepare Datasets

### Option A: Use Existing ML4CO-Bench-101 Datasets

The ML4CO-Bench-101 folder already contains test datasets:

```bash
# TSP datasets are available at:
ls ../../ML4CO-Bench-101/test_dataset/tsp/

# Available files:
# - tsp20_concorde_3.839.txt
# - tsp50_concorde_5.688.txt
# - tsp100_concorde_7.756.txt
# - tsp500_concorde_16.546.txt
# - tsp1000_concorde_23.118.txt
```

### Option B: Download from Hugging Face

```bash
# For more datasets, visit:
# https://huggingface.co/datasets/ML4CO/ML4CO-Bench-101-SL

# Download training datasets
# Download testing datasets
# Download cross datasets
```

## Step 3: Quick Test Run

### Test TSP with ML4CO Dataset

```bash
cd UniteFormer/UF-TSP

# Edit the test script to use ML4CO dataset
nano test_tsp_ml4co.py
```

Update these lines in the file:

```python
# Line ~33: Set problem size
TSP_SIZE = 50

# Line ~38: Set ML4CO dataset path (relative to UF-TSP folder)
ML4CO_TEST_DATASET = "../../ML4CO-Bench-101/test_dataset/tsp/tsp50_concorde_5.688.txt"

# Line ~37: Set USE_ML4CO to True
USE_ML4CO = True

# Line ~46: Set model path (use pretrained or train your own)
MODEL_PATH = "./train_models/tsp50/epoch-1010.pkl"
```

Run the test:

```bash
python test_tsp_ml4co.py
```

**Note:** If you don't have a pretrained model, you need to train first (Step 4).

## Step 4: Train UniteFormer with ML4CO Data

### Train TSP Model

```bash
cd UniteFormer/UF-TSP

# Edit training script
nano train_tsp_ml4co.py
```

Configure:

```python
# Line ~28: Set problem size
TSP_SIZE = 50

# Line ~35: Enable ML4CO
USE_ML4CO = True

# Line ~36: Set dataset path
ML4CO_DATASET_PATH = "../../ML4CO-Bench-101/test_dataset/tsp/tsp50_concorde_5.688.txt"

# Line ~52: Adjust batch size based on GPU memory
'train_batch_size': 128,  # Reduce if you get CUDA out of memory
```

Start training:

```bash
python train_tsp_ml4co.py
```

Training will take several hours. Models are saved periodically to `train_models/tsp_ml4co/`.

## Step 5: Test Your Trained Model

Once training is complete:

```bash
cd UniteFormer/UF-TSP

# Update model path in test script
MODEL_PATH = "./train_models/tsp_ml4co/epoch-1010.pkl"

# Run test
python test_tsp_ml4co.py
```

## Step 6: CVRP (Optional)

For CVRP problems, follow similar steps:

```bash
cd UniteFormer/UF-CVRP

# Edit train_cvrp_ml4co.py or test_cvrp_ml4co.py
# Configure dataset path and parameters
# Run training or testing
```

## Configuration Quick Reference

### Minimal Working Configuration

**For testing TSP50:**

```python
TSP_SIZE = 50
USE_ML4CO = True
ML4CO_TEST_DATASET = "../../ML4CO-Bench-101/test_dataset/tsp/tsp50_concorde_5.688.txt"
MODEL_PATH = "./train_models/tsp50/epoch-1010.pkl"
```

**For training TSP50 (quick test):**

```python
TSP_SIZE = 50
USE_ML4CO = True
ML4CO_DATASET_PATH = "../../ML4CO-Bench-101/test_dataset/tsp/tsp50_concorde_5.688.txt"
'train_batch_size': 64
'epochs': 100  # Reduced for quick testing
'train_episodes': 1000
```

## Common Issues and Solutions

### Issue 1: ModuleNotFoundError: ml4co_kit

```bash
# Solution:
pip install ml4co-kit==0.3.3
```

### Issue 2: CUDA out of memory

```python
# Reduce batch size in training/testing scripts:
'train_batch_size': 64  # or 32, 16, etc.
'test_batch_size': 1
```

### Issue 3: File not found

```bash
# Check dataset path is correct
# Use relative path from UF-TSP or UF-CVRP folder
# Example: "../../ML4CO-Bench-101/test_dataset/tsp/tsp50_concorde_5.688.txt"
```

### Issue 4: No pretrained model available

```bash
# Train your own model first using train_tsp_ml4co.py
# Or use original UniteFormer pretrained models (may work with ML4CO data)
```

## Next Steps

1. **Experiment with different problem sizes:**
   - TSP: 20, 50, 100, 500, 1000
   - CVRP: 20, 50, 100

2. **Try different datasets:**
   - ML4CO-Bench-101 provides multiple distributions
   - Compare performance across datasets

3. **Hyperparameter tuning:**
   - Learning rate
   - Batch size
   - Model architecture

4. **Compare with baselines:**
   - Use ML4CO-Bench-101 evaluation protocols
   - Compare with other methods in the benchmark

## Example Workflow

```bash
# 1. Install dependencies
pip install ml4co-kit==0.3.3

# 2. Navigate to TSP folder
cd UniteFormer/UF-TSP

# 3. Quick test run (if you have pretrained model)
python test_tsp_ml4co.py

# 4. Train model with ML4CO data
python train_tsp_ml4co.py

# 5. Test trained model
python test_tsp_ml4co.py

# 6. Check results in log files
ls -l result_folder/
```

## Performance Expectations

Based on ML4CO-Bench-101:

| Problem | Size | Optimal | Typical ML Solution | UniteFormer Target |
|---------|------|---------|---------------------|-------------------|
| TSP     | 20   | ~3.84   | ~3.85-3.90          | ~3.85             |
| TSP     | 50   | ~5.69   | ~5.70-5.75          | ~5.70             |
| TSP     | 100  | ~7.76   | ~7.77-7.85          | ~7.78             |
| TSP     | 500  | ~16.55  | ~16.60-16.80        | ~16.60            |
| TSP     | 1000 | ~23.12  | ~23.20-23.50        | ~23.25            |

*Values are average tour lengths. Lower is better.*

## Additional Resources

- **Full Documentation:** See `README_ML4CO_INTEGRATION.md`
- **ML4CO-Bench-101:** https://github.com/Thinklab-SJTU/ML4CO-Bench-101
- **ML4CO-Kit:** https://github.com/Thinklab-SJTU/ML4CO-Kit
- **Original UniteFormer:** Check original repository

## Support

For issues specific to:
- **ML4CO datasets:** Check ML4CO-Bench-101 repository
- **Data loading:** Check ML4CO-Kit documentation
- **UniteFormer model:** Check original UniteFormer repository

---

**Ready to start? Run this:**

```bash
cd UniteFormer/UF-TSP
python test_tsp_ml4co.py
```

Good luck with your experiments!
