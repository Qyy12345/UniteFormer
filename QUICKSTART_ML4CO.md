# Quick Start Guide: UniteFormer with ML4CO Data Generation

This guide will help you quickly set up and run UniteFormer with ML4CO data generation capabilities.


## Step 1: Install Dependencies

ML4CO-Kit Dependencies
&
```bash
# Install other dependencies (if not already installed)
pip install numpy==1.24.4 matplotlib==3.5.2 tqdm==4.67.1
```

## Step 2: determine your Data Source

UniteFormer supports **three** ways to get data(usually A for training and B,C for testing):

### Option A: Data Generation (ML4CO generator) 
- ( **Same as ML4CO-Bench-101**)

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

Update these lines(Configuration):

```python
# Line : Set problem size
TSP_SIZE = 50

# Line : Use existing dataset from ML4CO-Kit
USE_ML4CO = True
ML4CO_TEST_DATASET = "../../ML4CO-Kit/test_dataset/routing/tsp/wrapper/tsp50_uniform_16ins.txt"

# Line : Set model path (need pretrained model)
MODEL_PATH = "./train_models/tsp50/epoch-1010.pkl"
```

Run test:
```bash
python test_tsp_ml4co.py
```

**Note:** If you don't have a pretrained model, go to Step 4 to train one.

## Step 4: Train with Data Generation (Recommended)

### TSP Training

```bash
cd UniteFormer/UF-TSP

# Edit test script
nano test_tsp_ml4co.py
```

Update these lines(Configuration):

```python
# Line : Problem size
TSP_SIZE = 50  # Options: 20, 50, 100, 500, 1000

# Line : Enable data generation (already set!)
USE_DATA_GENERATOR = True

# Line : Generator config
GENERATOR_CONFIG = {
    'data_type': 'uniform',  # Uniform distribution (default)
    'nodes_num': TSP_SIZE,
}

# Line : Dataset file not needed
USE_DATASET_FILE = False
```

Start training:
```bash
python train_tsp_ml4co.py
```

### CVRP Training

```bash
cd UniteFormer/UF-CVRP
```

Configuration:

```python
# Line ~42: Problem size
CVRP_SIZE = 50

# Line ~46: Enable data generation
USE_DATA_GENERATOR = True

# Line ~47-52: Generator config
GENERATOR_CONFIG = {
    'distribution_type': CVRP_TYPE.UNIFORM if ML4CO_AVAILABLE else None,
    'nodes_num': CVRP_SIZE,
    'capacity': 40.0,  # Vehicle capacity (will be normalized)
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

### 2. Download ML4CO-Bench-101 Datasets(我的实践中ML4CO-Bench-101和UniteFormer在同级目录，故路径是例如../../ML4CO-Bench-101/test_dataset/tsp/tsp50_concorde_5.688.txt的形式)

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
