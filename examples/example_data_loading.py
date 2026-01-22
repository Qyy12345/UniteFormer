#!/usr/bin/env python3
"""
Example: Using ML4CO Data Loader for UniteFormer

This script demonstrates how to use the ML4CODataLoader to load datasets
without running the full training/testing pipeline.
"""

import sys
import os

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

from ml4co_data_loader import ML4CODataLoader, save_predictions_to_ml4co_format
import torch


def example_load_tsp():
    """Example: Load TSP dataset"""
    print("="*80)
    print("Example 1: Loading TSP Dataset")
    print("="*80)

    # Initialize loader
    loader = ML4CODataLoader(problem_type='tsp')

    # Dataset path
    dataset_path = "../../ML4CO-Bench-101/test_dataset/tsp/tsp50_concorde_5.688.txt"

    print(f"\nLoading TSP dataset from: {dataset_path}")

    try:
        # Load data
        nodes, solutions = loader.load_tsp_data(dataset_path)

        print(f"✓ Successfully loaded {nodes.shape[0]} samples")
        print(f"  - Nodes shape: {nodes.shape}")
        print(f"  - Solutions shape: {solutions.shape if solutions is not None else 'None'}")

        # Show first sample
        print(f"\nFirst sample:")
        print(f"  - Number of nodes: {nodes[0].shape[0]}")
        print(f"  - First 5 node coordinates:\n{nodes[0][:5]}")
        if solutions is not None:
            print(f"  - Solution tour (first 10): {solutions[0][:10].tolist()}...")

    except FileNotFoundError:
        print(f"✗ Dataset not found at: {dataset_path}")
        print("  Please download ML4CO datasets or update the path")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")


def example_load_cvrp():
    """Example: Load CVRP dataset"""
    print("\n" + "="*80)
    print("Example 2: Loading CVRP Dataset")
    print("="*80)

    # Initialize loader
    loader = ML4CODataLoader(problem_type='cvrp')

    # Note: CVRP datasets in ML4CO may be in different formats
    # This is a placeholder example
    print("\nNote: CVRP dataset loading example")
    print("To load CVRP data:")
    print("  1. Ensure you have CVRP dataset in ML4CO format")
    print("  2. Update the dataset path")
    print("  3. Run: depot, node_xy, demand, capacity, solutions = loader.load_cvrp_data(path)")


def example_convert_to_uniteformer():
    """Example: Convert ML4CO data to UniteFormer format"""
    print("\n" + "="*80)
    print("Example 3: Converting to UniteFormer Raw Format")
    print("="*80)

    # Initialize loader
    loader = ML4CODataLoader(problem_type='tsp')

    dataset_path = "../../ML4CO-Bench-101/test_dataset/tsp/tsp50_concorde_5.688.txt"

    print(f"\nConverting dataset: {dataset_path}")

    try:
        # Load and convert
        raw_nodes, raw_tours = loader.load_raw_tsp_for_uniteformer(dataset_path, num_samples=10)

        print(f"✓ Converted {raw_nodes.shape[0]} samples")
        print(f"  - Raw nodes shape: {raw_nodes.shape}")
        print(f"  - Raw tours shape: {raw_tours.shape if raw_tours is not None else 'None'}")

        print("\nThis data is now ready for use in TSPEnvML4CO:")
        print("  env.load_raw_data(episode=len(raw_nodes))")

    except FileNotFoundError:
        print(f"✗ Dataset not found at: {dataset_path}")
    except Exception as e:
        print(f"✗ Error: {e}")


def example_save_predictions():
    """Example: Save predictions in ML4CO format"""
    print("\n" + "="*80)
    print("Example 4: Saving Predictions in ML4CO Format")
    print("="*80)

    # Create dummy predictions
    num_samples = 3
    num_nodes = 5

    predictions = [
        torch.tensor([0, 1, 2, 3, 4]),
        torch.tensor([1, 2, 3, 4, 0]),
        torch.tensor([2, 3, 4, 0, 1]),
    ]

    nodes_list = [
        torch.rand(num_nodes, 2),
        torch.rand(num_nodes, 2),
        torch.rand(num_nodes, 2),
    ]

    output_path = "example_predictions.txt"

    print(f"\nSaving {num_samples} predictions to: {output_path}")
    save_predictions_to_ml4co_format(predictions, nodes_list, output_path)

    print("✓ Predictions saved successfully!")

    # Show first line
    with open(output_path, 'r') as f:
        first_line = f.readline()
        print(f"\nFirst line of saved file:")
        print(f"  {first_line[:100]}...")


def example_usage_in_training():
    """Example: How to use in training script"""
    print("\n" + "="*80)
    print("Example 5: Integration in Training Script")
    print("="*80)

    print("""
In your training script, you can use the ML4CO data loader as follows:

```python
from utils.ml4co_data_loader import ML4CODataLoader
from TSPEnv_ML4CO import TSPEnvML4CO

# Initialize environment with ML4CO support
env_params = {
    'problem_size': 50,
    'pomo_size': 50,
    'num_neighbors': -1,
    'data_path': '../../ML4CO-Bench-101/test_dataset/tsp/tsp50_concorde_5.688.txt',
    'mode': 'train',
    'use_ml4co': True,  # Enable ML4CO dataset support
}

env = TSPEnvML4CO(**env_params)

# Load and use data
env.load_raw_data(episode=100)  # Load 100 samples
env.load_problems(episode=0, batch_size=32)  # Load first batch of 32

# Reset environment
reset_state, reward, done = env.reset()

# Training loop...
```
    """)


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("ML4CO Data Loader Examples for UniteFormer")
    print("="*80)

    # Check if ml4co_kit is available
    try:
        import ml4co_kit
        print("✓ ml4co_kit is installed")
    except ImportError:
        print("✗ ml4co_kit is not installed")
        print("  Install with: pip install ml4co-kit==0.3.3")
        print("\nSome examples will not work without ml4co_kit.")
        print("Install the package and run this script again.\n")
        return

    # Run examples
    example_load_tsp()
    example_load_cvrp()
    example_convert_to_uniteformer()
    example_save_predictions()
    example_usage_in_training()

    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Install ML4CO datasets")
    print("  2. Run train_tsp_ml4co.py to train a model")
    print("  3. Run test_tsp_ml4co.py to test the model")
    print("  4. Check README_ML4CO_INTEGRATION.md for detailed documentation")
    print("\n")


if __name__ == "__main__":
    main()
