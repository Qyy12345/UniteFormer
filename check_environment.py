#!/usr/bin/env python3
"""
Environment Check Script for UniteFormer with ML4CO Integration

This script checks if all dependencies are properly installed
and if the integration is ready to use.
"""

import sys
import os

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (need >= 3.8)")
        return False


def check_torch():
    """Check PyTorch installation"""
    print("\nChecking PyTorch...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.version.cuda}")
            print(f"  ✓ GPU count: {torch.cuda.device_count()}")
        else:
            print("  ⚠ CUDA not available (CPU only)")
        return True
    except ImportError:
        print("  ✗ PyTorch not installed")
        print("    Install with: pip install torch>=2.0.1")
        return False


def check_ml4co_kit():
    """Check ML4CO-Kit installation"""
    print("\nChecking ML4CO-Kit...")
    try:
        import ml4co_kit
        print(f"  ✓ ml4co_kit installed")
        return True
    except ImportError:
        print("  ✗ ml4co_kit not installed")
        print("    Install with: pip install ml4co-kit==0.3.3")
        return False


def check_numpy():
    """Check NumPy installation"""
    print("\nChecking NumPy...")
    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
        return True
    except ImportError:
        print("  ✗ NumPy not installed")
        return False


def check_original_files():
    """Check if original UniteFormer files exist"""
    print("\nChecking UniteFormer files...")

    required_files = [
        "UF-TSP/TSPEnv.py",
        "UF-TSP/TSPModel.py",
        "UF-TSP/TSPTrainer.py",
        "UF-TSP/TSPTester.py",
        "UF-CVRP/CVRPEnv.py",
        "UF-CVRP/CVRPModel.py",
        "UF-CVRP/CVRPTrainer.py",
        "UF-CVRP/CVRPTester.py",
    ]

    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (not found)")
            all_exist = False

    return all_exist


def check_ml4co_files():
    """Check if ML4CO integration files exist"""
    print("\nChecking ML4CO integration files...")

    integration_files = [
        "utils/ml4co_data_loader.py",
        "UF-TSP/TSPEnv_ML4CO.py",
        "UF-TSP/train_tsp_ml4co.py",
        "UF-TSP/test_tsp_ml4co.py",
        "UF-CVRP/CVRPEnv_ML4CO.py",
        "UF-CVRP/train_cvrp_ml4co.py",
        "UF-CVRP/test_cvrp_ml4co.py",
    ]

    all_exist = True
    for file in integration_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (not found)")
            all_exist = False

    return all_exist


def check_documentation():
    """Check if documentation files exist"""
    print("\nChecking documentation...")

    doc_files = [
        "README_ML4CO_INTEGRATION.md",
        "QUICKSTART_ML4CO.md",
        "MODIFICATIONS_SUMMARY.md",
    ]

    all_exist = True
    for file in doc_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (not found)")
            all_exist = False

    return all_exist


def check_datasets():
    """Check if ML4CO datasets are available"""
    print("\nChecking ML4CO datasets...")

    dataset_paths = [
        ("../ML4CO-Bench-101/test_dataset/tsp/tsp20_concorde_3.839.txt", "TSP-20"),
        ("../ML4CO-Bench-101/test_dataset/tsp/tsp50_concorde_5.688.txt", "TSP-50"),
        ("../ML4CO-Bench-101/test_dataset/tsp/tsp100_concorde_7.756.txt", "TSP-100"),
    ]

    found_any = False
    for path, name in dataset_paths:
        if os.path.exists(path):
            print(f"  ✓ {name}: {path}")
            found_any = True
        else:
            print(f"  ✗ {name}: {path} (not found)")

    if not found_any:
        print("\n  ⚠ No ML4CO datasets found")
        print("    Download from: https://huggingface.co/datasets/ML4CO/ML4CO-Bench-101-SL")

    return found_any


def check_pretrained_models():
    """Check if pretrained models exist"""
    print("\nChecking pretrained models...")

    model_paths = [
        ("UF-TSP/train_models/tsp20/epoch-1010.pkl", "TSP-20"),
        ("UF-TSP/train_models/tsp50/epoch-1010.pkl", "TSP-50"),
        ("UF-TSP/train_models/tsp100/epoch-1010.pkl", "TSP-100"),
    ]

    found_any = False
    for path, name in model_paths:
        if os.path.exists(path):
            print(f"  ✓ {name}: {path}")
            found_any = True
        else:
            print(f"  ✗ {name}: {path} (not found)")

    if not found_any:
        print("\n  ⚠ No pretrained models found")
        print("    Train models using: python train_tsp_ml4co.py")

    return found_any


def test_imports():
    """Test if integration can be imported"""
    print("\nTesting imports...")

    try:
        # Test ML4CO data loader import
        sys.path.insert(0, "utils")
        from ml4co_data_loader import ML4CODataLoader
        print("  ✓ ml4co_data_loader import successful")

        # Test TSP environment import
        sys.path.insert(0, "UF-TSP")
        from TSPEnv_ML4CO import TSPEnvML4CO
        print("  ✓ TSPEnv_ML4CO import successful")

        # Test CVRP environment import
        sys.path.insert(0, "UF-CVRP")
        from CVRPEnv_ML4CO import CVRPEnvML4CO
        print("  ✓ CVRPEnv_ML4CO import successful")

        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def main():
    """Run all checks"""
    print("="*80)
    print("UniteFormer + ML4CO Integration Environment Check")
    print("="*80)

    results = {
        "Python": check_python_version(),
        "PyTorch": check_torch(),
        "ML4CO-Kit": check_ml4co_kit(),
        "NumPy": check_numpy(),
        "Original Files": check_original_files(),
        "Integration Files": check_ml4co_files(),
        "Documentation": check_documentation(),
        "Datasets": check_datasets(),
        "Models": check_pretrained_models(),
        "Imports": test_imports(),
    }

    print("\n" + "="*80)
    print("Check Summary")
    print("="*80)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print("\n" + "="*80)
    print(f"Results: {passed}/{total} checks passed")
    print("="*80)

    if passed == total:
        print("\n✓ All checks passed! You're ready to go.")
        print("\nNext steps:")
        print("  1. Read QUICKSTART_ML4CO.md for quick start guide")
        print("  2. Run: python UF-TSP/train_tsp_ml4co.py")
        print("  3. Or run: python UF-TSP/test_tsp_ml4co.py (if you have models)")
    else:
        print("\n⚠ Some checks failed. Please install missing dependencies.")
        print("\nTo fix:")
        print("  1. Install ML4CO-Kit: pip install ml4co-kit==0.3.3")
        print("  2. Download datasets from Hugging Face")
        print("  3. Train models or download pretrained ones")

    print()


if __name__ == "__main__":
    main()
