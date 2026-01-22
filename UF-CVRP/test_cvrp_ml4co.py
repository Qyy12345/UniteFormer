##########################################################################################
# Testing UniteFormer on ML4CO CVRP Dataset
#
# This script demonstrates how to test UniteFormer using ML4CO CVRP datasets
##########################################################################################

# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

# Path Config
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# import
import logging
from utils import create_logger
from CVRPTester import CVRPTester as Tester

# Use ML4CO-enabled environment
from CVRPEnv_ML4CO import CVRPEnvML4CO

##########################################################################################
# Configuration for ML4CO CVRP Dataset Testing

# Problem Configuration
CVRP_SIZE = 50  # Options: 20, 50, 100, etc.
NUM_NEIGHBORS = -1  # -1 for dense

# Dataset Configuration
USE_ML4CO = True  # Set to True to use ML4CO datasets
ML4CO_TEST_DATASET = "data/cvrp50_test_instances.pkl"
# Alternative: Use text format if available
# ML4CO_TEST_DATASET = "../ML4CO-Bench-101/test_dataset/cvrp/cvrp50.txt"

# Model Configuration
MODEL_PATH = "./train_models/cvrp50/epoch-1010.pkl"
# MODEL_PATH = "./train_models/cvrp_ml4co/epoch-1010.pkl"

# Testing Parameters
env_params = {
    'mode': 'test',  # Test mode
    'problem_size': CVRP_SIZE,
    'pomo_size': CVRP_SIZE,
    'num_neighbors': NUM_NEIGHBORS,
    'data_path': ML4CO_TEST_DATASET,
    'use_ml4co': USE_ML4CO,  # Enable ML4CO dataset support
    'optimal_label': None,  # Optional: provide optimal values if available
}

model_params = {
    'encoder_layer_num': 3,
    'embedding_dim': 256,
    'sqrt_embedding_dim': 256 ** (1 / 2),
    'head_num': 16,
    'qkv_dim': 16,
    'sqrt_qkv_dim': 16 ** (1 / 2),
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
    'GCN_dim': 3,
    'mlp_layers': 3,
    'aggregation': "mean",
    'ms_hidden_dim': 16,
    'ms_layer1_init': (1 / 2) ** (1 / 2),
    'ms_layer2_init': (1 / 16) ** (1 / 2),
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'test_episodes': 100,  # Number of test episodes
    'test_batch_size': 1,  # Batch size for testing
    'augmentation_enable': True,  # Enable 8-fold data augmentation
    'model_load': {
        'path': MODEL_PATH,
        'epoch': 1010,
    }
}

logger_params = {
    'log_file': {
        'desc': f'cvrp{CVRP_SIZE}_ml4co_test',
        'filename': 'run_log'
    }
}

##########################################################################################
# Custom Tester with ML4CO Support
class CVRPTesterML4CO(Tester):
    """
    Custom tester that uses CVRPEnvML4CO instead of CVRPEnv.
    """
    def __init__(self, env_params, model_params, tester_params):
        # Import here to avoid circular imports
        from CVRPModel import CVRPModel
        from CVRPEnv_ML4CO import CVRPEnvML4CO

        # Initialize environment with ML4CO support
        self.env = CVRPEnvML4CO(**env_params)

        # Initialize model
        self.model = CVRPModel(**model_params)

        # Rest of initialization follows parent class
        Tester.__init__(self, env_params, model_params, tester_params)


##########################################################################################
# main
def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    # Use custom tester with ML4CO support
    tester = CVRPTesterML4CO(
        env_params=env_params,
        model_params=model_params,
        tester_params=tester_params
    )

    print("\n" + "="*80)
    print(f"Testing UniteFormer on ML4CO CVRP{cvrp_size} Dataset")
    print("="*80)
    print(f"Problem Size: {CVRP_SIZE}")
    print(f"Dataset Path: {ML4CO_TEST_DATASET}")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Test Episodes: {tester_params['test_episodes']}")
    print(f"Data Augmentation: {tester_params['augmentation_enable']}")
    print("="*80 + "\n")

    tester.run()

    # Print results summary
    print("\n" + "="*80)
    print("Testing Results Summary")
    print("="*80)
    print(f"Average cost: {tester.avg_score:.4f}")
    print(f"Best cost: {tester.best_score:.4f}")
    if tester.avg_gap is not None:
        print(f"Average gap: {tester.avg_gap:.2f}%")
    print("="*80 + "\n")


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 2
    tester_params['test_batch_size'] = 1


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    logger.info('USE_ML4CO: {}'.format(USE_ML4CO))
    if USE_ML4CO:
        logger.info('ML4CO_TEST_DATASET: {}'.format(ML4CO_TEST_DATASET))
    logger.info('MODEL_PATH: {}'.format(MODEL_PATH))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
