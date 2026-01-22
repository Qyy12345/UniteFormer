##########################################################################################
# Training UniteFormer on ML4CO CVRP Dataset
#
# This script demonstrates how to train UniteFormer using ML4CO CVRP datasets
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
from utils import create_logger, copy_all_src
from CVRPTrainer import CVRPTrainer as Trainer

# Use ML4CO-enabled environment
from CVRPEnv_ML4CO import CVRPEnvML4CO

##########################################################################################
# Configuration for ML4CO CVRP Dataset Training

# Problem Configuration
CVRP_SIZE = 50  # Options: 20, 50, 100, etc.
NUM_NEIGHBORS = -1  # -1 for dense

# Dataset Configuration
USE_ML4CO = True  # Set to True to use ML4CO datasets
# ML4CO CVRP datasets are typically in .pkl format or custom format
ML4CO_DATASET_PATH = "data/cvrp50_instances.pkl"
# Alternative: Use text format if available
# ML4CO_DATASET_PATH = "../ML4CO-Bench-101/test_dataset/cvrp/cvrp50.txt"

# Training Parameters
env_params = {
    'mode': 'train',  # 'train' or 'test'
    'problem_size': CVRP_SIZE,
    'pomo_size': CVRP_SIZE,
    'num_neighbors': NUM_NEIGHBORS,
    'data_path': ML4CO_DATASET_PATH if USE_ML4CO else None,
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

optimizer_params = {
    'optimizer': {
        'lr': 4 * 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [801, 1001],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 1010,
    'train_episodes': 1000 * 100,  # Total training episodes
    'train_batch_size': 256,  # Adjust based on your GPU memory
    'logging': {
        'model_save_interval': 20,
        'img_save_interval': 100,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_cvrp_ml4co.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': False,  # Set to True to resume training
        'path': './train_models/cvrp_ml4co',
        'epoch': 1010,
    }
}

logger_params = {
    'log_file': {
        'desc': f'cvrp{CVRP_SIZE}_ml4co_train',
        'filename': 'run_log'
    }
}

##########################################################################################
# Custom Trainer with ML4CO Support
class CVRPTrainerML4CO(Trainer):
    """
    Custom trainer that uses CVRPEnvML4CO instead of CVRPEnv.
    """
    def __init__(self, env_params, model_params, optimizer_params, trainer_params):
        # Use ML4CO-enabled environment
        from CVRPModel import CVRPModel
        from CVRPEnv_ML4CO import CVRPEnvML4CO

        # Initialize environment with ML4CO support
        self.env = CVRPEnvML4CO(**env_params)

        # Initialize model
        self.model = CVRPModel(**model_params)

        # Rest of initialization follows parent class
        Trainer.__init__(self, env_params, model_params, optimizer_params, trainer_params)


##########################################################################################
# main
def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    # Use custom trainer with ML4CO support
    trainer = CVRPTrainerML4CO(
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params
    )

    copy_all_src(trainer.result_folder)

    print("\n" + "="*80)
    print(f"Training UniteFormer on ML4CO CVRP{cvrp_size} Dataset")
    print("="*80)
    print(f"Problem Size: {CVRP_SIZE}")
    print(f"Dataset Path: {ML4CO_DATASET_PATH}")
    print(f"Batch Size: {trainer_params['train_batch_size']}")
    print(f"Epochs: {trainer_params['epochs']}")
    print("="*80 + "\n")

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 4


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    logger.info('USE_ML4CO: {}'.format(USE_ML4CO))
    if USE_ML4CO:
        logger.info('ML4CO_DATASET_PATH: {}'.format(ML4CO_DATASET_PATH))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
