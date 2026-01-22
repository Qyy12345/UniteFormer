##########################################################################################
# Training UniteFormer on ML4CO CVRP Dataset (with Data Generation)
#
# This script uses ML4CO-Kit's CVRPDataGenerator to dynamically generate training data
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
sys.path.insert(0, "../../ML4CO-Kit")  # for ml4co_kit

##########################################################################################
# import
import logging
from utils import create_logger, copy_all_src
from CVRPTrainer import CVRPTrainer as Trainer

# Import ML4CO-Kit data generator
try:
    from ml4co_kit import CVRPDataGenerator, CVRP_TYPE
    ML4CO_AVAILABLE = True
except ImportError:
    print("Warning: ml4co_kit not available. Install with: pip install ml4co-kit")
    ML4CO_AVAILABLE = False

# Use ML4CO-enabled environment
from CVRPEnv_ML4CO import CVRPEnvML4CO

##########################################################################################
# Configuration for ML4CO CVRP Dataset Training

# Problem Configuration
CVRP_SIZE = 50  # Options: 20, 50, 100, etc.
NUM_NEIGHBORS = -1  # -1 for dense

# Data Generation Configuration
USE_DATA_GENERATOR = True  # Use ML4CO-Kit's CVRPDataGenerator
GENERATOR_CONFIG = {
    'distribution_type': CVRP_TYPE.UNIFORM if ML4CO_AVAILABLE else None,
    'nodes_num': CVRP_SIZE,
    'capacity': 40.0,  # Vehicle capacity (will be normalized)
    'precision': 'float32',
}

# Alternative: Use existing dataset file
USE_DATASET_FILE = False
DATASET_PATH = "data/cvrp50_instances.pkl"

# Training Parameters
env_params = {
    'mode': 'train',  # 'train' or 'test'
    'problem_size': CVRP_SIZE,
    'pomo_size': CVRP_SIZE,
    'num_neighbors': NUM_NEIGHBORS,
    'data_path': DATASET_PATH if USE_DATASET_FILE else None,
    'use_ml4co': USE_DATASET_FILE,  # Only True if using dataset file
    'use_data_generator': USE_DATA_GENERATOR,  # New parameter!
    'generator_config': GENERATOR_CONFIG if USE_DATA_GENERATOR else None,
    'optimal_label': None,
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
        'desc': f'cvrp{CVRP_SIZE}_ml4co_gen_train',
        'filename': 'run_log'
    }
}

##########################################################################################
# Custom Trainer with ML4CO Support
class CVRPTrainerML4CO(Trainer):
    """
    Custom trainer that uses CVRPEnvML4CO with data generation support.
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
# Data Generator Helper (if using ML4CO-Kit)
def create_data_generator(config):
    """
    Create a CVRPDataGenerator instance with the given configuration.

    Args:
        config: Dictionary with generator configuration

    Returns:
        CVRPDataGenerator instance or None
    """
    if not ML4CO_AVAILABLE:
        print("ML4CO-Kit not available, cannot use data generator")
        return None

    try:
        generator = CVRPDataGenerator(
            distribution_type=config.get('distribution_type', CVRP_TYPE.UNIFORM),
            nodes_num=config.get('nodes_num', 50),
            capacity=config.get('capacity', 40.0),
            precision=config.get('precision', 'float32'),
        )
        print(f"✓ Created CVRPDataGenerator with {config['nodes_num']} nodes")
        return generator
    except Exception as e:
        print(f"✗ Failed to create CVRPDataGenerator: {e}")
        return None


##########################################################################################
# main
def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    # Check ML4CO availability if using data generator
    if USE_DATA_GENERATOR and not ML4CO_AVAILABLE:
        print("\n" + "="*80)
        print("ERROR: ML4CO-Kit is required for data generation")
        print("="*80)
        print("Please install: pip install ml4co-kit==0.3.3")
        print("Or set USE_DATA_GENERATOR = False and use dataset file instead\n")
        return

    # Create data generator if needed
    data_generator = None
    if USE_DATA_GENERATOR:
        data_generator = create_data_generator(GENERATOR_CONFIG)
        if data_generator is None:
            print("Failed to create data generator. Exiting.")
            return

    # Use custom trainer with ML4CO support
    trainer = CVRPTrainerML4CO(
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params
    )

    # Attach data generator to environment if using
    if USE_DATA_GENERATOR and data_generator is not None:
        trainer.env.data_generator = data_generator
        print(f"\n✓ Data generator attached to environment")

    copy_all_src(trainer.result_folder)

    print("\n" + "="*80)
    print(f"Training UniteFormer on CVRP{cvrp_size}")
    print("="*80)
    print(f"Problem Size: {CVRP_SIZE}")
    print(f"Data Generation: {'ENABLED' if USE_DATA_GENERATOR else 'DISABLED'}")
    if USE_DATA_GENERATOR:
        print(f"Generator: CVRPDataGenerator")
        print(f"Distribution: {GENERATOR_CONFIG['distribution_type']}")
        print(f"Capacity: {GENERATOR_CONFIG['capacity']}")
    elif USE_DATASET_FILE:
        print(f"Dataset Path: {DATASET_PATH}")
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
    logger.info('USE_DATA_GENERATOR: {}'.format(USE_DATA_GENERATOR))
    logger.info('USE_DATASET_FILE: {}'.format(USE_DATASET_FILE))
    logger.info('ML4CO_AVAILABLE: {}'.format(ML4CO_AVAILABLE))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
