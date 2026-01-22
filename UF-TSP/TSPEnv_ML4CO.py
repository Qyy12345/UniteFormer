"""
Enhanced TSP Environment for UniteFormer with ML4CO Dataset Support

This module extends the original TSPEnv to support loading and using ML4CO datasets.
"""

from dataclasses import dataclass
import torch
from TSProblemDef import get_random_problems, get_edge_node_problems, augment_xy_data_by_8_fold
from tqdm import tqdm
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from ml4co_data_loader import ML4CODataLoader


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)


class TSPEnvML4CO:
    """
    Enhanced TSP Environment with ML4CO dataset support.

    This environment extends the original TSPEnv to support:
    1. Loading ML4CO format datasets
    2. Using ML4CO-Kit wrappers for data management
    3. Compatible with original UniteFormer training pipeline
    """

    def __init__(self, **env_params):
        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.num_neighbors = env_params['num_neighbors']
        self.data_path = env_params['data_path']
        self.mode = env_params['mode']
        self.use_ml4co = env_params.get('use_ml4co', False)
        self.ml4co_data_format = env_params.get('ml4co_data_format', 'wrapper')  # 'wrapper' or 'direct'

        # Data generation support
        self.use_data_generator = env_params.get('use_data_generator', False)
        self.generator_config = env_params.get('generator_config', None)
        self.data_generator = None  # Will be attached externally if needed

        self.raw_data_nodes = None
        self.raw_data_tours = None

        # ML4CO specific
        self.ml4co_loader = None
        if self.use_ml4co:
            self.ml4co_loader = ML4CODataLoader(problem_type='tsp')

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)
        self.solutions = None
        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

        self.x_edges = None
        self.x_edges_values = None
        self.x_node_indices = None
        self.x_nodes_false = None
        self.noaug_problems = None

    def load_problems(self, episode, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if self.mode == 'train':
            # Training: Generate problems
            if self.use_data_generator and self.data_generator is not None:
                # Use ML4CO-Kit's TSPDataGenerator
                try:
                    # Generate instances
                    instances = self.data_generator.generate_only_instance_for_us(batch_size)
                    self.problems = torch.tensor(instances, dtype=torch.float32)
                except Exception as e:
                    print(f"Warning: Data generation failed ({e}), falling back to random generation")
                    self.problems = get_random_problems(batch_size, self.problem_size)
            else:
                # Use original random problem generation
                self.problems = get_random_problems(batch_size, self.problem_size)
        else:
            # Testing/Validation: Load from dataset
            if self.use_ml4co and self.ml4co_loader is not None:
                # Use ML4CO data loader
                if self.raw_data_nodes is None:
                    # Load data if not already loaded
                    if self.ml4co_data_format == 'wrapper':
                        self.raw_data_nodes, self.raw_data_tours = \
                            self.ml4co_loader.load_raw_tsp_for_uniteformer(
                                self.data_path, num_samples=None
                            )

                self.problems = self.raw_data_nodes[episode: episode + batch_size]
                if self.raw_data_tours is not None:
                    self.solutions = self.raw_data_tours[episode: episode + batch_size]
                else:
                    self.solutions = None
            else:
                # Use original UniteFormer loading method
                self.problems, self.solutions = \
                    self.raw_data_nodes[episode: episode + batch_size], \
                    self.raw_data_tours[episode: episode + batch_size]

        self.x_edges, self.x_edges_values = get_edge_node_problems(self.problems, self.num_neighbors)

        self.noaug_problems = self.problems
        if aug_factor > 1:
            self.batch_size = self.batch_size * aug_factor
            if aug_factor == 8:  # test
                self.problems = augment_xy_data_by_8_fold(self.problems)
                self.x_edges = self.x_edges.repeat(aug_factor, 1, 1)
                self.x_edges_values = self.x_edges_values.repeat(aug_factor, 1, 1)
                # shape: (8*batch, problem, 2)

            else:
                self.problems = self.problems.repeat(aug_factor, 1, 1)
                self.x_edges = self.x_edges.repeat(aug_factor, 1, 1)
                self.x_edges_values = self.x_edges_values.repeat(aug_factor, 1, 1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def load_raw_data(self, episode, begin_index=0):
        """
        Load raw dataset from file.

        Compatible with both original UniteFormer format and ML4CO format.
        """
        print('load raw dataset begin!')

        if self.use_ml4co and self.ml4co_loader is not None:
            # Use ML4CO loader
            self.raw_data_nodes, self.raw_data_tours = \
                self.ml4co_loader.load_raw_tsp_for_uniteformer(
                    self.data_path, num_samples=episode
                )
            print(f'load raw dataset done! Loaded {len(self.raw_data_nodes)} samples')
            return

        # Original loading method
        self.raw_data_nodes = []
        self.raw_data_tours = []
        for line in tqdm(open(self.data_path, "r").readlines()[0 + begin_index: episode + begin_index], ascii=True):
            line = line.split(" ")
            num_nodes = int(line.index('output') // 2)
            nodes = [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]
            self.raw_data_nodes.append(nodes)
            tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]]

            self.raw_data_tours.append(tour_nodes)

        self.raw_data_nodes = torch.tensor(self.raw_data_nodes, requires_grad=False)
        self.raw_data_tours = torch.tensor(self.raw_data_tours, requires_grad=False)
        print(f'load raw dataset done!')

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
        # shape: (batch, pomo, problem)

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, node)

        # returning values
        done = (self.selected_count == self.problem_size)
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None
        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = self.problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 2)
        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def _get_best_distance(self, batch_size):
        if self.solutions is None:
            return None

        gathering_index = self.solutions.unsqueeze(2).expand(batch_size, self.problem_size, 2)
        # shape: (batch, problem, 2)
        seq_expanded = self.noaug_problems
        ordered_seq = seq_expanded.gather(dim=-2, index=gathering_index)
        # shape: (batch, problem, 2)
        rolled_seq = ordered_seq.roll(dims=-2, shifts=-1)  # shifts=-1
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(2).sqrt()
        # shape: (batch, problem)
        travel_distances = segment_lengths.sum(-1)
        # shape: (batch, pomo)
        return travel_distances
