"""
ML4CO Data Loader for UniteFormer

This module provides data loading utilities to use ML4CO datasets with UniteFormer models.
It converts ML4CO-Kit data format to UniteFormer's expected format.
"""

import torch
import numpy as np
from typing import Tuple, Optional
import os


class ML4CODataLoader:
    """
    DataLoader for converting ML4CO datasets to UniteFormer format.

    This class handles loading TSP and CVRP datasets from ML4CO format
    and converting them to the format expected by UniteFormer.
    """

    def __init__(self, problem_type: str = 'tsp', precision=np.float32):
        """
        Initialize ML4CODataLoader.

        Args:
            problem_type: Type of problem ('tsp' or 'cvrp')
            precision: Numerical precision (default: np.float32)
        """
        self.problem_type = problem_type.lower()
        self.precision = precision

        # Try to import ml4co_kit
        try:
            import ml4co_kit
            self.ml4co_kit = ml4co_kit
            self.ml4co_available = True
        except ImportError:
            print("Warning: ml4co_kit not available. Install with: pip install ml4co-kit")
            self.ml4co_available = False

    def load_tsp_data(self, file_path: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Load TSP data from ML4CO format file.

        Args:
            file_path: Path to the TSP data file

        Returns:
            Tuple of (nodes, solutions)
                - nodes: Tensor of shape (num_samples, num_nodes, 2)
                - solutions: Tensor of shape (num_samples, num_nodes) or None
        """
        if not self.ml4co_available:
            raise ImportError("ml4co_kit is required to load ML4CO datasets")

        # Use TSPUniformDataset to load TSP data
        try:
            dataset = self.ml4co_kit.TSPUniformDataset(
                file_path=file_path,
                num_nodes=None,  # Auto-detect from file
                preload=True
            )

            nodes_list = []
            solutions_list = []

            # Extract data from dataset
            for i in range(len(dataset)):
                instance = dataset[i]
                nodes_list.append(instance.nodes)
                if hasattr(instance, 'solution') and instance.solution is not None:
                    solutions_list.append(instance.solution)

            # Convert to tensors
            if len(nodes_list) > 0:
                nodes_tensor = torch.tensor(np.array(nodes_list), dtype=torch.float32)
            else:
                nodes_tensor = torch.empty((0, 0, 2), dtype=torch.float32)

            solutions_tensor = None
            if len(solutions_list) > 0:
                solutions_tensor = torch.tensor(np.array(solutions_list), dtype=torch.long)

            return nodes_tensor, solutions_tensor

        except Exception as e:
            raise RuntimeError(f"Failed to load TSP data from {file_path}: {e}")

    def load_cvrp_data(self, file_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Load CVRP data from ML4CO format file.

        Args:
            file_path: Path to the CVRP data file

        Returns:
            Tuple of (depot_xy, node_xy, node_demand, capacity, solutions)
                - depot_xy: Tensor of shape (num_samples, 1, 2)
                - node_xy: Tensor of shape (num_samples, num_nodes, 2)
                - node_demand: Tensor of shape (num_samples, num_nodes)
                - capacity: Tensor of shape (num_samples,)
                - solutions: Tensor of shape (num_samples, num_nodes, 2) or None
        """
        if not self.ml4co_available:
            raise ImportError("ml4co_kit is required to load ML4CO datasets")

        # Use CVRPUniformDataset to load CVRP data
        try:
            dataset = self.ml4co_kit.CVRPUniformDataset(
                file_path=file_path,
                num_nodes=None,  # Auto-detect from file
                preload=True
            )

            depot_list = []
            node_xy_list = []
            demand_list = []
            capacity_list = []
            solutions_list = []

            # Extract data from dataset
            for i in range(len(dataset)):
                instance = dataset[i]
                depot_list.append(instance.depot_xy)
                node_xy_list.append(instance.node_xy)
                demand_list.append(instance.demand)
                capacity_list.append(instance.capacity)

                if hasattr(instance, 'solution') and instance.solution is not None:
                    solutions_list.append(instance.solution)

            # Convert to tensors
            depot_tensor = torch.tensor(np.array(depot_list), dtype=torch.float32)
            node_xy_tensor = torch.tensor(np.array(node_xy_list), dtype=torch.float32)
            demand_tensor = torch.tensor(np.array(demand_list), dtype=torch.float32)
            capacity_tensor = torch.tensor(np.array(capacity_list), dtype=torch.float32)

            solutions_tensor = None
            if len(solutions_list) > 0:
                solutions_tensor = torch.tensor(np.array(solutions_list), dtype=torch.long)

            return depot_tensor, node_xy_tensor, demand_tensor, capacity_tensor, solutions_tensor

        except Exception as e:
            raise RuntimeError(f"Failed to load CVRP data from {file_path}: {e}")

    def _convert_cvrp_solutions(self, solutions_list):
        """Convert CVRP solutions to UniteFormer format."""
        # This depends on the specific format of ML4CO CVRP solutions
        # Adjust based on actual format
        converted = []
        for sol in solutions_list:
            # Convert solution format if needed
            converted.append(sol)
        return torch.tensor(np.array(converted), dtype=torch.long)

    def load_raw_tsp_for_uniteformer(self, file_path: str, num_samples: int = None):
        """
        Load TSP data and convert to UniteFormer's raw_data format.

        This creates the format expected by TSPEnv.load_raw_data()

        Args:
            file_path: Path to the TSP data file
            num_samples: Maximum number of samples to load (None for all)

        Returns:
            Tuple of (raw_data_nodes, raw_data_tours)
        """
        nodes, solutions = self.load_tsp_data(file_path)

        if num_samples is not None:
            nodes = nodes[:num_samples]
            if solutions is not None:
                solutions = solutions[:num_samples]

        return nodes, solutions

    def load_raw_cvrp_for_uniteformer(self, file_path: str, num_samples: int = None):
        """
        Load CVRP data and convert to UniteFormer's raw_data format.

        This creates the format expected by CVRPEnv.load_raw_data()

        Args:
            file_path: Path to the CVRP data file
            num_samples: Maximum number of samples to load (None for all)

        Returns:
            Dictionary with CVRP data in UniteFormer format
        """
        depot, node_xy, demand, capacity, solutions = self.load_cvrp_data(file_path)

        # Combine depot and nodes for UniteFormer format
        # UniteFormer expects: problems_nodes[B, V+1, 2], Batch_demand[B, V+1]
        problems_nodes = torch.cat([depot, node_xy], dim=1)

        # Add depot demand (zeros)
        depot_demand = torch.zeros((demand.shape[0], 1))
        batch_demand = torch.cat([depot_demand, demand], dim=1)

        if num_samples is not None:
            problems_nodes = problems_nodes[:num_samples]
            batch_demand = batch_demand[:num_samples]
            capacity = capacity[:num_samples]
            if solutions is not None:
                solutions = solutions[:num_samples]

        return {
            'raw_data_nodes': problems_nodes,
            'raw_data_demand': batch_demand,
            'raw_data_capacity': capacity,
            'raw_data_node_flag': solutions
        }


def convert_uniteformer_to_ml4co_format(nodes: torch.Tensor, tour: torch.Tensor) -> str:
    """
    Convert UniteFormer prediction to ML4CO format string.

    Args:
        nodes: Node coordinates (num_nodes, 2)
        tour: Tour indices (num_nodes,)

    Returns:
        String in ML4CO format: "x1 y1 x2 y2 ... xn yn output t1 t2 ... tn"
    """
    nodes_np = nodes.cpu().numpy()
    tour_np = tour.cpu().numpy() + 1  # Convert to 1-indexed

    parts = []
    for i in range(len(nodes_np)):
        parts.extend([f"{nodes_np[i, 0]:.6f}", f"{nodes_np[i, 1]:.6f}"])

    parts.append("output")
    parts.extend([str(int(t)) for t in tour_np])

    return " ".join(parts) + "\n"


def save_predictions_to_ml4co_format(predictions: list, nodes_list: list, output_path: str):
    """
    Save UniteFormer predictions to ML4CO format file.

    Args:
        predictions: List of tour predictions
        nodes_list: List of node coordinates
        output_path: Path to save the predictions
    """
    with open(output_path, 'w') as f:
        for tour, nodes in zip(predictions, nodes_list):
            line = convert_uniteformer_to_ml4co_format(nodes, tour)
            f.write(line)


if __name__ == "__main__":
    # Example usage
    print("ML4CO Data Loader for UniteFormer")
    print("This module provides utilities to load ML4CO datasets for UniteFormer training/testing")
