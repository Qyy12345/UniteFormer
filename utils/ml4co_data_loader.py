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

        Supports both ML4CO-Bench-101 format and ml4co_kit CVRPUniformDataset.

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
        # First, try to parse as ML4CO-Bench-101 format
        try:
            return self._load_cvrp_bench101_format(file_path)
        except Exception as e:
            # Print the actual error for debugging
            import traceback
            print(f"Warning: ML4CO-Bench-101 format parsing failed: {e}")
            traceback.print_exc()
            pass  # Fall through to ml4co_kit format

        # If ml4co_kit is available, try CVRPUniformDataset
        if self.ml4co_available:
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
                print(f"Warning: ml4co_kit CVRPUniformDataset failed: {e}")
                pass  # Fall through to error

        raise RuntimeError(f"Failed to load CVRP data from {file_path}: Unsupported format")

    def _load_cvrp_bench101_format(self, file_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Load CVRP data from ML4CO-Bench-101 format file.

        Format: depots <x> <y> points <x1> <y1> <x2> <y2> ... demands <d1> <d2> ... capacity <C> output <tour>

        Args:
            file_path: Path to the CVRP data file

        Returns:
            Tuple of (depot_xy, node_xy, node_demand, capacity, solutions)
        """
        depot_list = []
        node_xy_list = []
        demand_list = []
        capacity_list = []
        solutions_list = []

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()

                # Check if this is ML4CO-Bench-101 format
                if 'depots' not in parts:
                    raise ValueError(f"Not ML4CO-Bench-101 format: missing 'depots' keyword")

                # Parse depot
                depot_idx = parts.index('depots')
                depot_x = float(parts[depot_idx + 1])
                depot_y = float(parts[depot_idx + 2])
                depot_list.append([depot_x, depot_y])

                # Parse points (customer nodes)
                points_idx = parts.index('points')
                demands_idx = parts.index('demands')

                # Extract customer coordinates
                points = []
                idx = points_idx + 1
                while idx < demands_idx:
                    x = float(parts[idx])
                    y = float(parts[idx + 1])
                    points.append([x, y])
                    idx += 2
                node_xy_list.append(points)

                # Parse demands
                capacity_idx = parts.index('capacity')
                demands = []
                idx = demands_idx + 1
                while idx < capacity_idx:
                    demands.append(float(parts[idx]))
                    idx += 1
                demand_list.append(demands)

                # Parse capacity
                cap = float(parts[capacity_idx + 1])
                capacity_list.append(cap)

                # Parse solution (tour) if exists
                try:
                    output_idx = parts.index('output')
                    tour = []
                    idx = output_idx + 1
                    while idx < len(parts):
                        tour.append(int(parts[idx]))
                        idx += 1
                    solutions_list.append(tour)
                except ValueError:
                    pass  # No solution in this line

        if len(depot_list) == 0:
            raise ValueError(f"No valid data found in {file_path}")

        # Convert to tensors
        depot_tensor = torch.tensor(np.array(depot_list), dtype=torch.float32).unsqueeze(1)  # (N, 1, 2)
        node_xy_tensor = torch.tensor(np.array(node_xy_list), dtype=torch.float32)  # (N, V, 2)
        demand_tensor = torch.tensor(np.array(demand_list), dtype=torch.float32)  # (N, V)
        capacity_tensor = torch.tensor(np.array(capacity_list), dtype=torch.float32)  # (N,)

        solutions_tensor = None
        if len(solutions_list) > 0:
            # Convert tour to node_flag format (2-column format for UniteFormer)
            # Format: [visited_order, is_depot_flag]
            # Since tours have different lengths, store as list of lists
            node_flag_list = []
            for tour in solutions_list:
                # Create a list of [node, depot_flag] pairs
                # depot_flag: 1 if depot, 0 if customer
                node_flag = []
                for node in tour:
                    if node == 0:
                        # Depot
                        node_flag.extend([0, 1])  # UniteFormer uses 0-indexed depot with flag
                    else:
                        # Customer (convert to 0-indexed)
                        node_flag.extend([node - 1, 0])
                node_flag_list.append(node_flag)

            # Keep as list of lists since tours have different lengths
            # The CVRP environment will handle this format
            solutions_tensor = node_flag_list

        return depot_tensor, node_xy_tensor, demand_tensor, capacity_tensor, solutions_tensor

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
