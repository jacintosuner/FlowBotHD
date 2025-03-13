from typing import Optional, Protocol

import numpy as np
import torch
import torch_geometric.data as tgd

from flowbothd.datasets.flow_3d_dataset import Flow3DDataset


class Flow3DTGData(Protocol):
    id: str  # Sequence ID
    pos: torch.Tensor  # (N, 3) Points in the point cloud
    delta: torch.Tensor  # (N, K, 3) Flow trajectories
    point: torch.Tensor  # (N, K, 3) Trajectory waypoints
    mask: torch.Tensor  # (N,) Mask of points with motion


class Flow3DPyGDataset(tgd.Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        n_points: Optional[int] = None,
        randomize_size: bool = False,
        augmentation: bool = False,
        seed: int = 42,
    ) -> None:
        """PyG version of Flow3D dataset with augmentation capabilities.
        
        Args:
            root: Root directory containing flow_3d_data
            split: Dataset split (train/val/test)
            n_points: If set, randomly sample this many points
            randomize_size: Whether to randomly scale the point cloud
            augmentation: Whether to apply random flips
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.dataset = Flow3DDataset(
            root=root,
            split=split,
            n_points=n_points,
        )
        self.randomize_size = randomize_size
        self.augmentation = augmentation
        self.seed = seed

    def len(self) -> int:
        return len(self.dataset)

    def get(self, idx: int) -> Flow3DTGData:
        # Get numpy data from base dataset
        data_dict = self.dataset[idx]
        
        # Random scaling
        rsz = 1.0 if not self.randomize_size else np.random.uniform(0.1, 5)
        
        # Random flips (4 modes: normal, left-right, front-back, both)
        flip = 0 if not self.augmentation else np.random.randint(0, 4)
        flip_mat = torch.tensor([
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Normal
            [[1, 0, 0], [0, -1, 0], [0, 0, 1]],  # Left-right
            [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Front-back
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],  # Both
        ]).float()

        # Convert numpy arrays to tensors and apply augmentations
        pos = torch.matmul(
            torch.from_numpy(data_dict["pos"]).float() * rsz,
            flip_mat[flip]
        )
        
        # Process all trajectory points (K=3)
        delta = torch.matmul(
            torch.from_numpy(data_dict["delta"]).float(),  # Use all points
            flip_mat[flip]
        )
        
        point = torch.matmul(
            torch.from_numpy(data_dict["point"]).float() * rsz,  # Use all points
            flip_mat[flip]
        )
        
        # Convert to PyG Data object
        data = tgd.Data(
            id=data_dict["id"],
            pos=pos,
            delta=delta,
            point=point,
            mask=torch.from_numpy(data_dict["mask"]).float())
        
        return data

    @staticmethod
    def get_processed_dir(
        split: str,
        n_points: Optional[int] = None,
        randomize_size: bool = False,
        augmentation: bool = False,
    ) -> str:
        """Get the processed directory name based on dataset parameters."""
        n_points_str = f"_{n_points}p" if n_points else ""
        random_size_str = "_rsz" if randomize_size else ""
        augmentation_str = "_aug" if augmentation else ""
        
        return f"processed_{split}{n_points_str}{random_size_str}{augmentation_str}" 