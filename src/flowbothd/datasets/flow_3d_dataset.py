import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, NamedTuple
import json


class FlowTrajectoryData(TypedDict):
    id: str
    pos: npt.NDArray[np.float32]  # (N, 3): Point cloud observation.
    delta: npt.NDArray[np.float32]  # (N, K, 3): Ground-truth flow.
    point: npt.NDArray[np.float32]  # (N, K, 3): Ground-truth waypoints.
    mask: npt.NDArray[np.bool_]  #  (N,): Mask the point of interest.


class PointCloudInfo(NamedTuple):
    """Information about a point cloud and its sequence."""
    sequence_id: str
    cloud_idx: int
    absolute_idx: int  # Global index in dataset


class Flow3DDataset:
    def __init__(
        self,
        root: str,
        split: str = "train",
        n_points: Optional[int] = None,
        cache_size: int = 10,
    ) -> None:
        """Dataset adapter for Flow3D data format to FlowBot format.
        
        Args:
            root (str): Root directory containing flow_3d_data
            split (str): Dataset split (train/val/test)
            n_points (Optional[int]): Whether to downsample points to this number
            cache_size (int): Number of sequences to cache in memory
        """
        self.root = Path(root) / split
        self.n_points = n_points
        self.cache_size = cache_size
        self.cache = {}  # Cache for storing loaded sequences
        
        # Get all sequence directories
        self.sequences = [d for d in self.root.iterdir() if d.is_dir()]
        if not self.sequences:
            raise ValueError(f"No sequences found in {self.root}")
            
        # Get all valid point clouds across sequences
        self.cloud_infos = self._get_valid_clouds()

    def _get_valid_clouds(self) -> List[PointCloudInfo]:
        """Get all valid starting point clouds across all sequences."""
        valid_clouds = []
        absolute_idx = 0
        
        for sequence_path in self.sequences:
            # Load metadata to get number of valid frames
            with open(sequence_path / "metadata.json", 'r') as f:
                metadata = json.load(f)
            
            num_valid_frames = metadata['num_valid_frames']
            
            if num_valid_frames <= 0:
                print(f"Warning: Sequence {sequence_path.name} has no valid frames. Skipping sequence.")
                continue
                
            for i in range(num_valid_frames):
                valid_clouds.append(PointCloudInfo(sequence_path.name, i, absolute_idx))
                absolute_idx += 1
                
        if not valid_clouds:
            raise ValueError(f"No valid point clouds found in any sequence.")
                
        return valid_clouds

    def _load_sequence_data(self, sequence_id: str):
        """Load point clouds and motion data for a sequence with caching."""
        # Try to get from cache first
        if sequence_id in self.cache:
            return self.cache[sequence_id]
        
        # Load data
        sequence_path = self.root / sequence_id
        points = np.load(sequence_path / "point_clouds.npy")  # (num_frames, num_points, 3)
        motions = np.load(sequence_path / "point_motions.npy")  # (num_valid_frames, num_points, prediction_horizon, 3)
        
        # Update cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest item if cache is full
            self.cache.pop(next(iter(self.cache)))
        self.cache[sequence_id] = (points, motions)
            
        return points, motions

    def get_data(self, idx: int) -> FlowTrajectoryData:
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        
        cloud_info = self.cloud_infos[idx]
        
        # Load sequence data
        points, motions = self._load_sequence_data(cloud_info.sequence_id)
        
        # Get the relevant slice for this index
        pos = points[cloud_info.cloud_idx]  # (N, 3)
        delta = motions[cloud_info.cloud_idx]  # (N, K, 3)
        
        # Calculate waypoints by accumulating motions
        point = np.zeros_like(delta)  # (N, K, 3)
        current_pos = pos.copy()
        for k in range(delta.shape[1]):
            current_pos = current_pos + delta[:, k]
            point[:, k] = current_pos
        
        # Create mask - include all points
        mask = np.ones(pos.shape[0], dtype=np.float32)
        
        # Optionally downsample points
        if self.n_points is not None and self.n_points < pos.shape[0]:
            indices = np.random.choice(pos.shape[0], self.n_points, replace=False)
            pos = pos[indices]
            delta = delta[indices]
            point = point[indices]
            mask = mask[indices]
        
        return {
            "id": cloud_info.sequence_id,
            "pos": pos.astype(np.float32),
            "delta": delta.astype(np.float32),
            "point": point.astype(np.float32),
            "mask": mask.astype(np.float32),
        }

    def __getitem__(self, idx: int) -> FlowTrajectoryData:
        return self.get_data(idx)

    def __len__(self) -> int:
        return len(self.cloud_infos) 