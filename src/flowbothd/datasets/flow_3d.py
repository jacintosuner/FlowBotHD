import lightning as L
from torch.utils.data import DataLoader
from typing import Optional
from torch_geometric.loader import DataLoader as PyGDataLoader

from flowbothd.datasets.flow_3d_dataset_pyg import Flow3DPyGDataset


class Flow3DDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        n_proc: int = 1,
        seed: int = 42,
        history: bool = False,
        randomize_size: bool = False,
        augmentation: bool = False,
        trajectory_len: int = 5,
        special_req: Optional[str] = None,
        n_repeat: int = 1,
        toy_dataset=None,
        n_points: Optional[int] = None,
    ):
        """Lightning DataModule for Flow3D dataset.
        
        Args:
            root: Root directory containing flow_3d_data
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            n_proc: Number of processes per worker
            seed: Random seed
            history: Whether to use history (not used in Flow3D)
            randomize_size: Whether to randomly scale point clouds
            augmentation: Whether to apply random flips
            trajectory_len: Length of trajectory sequences
            special_req: Special requirements (not used in Flow3D)
            n_repeat: Number of times to repeat dataset (not used in Flow3D)
            toy_dataset: Toy dataset configuration (not used in Flow3D)
            n_points: If set, randomly sample this many points
        """
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.randomize_size = randomize_size
        self.augmentation = augmentation
        self.n_points = n_points
        self.trajectory_len = trajectory_len  # Store trajectory_len
        
        # Create datasets
        common_args = dict(
            n_points=self.n_points,
            seed=self.seed,
        )
        
        self.train_dataset = Flow3DPyGDataset(
            root=self.root,
            split="train",
            randomize_size=self.randomize_size,
            augmentation=self.augmentation,
            **common_args
        )
        
        self.val_dataset = Flow3DPyGDataset(
            root=self.root,
            split="val",
            randomize_size=False,  # No augmentation for validation
            augmentation=False,
            **common_args
        )
        
        self.test_dataset = Flow3DPyGDataset(
            root=self.root,
            split="test",
            randomize_size=False,  # No augmentation for testing
            augmentation=False,
            **common_args
        )

    def train_dataloader(self):
        return PyGDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self, bsz=None):
        bsz = self.batch_size if bsz is None else bsz
        return PyGDataLoader(
            self.val_dataset,
            batch_size=bsz,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self, bsz=None):
        bsz = self.batch_size if bsz is None else bsz
        return PyGDataLoader(
            self.test_dataset,
            batch_size=bsz,
            shuffle=False,
            num_workers=self.num_workers,
        )
        
    def train_val_dataloader(self, bsz=None):
        """Get a dataloader for validation on training data."""
        bsz = self.batch_size if bsz is None else bsz
        train_val_dataset = Flow3DPyGDataset(
            root=self.root,
            split="train",
            n_points=self.n_points,
            randomize_size=False,  # No augmentation for validation
            augmentation=False,
            seed=self.seed,
        )
        return PyGDataLoader(
            train_val_dataset,
            batch_size=bsz,
            shuffle=False,
            num_workers=self.num_workers,
        )
        
    def unseen_dataloader(self, bsz=None):
        """Get a dataloader for unseen data (using test split)."""
        return self.test_dataloader(bsz) 