#!/usr/bin/env python3

import hydra
import torch
from omegaconf import OmegaConf

from flowbothd.datasets.flow_3d import Flow3DDataModule


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg):
    # Override config to use flow3d dataset
    cfg.dataset = OmegaConf.load("../configs/dataset/flow3d.yaml")
    
    # Create datamodule
    datamodule = Flow3DDataModule(
        root=cfg.dataset.data_dir,
        batch_size=2,  # Small batch for testing
        num_workers=0,  # No multiprocessing for testing
        n_points=cfg.dataset.n_points,
        trajectory_len=cfg.dataset.trajectory_len,
        randomize_size=cfg.dataset.randomize_size,
        augmentation=cfg.dataset.augmentation,
    )
    
    # Set up datasets
    datamodule.setup()
    
    # Test dataloaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    train_val_loader = datamodule.train_val_dataloader()
    unseen_loader = datamodule.unseen_dataloader()
    
    # Get a batch from each loader
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    train_val_batch = next(iter(train_val_loader))
    unseen_batch = next(iter(unseen_loader))
    
    # Print shapes to verify
    print("\nTrain batch:")
    for k, v in train_batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")
    
    print("\nValidation batch:")
    for k, v in val_batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")
    
    print("\nTrain-val batch:")
    for k, v in train_val_batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")
    
    print("\nUnseen batch:")
    for k, v in unseen_batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")
    
    print("\nAll dataloaders work correctly!")


if __name__ == "__main__":
    main() 