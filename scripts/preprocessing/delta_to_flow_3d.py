#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import json
from typing import Optional
from omegaconf import OmegaConf


class DeltaToFlow3D:
    """Converts DELTA tracking results to SceneFlowDataset format.
    
    This class handles the conversion of 3D tracking data from DELTA format to the format
    required by SceneFlowDataset. Each sequence is preprocessed to ensure:
    1. Point correspondence across frames
    2. Consistent number of points
    3. Valid motion trajectories
    4. Proper train/val/test organization
    
    Input Format:
        input_dir/
            sequence_000/
                dense_3d_track.pkl  # Contains DELTA tracking results with structure:
                    {
                        'coords': array of shape (num_frames, num_points, 3),
                        'colors': array of shape (num_points, 3),
                        'vis': array of shape (num_frames, num_points)
                    }
            sequence_001/
                dense_3d_track.pkl
            ...
    
    Output Format:
        output_dir/
            train/
                sequence_000/
                    point_clouds.npy     # Shape: (num_frames, num_points, 3)
                    point_motions.npy    # Shape: (num_valid_frames, num_points, prediction_horizon, 3)
                                        # This file contains motion vectors for sequential future predictions:
                                        # - First dimension (num_valid_frames): Each index t represents a source frame
                                        # - Second dimension (num_points): Each point in the point cloud
                                        # - Third dimension (prediction_horizon): Sequential motion predictions where:
                                        #   * point_motions[t, p, h] represents the displacement vector from
                                        #     the position at frame (t + h*frame_step) to the position at
                                        #     frame (t + (h+1)*frame_step) for point p, where frame_step
                                        #     is computed as round(fps * seconds_per_prediction)
                                        #   * Example with fps=15, seconds_per_prediction=0.5 (frame_step=8), prediction_horizon=3:
                                        #     - h=0: motion from t to t+8 (0.5 seconds)
                                        #     - h=1: motion from t+8 to t+16 (next 0.5 seconds)
                                        #     - h=2: motion from t+16 to t+24 (next 0.5 seconds)
                                        #   Each prediction represents the motion over the specified time interval.
                                        # - Fourth dimension (3): XYZ components of the motion vector
                                        #
                                        # Note: num_valid_frames = num_frames - (prediction_horizon * frame_step)
                                        # because we need enough future frames for all predictions
                    metadata.json        # Contains sequence info and statistics
                sequence_001/
                    ...
            val/
                ...
            test/
                ...
                
    Notes:
        - Each input sequence directory must contain a 'dense_3d_track.pkl' file
        - The 'coords' field in the pickle file must be a numpy array of shape (F, N, 3)
          where F is the number of frames and N is the number of tracked points
        - Points in 'coords' must maintain correspondence across frames
          (i.e., point i in frame t corresponds to point i in frame t+1)
        - The converter can optionally normalize the number of points across all sequences
          using the num_points parameter
        - The prediction_horizon parameter determines how many future predictions to compute
          for each source frame
        - The frame_step parameter is computed as round(fps * seconds_per_prediction)
          and determines the temporal distance / separation between predictions. For example, with
          fps=15 and seconds_per_prediction=0.5, frame_step=8, meaning each prediction
          covers a 0.5 second interval
    """
    
    def __init__(
        self,
        config: OmegaConf
    ):
        """Initialize the converter.
        
        Args:
            config: OmegaConf configuration object containing dataset parameters
        """
        self.input_path = Path(config.input_dir)
        self.output_path = Path(config.output_dir)
        self.splits = config.splits
        self.num_points = config.num_points
        self.prediction_horizon = config.prediction_horizon
        
        # Compute frame_step from fps and seconds_per_prediction
        self.frame_step = round(config.fps * config.seconds_per_prediction)
        print(f"Computing frame_step as round({config.fps} * {config.seconds_per_prediction}) = {self.frame_step}")
        
        self.distance_threshold = config.distance_threshold
        
        # Clear and recreate output directory structure
        for split in ["train", "val", "test"]:
            split_dir = self.output_path / split
            if split_dir.exists():
                print(f"Clearing existing data from {split} directory...")
                # Remove all contents of the split directory
                for item in split_dir.iterdir():
                    if item.is_file():
                        item.unlink()  # Delete files
                    elif item.is_dir():
                        import shutil
                        shutil.rmtree(item)  # Delete directories and their contents
            # Create fresh split directory
            split_dir.mkdir(parents=True, exist_ok=True)

    def process_sequence(self, sequence_path: Path, split: str) -> None:
        """Process a single sequence directory containing DELTA tracking results.
        
        Args:
            sequence_path: Path to sequence directory containing dense_3d_track.pkl
            split: Dataset split (train/val/test)
        """
        # Load tracking data
        tracking_file = sequence_path / "dense_3d_track.pkl"
        if not tracking_file.exists():
            print(f"Warning: No tracking data found at {tracking_file}")
            return
            
        with open(tracking_file, "rb") as f:
            tracking_data = pickle.load(f)
        
        # Create sequence output directory
        sequence_name = sequence_path.name
        sequence_out_path = self.output_path / split / sequence_name
        sequence_out_path.mkdir(parents=True, exist_ok=True)
        
        # Extract point trajectories
        points = tracking_data['coords']  # Shape: (num_frames, num_points, 3)
        num_frames, num_points_orig, _ = points.shape
        
        # Filter points based on distance threshold
        # Calculate distances from origin for each point in each frame
        distances = np.linalg.norm(points, axis=2)  # Shape: (num_frames, num_points)
        # A point is kept if it's within threshold in all frames
        valid_points_mask = np.all(distances <= self.distance_threshold, axis=0)
        points = points[:, valid_points_mask, :]
        num_points_after_filter = points.shape[1]


        if num_points_after_filter == 0:
            print(f"Warning: All points in sequence {sequence_name} were filtered out by distance threshold {self.distance_threshold}")
            return
        
        # Handle point count adjustment if needed
        if self.num_points is not None:
            if num_points_after_filter > self.num_points:
                # Randomly subsample points without replacement
                indices = np.random.choice(num_points_after_filter, self.num_points, replace=False)
                points = points[:, indices, :]
            elif num_points_after_filter < self.num_points:
                # Sample additional points with replacement to maintain point distribution
                num_points_to_add = self.num_points - num_points_after_filter
                additional_indices = np.random.choice(num_points_after_filter, num_points_to_add, replace=True)
                additional_points = points[:, additional_indices, :]
                points = np.concatenate([points, additional_points], axis=1)
                print(f"Sampled {num_points_to_add} additional points with replacement to reach {self.num_points} points")
        
        # Calculate number of valid frames for motion computation
        # We need enough frames for prediction_horizon sequential predictions
        # Each prediction covers seconds_per_prediction time (frame_step frames),
        # so total frames needed is: prediction_horizon * frame_step
        max_future_frame = (self.prediction_horizon * self.frame_step)
        num_valid_frames = num_frames - max_future_frame
        
        if num_valid_frames <= 0:
            print(f"Warning: Sequence {sequence_name} is too short for the specified prediction horizon "
                  f"and temporal stride. Need at least {max_future_frame + 1} frames, but got {num_frames}.")
            return
        
        # For each starting frame, compute the ground truth sequential motions
        # These motions represent what the model should learn to predict from just the initial frame
        point_motions = np.zeros((num_valid_frames, points.shape[1], self.prediction_horizon, 3))
        
        for t in range(num_valid_frames):
            # For each starting frame t, compute motions over seconds_per_prediction intervals:
            # - Motion 0: t -> t+frame_step (first interval of seconds_per_prediction)
            # - Motion 1: t+frame_step -> t+2*frame_step (second interval)
            # - Motion 2: t+2*frame_step -> t+3*frame_step (third interval)
            # The model will need to predict all these motions just from the points at frame t
            for h in range(self.prediction_horizon):
                source_frame = t + (h * self.frame_step)
                target_frame = source_frame + self.frame_step
                point_motions[t, :, h, :] = points[target_frame] - points[source_frame]
        
        # Save processed data
        np.save(sequence_out_path / "point_clouds.npy", points)
        np.save(sequence_out_path / "point_motions.npy", point_motions)
        
        # Save metadata with useful statistics
        metadata = {
            'num_frames': num_frames,
            'num_valid_frames': num_valid_frames,
            'num_points': points.shape[1],
            'num_points_original': num_points_orig,
            'prediction_horizon': self.prediction_horizon,
            'frame_step': self.frame_step,
            'sequence_name': sequence_name,
            'split': split,
            'stats': {
                'motion_magnitude_mean': float(np.mean(np.linalg.norm(point_motions, axis=3))),
                'motion_magnitude_std': float(np.std(np.linalg.norm(point_motions, axis=3))),
                'position_bounds': {
                    'min': points.min(axis=(0, 1)).tolist(),
                    'max': points.max(axis=(0, 1)).tolist()
                }
            }
        }
        with open(sequence_out_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def assign_splits(self, sequence_dirs: list) -> dict:
        """Assign sequences to splits based on the specified ratios.
        
        Args:
            sequence_dirs: List of sequence directory paths
            
        Returns:
            dict: Mapping of sequence names to their assigned splits
        """
        np.random.shuffle(sequence_dirs)
        total_seqs = len(sequence_dirs)
        split_points = np.cumsum([int(total_seqs * ratio) for ratio in self.splits.values()])
        
        return {
            seq_dir.name: split
            for seq_dir, split in zip(
                sequence_dirs,
                sum([[s] * n for s, n in zip(
                    self.splits.keys(),
                    np.diff([0] + split_points.tolist())
                )], [])
            )
        }

    def convert(self) -> None:
        """Convert all sequences from DELTA format to SceneFlowDataset format."""
        # Get all sequence directories
        sequence_dirs = [d for d in self.input_path.iterdir() if d.is_dir()]
        
        if not sequence_dirs:
            raise ValueError(f"No sequence directories found in {self.input_path}")
        
        # Assign sequences to splits
        split_assignments = self.assign_splits(sequence_dirs)
        
        # Process each sequence
        for sequence_dir in tqdm(sequence_dirs, desc="Processing sequences"):
            split = split_assignments.get(sequence_dir.name, "train")
            self.process_sequence(sequence_dir, split)


def main():
    parser = argparse.ArgumentParser(
        description="Convert DELTA tracking results to SceneFlowDataset format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset/scene_flow.yaml",
        help="Path to dataset configuration file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Create and run converter
    converter = DeltaToFlow3D(config)
    converter.convert()


if __name__ == "__main__":
    main()
