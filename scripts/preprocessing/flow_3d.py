"""
Scene Flow Prediction using Diffusion Transformers.
This module implements scene-level flow prediction using a DiT architecture.
"""

import os
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from pytorch3d.transforms import Transform3d
import torchmetrics
import wandb
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import logging

from flow_3d.models.dit.models import DiTBlock, TimestepEmbedder, FinalLayer
from flow_3d.utils.pointcloud_utils import expand_pcd
from flow_3d.models.dit.diffusion import create_diffusion
from flow_3d.metrics.flow_metrics import (
    flow_epe, flow_accuracy, FlowRMSE, FlowWeightedRMSE, FlowCosineSimilarity
)

# Configure logger
logger = logging.getLogger(__name__)

class SceneFlowDiT(nn.Module):
    """
    Diffusion Transformer for scene flow prediction.
    Takes a scene point cloud and noised scene flow as input, embeds both,
    concatenates the embeddings, and applies self-attention with adaLN-Zero conditioning.
    
    The model expects scene flow in the format (B, 3*K, N) where:
    - B is the batch size
    - 3*K represents the stacked XYZ coordinates for K timesteps
    - N is the number of points (treated as a spatial dimension)
    
    The flow is treated similarly to an image, where the coordinates across timesteps
    are stacked as channels (like multiple RGB images stacked together).
    """
    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        prediction_horizon: int = 1,  # K timesteps to predict
        model_cfg = None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.prediction_horizon = prediction_horizon
        # Output channels: 3 coordinates × K timesteps
        self.out_channels = 3 * prediction_horizon
        if self.learn_sigma:
            # Double the output channels to predict both mean and variance
            self.out_channels *= 2
        self.num_heads = num_heads
        self.model_cfg = model_cfg
        
        # Scene point cloud encoder - expects (B, 3, N)
        self.scene_embedder = nn.Conv1d(
            in_channels,
            hidden_size // 2,  # Half size since we'll concatenate
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        
        # Flow encoder - processes (B, 3*K, N) format
        self.flow_embedder = nn.Conv1d(
            in_channels * prediction_horizon,  # 3*K input channels
            hidden_size // 2,  # Half size since we'll concatenate
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        
        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # DiT blocks with self-attention
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) 
            for _ in range(depth)
        ])
        
        # Final layer - outputs 3*K channels (or 6*K if learning sigma)
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        
        # Log model configuration
        logger.info(f"Model initialized with out_channels={self.out_channels}, learn_sigma={self.learn_sigma}")
        
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize embedders
        for embedder in [self.scene_embedder, self.flow_embedder]:
            w = embedder.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(embedder.bias, 0)
        
        # Initialize timestep embedding
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of SceneFlowDiT.
        
        Args:
            x: Flow tensor in format (B, 3*K, N)
            t: Diffusion timesteps, shape (B,)
            **kwargs: Additional keyword arguments, must include "scene" in (B, 3, N) format
            
        Returns:
            If learn_sigma:
                Tensor of shape (B, 6*K, N) where first 3*K channels are mean, next 3*K are variance
            If not learn_sigma:
                Tensor of shape (B, 3*K, N)
        """
        scene = kwargs.get("scene")  # (B, 3, N)
        if scene is None:
            raise ValueError("Scene point cloud must be provided as a keyword argument")
        
        B, C, N = x.shape
        K = self.prediction_horizon
        
        if self.model_cfg.center_scene:
            scene_center = torch.mean(scene, dim=2, keepdim=True)  # (B, 3, 1)
            scene = scene - scene_center  # Center the scene points
        
        # Embed scene - already in (B, 3, N)
        scene_feats = self.scene_embedder(scene)
        
        # Embed flow - already in (B, 3*K, N) format
        flow_feats = self.flow_embedder(x)
        
        # Concatenate embeddings along channel dimension
        x = torch.cat([scene_feats, flow_feats], dim=1)  # (B, C, N)
        
        # Convert to (B, N, C) for transformer
        x = x.transpose(1, 2)
        
        # Get timestep embedding
        t = self.t_embedder(t)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, t)
        
        # Final layer
        x = self.final_layer(x, t)
        
        # Convert back to (B, C, N)
        x = x.transpose(1, 2)
        
        return x


class SceneFlowDiffusionModule(L.LightningModule):
    """
    Lightning module for training and inference of scene flow prediction using diffusion.
    """
    def __init__(self, network: SceneFlowDiT, cfg) -> None:
        super().__init__()
        self.network = network
        self.model_cfg = cfg.model
        self.training_cfg = cfg.training
        
        # Training parameters
        self.lr = self.training_cfg.optimizer.lr
        self.weight_decay = self.training_cfg.optimizer.weight_decay
        self.num_training_steps = self.training_cfg.lr_schedule.num_training_steps
        self.lr_warmup_steps = self.training_cfg.lr_schedule.warmup_steps
        
        # Validation/Testing parameters
        self.num_val_samples = self.training_cfg.get('num_val_samples', 1)
        self.test_samples = self.training_cfg.get('test_samples', 5)
        
        # Additional training config with defaults
        self.additional_train_logging_period = self.training_cfg.get('additional_train_logging_period', 100)
        self.num_wta_trials = self.training_cfg.get('num_wta_trials', 5)
        
        # Diffusion parameters
        self.diffusion = create_diffusion(
            timestep_respacing=None,
            diffusion_steps=self.model_cfg.diff_train_steps,
        )
        
        # Base metrics using torchmetrics for efficiency and distributed training
        self.val_flow_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_flow_weighted_rmse = FlowWeightedRMSE()
        self.val_flow_cos_sim = torchmetrics.CosineSimilarity()
        self.test_flow_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.test_flow_weighted_rmse = FlowWeightedRMSE()
        self.test_flow_cos_sim = torchmetrics.CosineSimilarity()
    
    def _standardize_flow(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Standardize flow to format (B, 3*K, N) expected by the network and the diffusion process.
        
        Args:
            flow: Input flow in format (B, N, K, 3)
                 
        Returns:
            Flow in standardized format (B, 3*K, N)
        """
        B, N, K, _ = flow.shape
        # Reshape from (B, N, K, 3) to (B, N, 3*K)
        flow = flow.reshape(B, N, K * 3)
        # Transpose to (B, 3*K, N)
        return flow.transpose(1, 2)

    def forward(self, batch, t):
        """Forward pass for training."""
        # Get scene point cloud and ground truth flow
        scene = batch["points"].to(self.device)
        flow = batch["point_motions"].to(self.device)  # Ground truth flow (B, N, K, 3)
        
        # Standardize flow format to (B, 3*K, N) for diffusion
        flow = self._standardize_flow(flow)
        
        # Run diffusion process
        loss_dict = self.diffusion.training_losses(
            model=self.network,
            x_start=flow,
            t=t,
            model_kwargs={"scene": scene.transpose(1, 2)}  # Convert from (B, N, 3) to (B, 3, N)
        )
        
        return loss_dict, loss_dict["loss"].mean()

    def training_step(self, batch, batch_idx):
        """Training step that computes and returns the loss."""
        self.train()
        # Sample random timesteps for this batch
        B = batch["points"].shape[0]
        t = torch.randint(0, self.model_cfg.diff_train_steps, (B,), device=self.device)
        
        # Forward pass and get loss
        loss_dict, loss = self.forward(batch, t)
        
        # Basic training metrics logging
        # Log all loss components
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v.mean(), prog_bar=(k == "loss"), on_step=True, on_epoch=True, batch_size=B)

        # Determine if additional logging should be done (every N steps)
        do_additional_logging = (self.global_step % self.additional_train_logging_period == 0)

        if do_additional_logging:
            # Additional metrics logging using winner-take-all approach
            all_pred_flows = []
            
            # Generate multiple predictions
            for _ in range(self.num_wta_trials):
                pred_dict = self.predict(batch, num_samples=1, progress=False)
                all_pred_flows.append(pred_dict["pred_flow"])
            
            # Stack predictions and prepare ground truth
            pred_flows = torch.stack(all_pred_flows, dim=1)
            gt_flow = batch["point_motions"].to(self.device)
            
            # Standardize ground truth for comparison
            gt_flow_std = self._standardize_flow(gt_flow)
            
            # Compute RMSE for each sample
            rmses = []
            weighted_rmses = []
            for i in range(self.num_wta_trials):
                pred_flow = pred_flows[:, i]  # (B, N, K, 3)
                
                # Standardize predictions for metrics
                pred_flow_std = self._standardize_flow(pred_flow)
                
                rmse = self.val_flow_rmse(pred_flow_std.reshape(-1, 3), gt_flow_std.reshape(-1, 3))
                rmses.append(rmse)

                weighted_rmse = self.val_flow_weighted_rmse(pred_flow_std.reshape(-1, 3), gt_flow_std.reshape(-1, 3), use_gt_as_weights=True)
                weighted_rmses.append(weighted_rmse)
            
            rmses = torch.stack(rmses)
            weighted_rmses = torch.stack(weighted_rmses)
            
            # Compute winner-take-all metrics (the best prediction)
            rmse_wta = rmses.min()
            winner = torch.argmin(rmses)
            pred_flow_wta = pred_flows[:, winner]
            
            # Standardize winner for metrics
            pred_flow_wta_std = self._standardize_flow(pred_flow_wta)
            
            # Log metrics
            self.log_dict(
                {
                    "train/rmse": rmses.mean(),
                    "train/rmse_wta": rmse_wta,
                    "train/weighted_rmse": weighted_rmses.mean(),
                },
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=B
            )
            
            # Additional flow-specific metrics for WTA prediction
            epe_wta = flow_epe(pred_flow_wta, gt_flow)
            accuracy_wta = flow_accuracy(pred_flow_wta, gt_flow)
            cos_sim_wta = self.val_flow_cos_sim(
                pred_flow_wta_std.reshape(-1, 3),
                gt_flow_std.reshape(-1, 3)
            )
            
            self.log_dict(
                {
                    "train/flow_epe_wta": epe_wta,
                    "train/flow_accuracy_wta": accuracy_wta,
                    "train/flow_cos_sim_wta": cos_sim_wta,
                },
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=B
            )

            # Visualization logging
            if wandb.run:
                self._log_visualizations(
                    batch,
                    pred_flows[0],  # First batch only
                    "train",
                    max_samples=min(3, self.num_wta_trials)
                )
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with multiple prediction samples."""
        # Skip visualization and heavy processing during sanity checking
        # This helps diagnose hangs during sanity check
        if self.trainer.sanity_checking:
            logger.info("Sanity checking - using simplified validation")
            # Process all sanity check batches with simplified validation
            scene = batch["points"]
            gt_flow = batch["point_motions"]
            dummy_metric = torch.tensor(0.5, device=self.device)
            
            # Log metrics (this helps progress bar tracking)
            self.log("val/flow_rmse_sanity", dummy_metric, batch_size=batch["points"].shape[0])
            self.log("val/flow_cos_sim_sanity", dummy_metric, batch_size=batch["points"].shape[0])
            
            # Also log the metrics that the checkpoint monitor is looking for
            self.log("val/flow_rmse", dummy_metric, batch_size=batch["points"].shape[0])
            self.log("val/flow_cos_sim", dummy_metric, batch_size=batch["points"].shape[0])
            self.log("val/flow_epe", dummy_metric, batch_size=batch["points"].shape[0])
            self.log("val/flow_accuracy", dummy_metric, batch_size=batch["points"].shape[0])
            
            # Force metrics to update immediately to help progress bar
            self.val_flow_rmse(
                torch.ones((10, 3), device=self.device),
                torch.ones((10, 3), device=self.device)
            )
            self.val_flow_cos_sim(
                torch.ones((10, 3), device=self.device),
                torch.ones((10, 3), device=self.device)
            )
            
            # Return metrics dict to help progress bar complete
            return {"flow_rmse": dummy_metric, "flow_cos_sim": dummy_metric}
            
        # Calculate diffusion loss for comparison with training loss
        B = batch["points"].shape[0]
        # Use the same timestep sampling strategy as in training
        t = torch.randint(0, self.model_cfg.diff_train_steps, (B,), device=self.device)
        
        # Calculate loss using the same method as training, but without gradients
        with torch.no_grad():
            loss_dict, loss = self.forward(batch, t)
        
        # Log the diffusion loss for comparison with training loss
        self.log("val/diffusion_loss", loss, batch_size=B, on_step=False, on_epoch=True)
        
        # Also log individual loss components if they exist
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v.mean(), batch_size=B, on_step=False, on_epoch=True)
            
        # Original validation code
        scene = batch["points"]
        gt_flow = batch["point_motions"]
        
        # Standardize ground truth for consistent comparison
        gt_flow_std = self._standardize_flow(gt_flow)
        
        # Generate multiple predictions
        all_pred_flows = []
        for _ in range(self.num_val_samples):
            pred_dict = self.predict(batch, num_samples=1, progress=False)
            all_pred_flows.append(pred_dict["pred_flow"])
        
        # Stack predictions
        pred_flows = torch.stack(all_pred_flows, dim=1)  # (B, num_samples, N, K, 3) or (B, num_samples, N, 3)
        
        # Compute metrics for each sample
        rmse_values = []  # Track RMSE values for all samples
        cos_sim_values = []  # Track cosine similarity values for all samples
        epe_values = []  # Track EPE values for all samples
        accuracy_values = []  # Track accuracy values for all samples
        
        for i in range(self.num_val_samples):
            pred_flow = pred_flows[:, i]  # (B, N, K, 3)
            
            # Standardize predictions for metrics
            pred_flow_std = self._standardize_flow(pred_flow)
            
            # Base metrics using torchmetrics
            rmse = self.val_flow_rmse(pred_flow_std.reshape(-1, 3), gt_flow_std.reshape(-1, 3))
            rmse_values.append(rmse)
            self.log(f"val/flow_rmse_sample_{i}", rmse, batch_size=B)
            
            cos_sim = self.val_flow_cos_sim(pred_flow_std.reshape(-1, 3), gt_flow_std.reshape(-1, 3))
            cos_sim_values.append(cos_sim)
            self.log(f"val/flow_cos_sim_sample_{i}", cos_sim, batch_size=B)
            
            # Additional flow-specific metrics
            epe = flow_epe(pred_flow, gt_flow)
            epe_values.append(epe)
            self.log(f"val/flow_epe_sample_{i}", epe, batch_size=B)
            
            accuracy = flow_accuracy(pred_flow, gt_flow)
            accuracy_values.append(accuracy)
            self.log(f"val/flow_accuracy_sample_{i}", accuracy, batch_size=B)
        
        # Calculate and log average metrics across all samples
        if rmse_values:
            avg_rmse = torch.stack(rmse_values).mean()
            avg_cos_sim = torch.stack(cos_sim_values).mean()
            avg_epe = torch.stack(epe_values).mean()
            avg_accuracy = torch.stack(accuracy_values).mean()
            
            # Log average metrics - these will be used for model checkpointing
            self.log("val/flow_rmse", avg_rmse, batch_size=B)
            self.log("val/flow_cos_sim", avg_cos_sim, batch_size=B)
            self.log("val/flow_epe", avg_epe, batch_size=B)
            self.log("val/flow_accuracy", avg_accuracy, batch_size=B)
        
        # Visualize predictions (first batch only)
        if batch_idx == 0:
            self._log_visualizations(
                batch,
                pred_flows[0],  # First batch only
                "val",
                max_samples=min(3, self.num_val_samples)
            )

    def test_step(self, batch, batch_idx):
        """Test step with comprehensive evaluation."""
        # Calculate diffusion loss for comparison with training and validation loss
        B = batch["points"].shape[0]
        # Use the same timestep sampling strategy as in training
        t = torch.randint(0, self.model_cfg.diff_train_steps, (B,), device=self.device)
        
        # Calculate loss using the same method as training, but without gradients
        with torch.no_grad():
            loss_dict, loss = self.forward(batch, t)
        
        # Log the diffusion loss for comparison
        self.log("test/diffusion_loss", loss, batch_size=B, on_step=False, on_epoch=True)
        
        # Also log individual loss components if they exist
        for k, v in loss_dict.items():
            self.log(f"test/{k}", v.mean(), batch_size=B, on_step=False, on_epoch=True)
        
        # Original test code
        scene = batch["points"]
        gt_flow = batch["point_motions"]
        
        # Generate predictions with more samples than validation
        all_metrics = defaultdict(list)
        all_pred_flows = []
        
        for sample_idx in range(self.test_samples):
            pred_dict = self.predict(batch, num_samples=1, progress=False)
            pred_flow = pred_dict["pred_flow"]
            all_pred_flows.append(pred_flow)
            
            # Compute detailed metrics
            metrics = self._compute_test_metrics(pred_flow, gt_flow, scene)
            
            # Store metrics
            for k, v in metrics.items():
                all_metrics[k].append(v)
        
        # Stack predictions
        pred_flows = torch.stack(all_pred_flows, dim=1)  # (B, num_samples, N, K, 3) or (B, num_samples, N, 3)
        
        # Compute and log statistics across samples
        for metric_name, values in all_metrics.items():
            values_tensor = torch.stack(values)
            self.log(f"test/{metric_name}_mean", values_tensor.mean(), batch_size=B)
            self.log(f"test/{metric_name}_std", values_tensor.std(), batch_size=B)
            self.log(f"test/{metric_name}_min", values_tensor.min(), batch_size=B)
            self.log(f"test/{metric_name}_max", values_tensor.max(), batch_size=B)
        
        # Compute best metrics across samples
        best_metrics = self._compute_best_metrics(pred_flows, gt_flow)
        for k, v in best_metrics.items():
            self.log(f"test/best_{k}", v, batch_size=B)
        
        # Visualize predictions (first few batches only)
        if batch_idx < 3:
            self._log_visualizations(
                batch,
                pred_flows[0],  # First batch only
                "test",
                max_samples=min(5, self.test_samples)
            )
        
        return best_metrics

    def _compute_test_metrics(self, pred_flow, gt_flow, scene):
        """Compute comprehensive metrics for test evaluation."""
        metrics = {}
        
        # Base metrics using torchmetrics
        metrics['flow_rmse'] = self.test_flow_rmse(
            pred_flow.reshape(-1, 3),
            gt_flow.reshape(-1, 3)
        )
        metrics['flow_cos_sim'] = self.test_flow_cos_sim(
            pred_flow.reshape(-1, 3),
            gt_flow.reshape(-1, 3)
        )
        
        # Flow-specific metrics
        metrics['flow_epe'] = flow_epe(pred_flow, gt_flow)
        metrics['flow_accuracy'] = flow_accuracy(pred_flow, gt_flow)
        
        # Per-point metrics
        point_distances = torch.norm(pred_flow - gt_flow, dim=-1)
        metrics['max_error'] = point_distances.max()
        metrics['min_error'] = point_distances.min()
        
        # Flow magnitude correlation
        gt_magnitudes = torch.norm(gt_flow, dim=-1)
        pred_magnitudes = torch.norm(pred_flow, dim=-1)
        metrics['magnitude_correlation'] = F.cosine_similarity(
            gt_magnitudes.flatten(),
            pred_magnitudes.flatten(),
            dim=0
        )
        
        return metrics

    def _compute_best_metrics(self, pred_flows, gt_flow):
        """Compute best metrics across multiple predictions."""
        # Handle shapes: pred_flows is (B, S, N, K, 3) or (B, S, N, 3)
        # gt_flow is (B, N, K, 3)
        
        K = self.network.prediction_horizon
        if K > 1:
            if pred_flows.ndim == 5:  # (B, S, N, K, 3)
                B, S, N, K, D = pred_flows.shape
            else:  # (B, S, N, 3)
                B, S, N, D = pred_flows.shape
                K = 1
                
            if gt_flow.ndim == 4:  # (B, N, K, 3)
                pass  # Shape is already correct
            else:  # (B, N, 3)
                gt_flow = gt_flow.unsqueeze(2)  # Add K dimension
        else:
            # Single horizon case
            B, S, N, D = pred_flows.shape
        
        best_metrics = {}
        
        # Reshape for easier computation
        if K > 1:
            # Multi-horizon case, flatten K dimension for comparison
            pred_flows_flat = pred_flows.reshape(B * S, N, K * D)
            gt_flow_flat = gt_flow.unsqueeze(1).expand(-1, S, -1, -1, -1).reshape(B * S, N, K * D)
        else:
            # Single horizon case
            pred_flows_flat = pred_flows.reshape(B * S, N, D)
            gt_flow_flat = gt_flow.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, N, D)
        
        # Compute all RMSEs
        rmses = torch.sqrt(torch.mean((pred_flows_flat - gt_flow_flat) ** 2, dim=(1, 2)))
        rmses = rmses.reshape(B, S)
        
        # Compute all cosine similarities (for each point, then average)
        if K > 1:
            # For multi-horizon, reshape to get per-point, per-timestep cosine similarity
            pred_flat = pred_flows_flat.reshape(-1, D)
            gt_flat = gt_flow_flat.reshape(-1, D)
            cos_sims = F.cosine_similarity(pred_flat, gt_flat, dim=1)
            cos_sims = cos_sims.reshape(B, S, N, K).mean(dim=(2, 3))  # Average over points and timesteps
        else:
            # For single horizon
            cos_sims = F.cosine_similarity(
                pred_flows_flat.reshape(-1, D),
                gt_flow_flat.reshape(-1, D)
            ).reshape(B, S, N).mean(dim=-1)  # Average over points
        
        # Get best metrics
        best_metrics['flow_rmse'] = rmses.min(dim=1)[0].mean()
        best_metrics['flow_cos_sim'] = cos_sims.max(dim=1)[0].mean()
        
        return best_metrics

    def _log_visualizations(self, batch, pred_flows, stage, max_samples=3):
        """Log visualizations of predictions to wandb using the dedicated visualizer utility.
        
        This method delegates to the specialized wandb_visualizer module for creating
        enhanced visualizations of 3D point cloud flow predictions.
        """
        # Check if wandb visualizations are enabled in config
        if not wandb.run or not self.trainer.logger.experiment.config.get('wandb', {}).get('enable_visualizations', True):
            logger.info("Wandb visualizations are disabled in config")
            return
            
        from flow_3d.utils.wandb_visualizer import log_flow_predictions
        
        # Pass the prediction_horizon from the network to the visualizer
        log_flow_predictions(
            batch=batch,
            pred_flows=pred_flows,
            stage=stage,
            step=self.global_step,
            max_samples=max_samples,
            prediction_horizon=self.network.prediction_horizon,
            flowscale=0.05
        )

    @torch.no_grad()
    def predict(self, batch, num_samples=1, progress=True):
        """Generate predictions."""
        # If we're in sanity checking mode, return a simple random prediction
        if hasattr(self, 'trainer') and self.trainer.sanity_checking:
            logger.info("Sanity checking - returning quick random predictions")
            scene = batch["points"].to(self.device)  # (B, N, 3)
            K = self.network.prediction_horizon
            B, N, _ = scene.shape
            
            # Generate random predictions with proper shape
            random_pred = torch.randn(B, N, K, 3, device=self.device)
            return {
                "pred_flow": random_pred,
                "results": {"sanity_check": True}
            }
            
        # Original prediction code
        scene = batch["points"].to(self.device).transpose(1, 2)  # Convert to (B, 3, N)
        
        # Expand scene if doing multiple samples
        if num_samples > 1:
            scene = expand_pcd(scene, num_samples)
        
        # Generate initial noise
        B = scene.shape[0]
        N = scene.shape[2]  # Note: N is now the last dimension
        K = self.network.prediction_horizon
        
        # Initialize noise in (B, 3*K, N) format
        z = torch.randn(B, 3*K, N, device=self.device)
        
        # Run the denoising diffusion process
        pred_flow, results = self.diffusion.p_sample_loop(
            self.network,
            z.shape,
            z,
            model_kwargs={"scene": scene},
            progress=progress,
            device=self.device
        )
        
        # Reshape from (B, 3*K, N) to (B, N, K, 3)
        pred_flow = pred_flow.transpose(1, 2).reshape(B, N, K, 3)
        
        # Transform to world frame if needed
        if "T_goal2world" in batch:
            T_goal2world = Transform3d(
                matrix=expand_pcd(batch["T_goal2world"].to(self.device), num_samples)
            )
            pred_flow = T_goal2world.transform_points(pred_flow)
        
        return {
            "pred_flow": pred_flow,
            "results": results
        }

    def configure_optimizers(self):
        """Configure optimizer with warmup."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.num_training_steps
        )
        return [optimizer], [scheduler]