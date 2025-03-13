import os
import numpy as np
import torch
import argparse
import rpad.pyg.nets.pointnet2 as pnp_orig
from flowbothd.models.modules.dit_models import PN2DiT
from flowbothd.models.modules.history_encoder import HistoryEncoder
from flowbothd.models.flow_diffuser_pndit import (
    FlowTrajectoryDiffuserInferenceModule_PNDiT,
    FlowTrajectoryDiffuserSimulationModule_PNDiT,
)
from flowbot3d.grasping.agents.flowbot3d import FlowNetAnimation

def create_network():
    return PN2DiT(
        in_channels=3,
        depth=5,
        hidden_size=128,
        num_heads=4,
        learn_sigma=True,
    ).cuda()


class InferenceConfig:
    def __init__(self):
        self.batch_size = 1
        self.trajectory_len = 1
        self.mask_input_channel = False


class ModelConfig:
    def __init__(self):
        self.num_train_timesteps = 100
        self.sample_size = 1200  # Number of points to process in each batch


def load_models(network, inference_config, model_config, ckpt_path):
    # Load inference model
    model = FlowTrajectoryDiffuserInferenceModule_PNDiT(
        network, inference_cfg=inference_config, model_cfg=model_config
    )
    model.load_from_ckpt(ckpt_path)
    model.eval()
    model.cuda()

    # Load simulation model
    # sim_model = FlowTrajectoryDiffuserSimulationModule_PNDiT(
    #     network, inference_cfg=inference_config, model_cfg=model_config
    # ).cuda()
    # sim_model.load_from_ckpt(ckpt_path)
    # sim_model.eval()

    return None, model


def load_point_clouds(sequence_path, frame_idx=0):
    """Load point clouds and motions from the dataset format.
    
    Args:
        sequence_path: Path to the sequence directory containing point_clouds.npy and point_motions.npy
        frame_idx: Which frame to use as current frame
        
    Returns:
        history_pcd: Previous frame point cloud
        history_flow: Flow from previous to current frame
        curr_pcd: Current frame point cloud
    """
    # Load point clouds
    point_clouds_path = os.path.join(sequence_path, "point_clouds.npy")
    point_clouds = np.load(point_clouds_path)  # Shape: (num_frames, num_points, 3)
    
    if frame_idx >= point_clouds.shape[0]:
        raise ValueError(f"frame_idx {frame_idx} is out of bounds for point cloud with {point_clouds.shape[0]} frames")
    
    # Get current frame point cloud
    curr_pcd = point_clouds[frame_idx]
    
    # If we have a previous frame, use it for history
    if frame_idx > 0:
        history_pcd = point_clouds[frame_idx - 1]
        history_flow = curr_pcd - history_pcd
    else:
        # If this is the first frame, use zeros for history
        history_pcd = np.zeros_like(curr_pcd)
        history_flow = np.zeros_like(curr_pcd)
    
    return history_pcd, history_flow, curr_pcd


def visualize_flow(history_pcd, history_flow, curr_pcd):
    animation = FlowNetAnimation()

    animation.add_trace(
        torch.as_tensor(history_pcd),
        torch.as_tensor([history_pcd]),
        torch.as_tensor([history_flow * 3]),
        "red",
    )
    animation.add_trace(
        torch.as_tensor(curr_pcd),
        torch.as_tensor([curr_pcd]),
        torch.as_tensor([np.zeros_like(curr_pcd)]),
        "red",
    )

    fig = animation.animate()
    return fig


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='FlowBot3D Demo')
    parser.add_argument('--ckpt_path', type=str, default='./pretrained/door_pndit.ckpt',
                      help='Path to the checkpoint file')
    parser.add_argument('--sequence_path', type=str, required=True,
                      help='Path to the sequence directory containing point_clouds.npy and point_motions.npy')
    parser.add_argument('--frame_idx', type=int, default=0,
                      help='Frame index to use as current frame')
    parser.add_argument('--use_history', action='store_true',
                      help='Whether to use history for prediction')
    args = parser.parse_args()

    # Create configurations
    inference_config = InferenceConfig()
    model_config = ModelConfig()

    # Check if checkpoint exists
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {args.ckpt_path}")

    # Check if sequence directory exists and contains required files
    if not os.path.exists(args.sequence_path):
        raise FileNotFoundError(f"Sequence directory not found at: {args.sequence_path}")
    
    point_clouds_path = os.path.join(args.sequence_path, "point_clouds.npy")
    point_motions_path = os.path.join(args.sequence_path, "point_motions.npy")
    
    if not os.path.exists(point_clouds_path):
        raise FileNotFoundError(f"point_clouds.npy not found at: {point_clouds_path}")
    if not os.path.exists(point_motions_path):
        raise FileNotFoundError(f"point_motions.npy not found at: {point_motions_path}")

    # Initialize networks and models
    network = create_network()
    dummy_model, model = load_models(network, inference_config, model_config, args.ckpt_path)

    # Load point cloud data
    history_pcd, history_flow, curr_pcd = load_point_clouds(
        args.sequence_path,
        args.frame_idx
    )

    # Convert numpy arrays to torch tensors and move to GPU
    curr_pcd_tensor = curr_pcd
    # history_pcd_tensor = torch.from_numpy(history_pcd).cuda()
    # history_flow_tensor = torch.from_numpy(history_flow).cuda()

    # Random sampling without history
    pred_flow = model.predict(curr_pcd_tensor)[:, 0, :]

    # Prediction with history
    # if args.use_history:
    #     pred_flow_history = model.predict(
    #         curr_pcd_tensor,
    #         history_pcd=history_pcd_tensor,
    #         history_flow=history_flow_tensor
    #     )[:, 0, :]

    # Visualize
    animation = FlowNetAnimation()
    animation.add_trace(
        torch.as_tensor(history_pcd),
        torch.as_tensor([history_pcd]),
        torch.as_tensor([history_flow * 3]),
        "red",
    )
    animation.add_trace(
        torch.as_tensor(curr_pcd),
        torch.as_tensor([curr_pcd]),
        torch.as_tensor([pred_flow_history.cpu().numpy() if args.use_history else pred_flow.cpu().numpy()]),
        "red",
    )

    fig = animation.animate()
    fig.show()


if __name__ == "__main__":
    main()