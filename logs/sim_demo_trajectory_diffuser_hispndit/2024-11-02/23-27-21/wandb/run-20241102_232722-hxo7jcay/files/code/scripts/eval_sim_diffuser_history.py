# The evaluation script that runs a rollout for each object in the eval-ed dataset and calculates:
# - success : 90% open
# - distance to open

import json
import os
import pickle as pkl

import hydra
import lightning as L
import numpy as np
import omegaconf
import pandas as pd
import plotly.graph_objects as go
import rpad.pyg.nets.pointnet2 as pnp
import torch
import tqdm
import wandb
from rpad.visualize_3d import html

from flowbothd.models.flow_diffuser_dgdit import (
    FlowTrajectoryDiffuserSimulationModule_DGDiT,
)
from flowbothd.models.flow_diffuser_dit import (
    FlowTrajectoryDiffuserSimulationModule_DiT,
)
from flowbothd.models.flow_diffuser_hisdit import (
    FlowTrajectoryDiffuserSimulationModule_HisDiT,
)
from flowbothd.models.flow_diffuser_hispndit import (
    FlowTrajectoryDiffuserSimulationModule_HisPNDiT,
)
from flowbothd.models.flow_diffuser_pndit import (
    FlowTrajectoryDiffuserSimulationModule_PNDiT,
)
from flowbothd.models.flow_trajectory_diffuser import (
    FlowTrajectoryDiffuserSimulationModule_PN2,
)
from flowbothd.models.modules.dit_models import (
    DGDiT,
    DiT,
    PN2DiT,
    PN2HisDiT,
)
from flowbothd.models.modules.history_encoder import HistoryEncoder
from flowbothd.simulations.simulation import trial_with_diffuser_history
from flowbothd.utils.script_utils import PROJECT_ROOT, match_fn

PROJECT_ROOT = "/home/yishu/FlowBotHD" #"YOUR CURRENT DIRECTORY"


def load_obj_id_to_category(toy_dataset=None):
    id_to_cat = {}
    if toy_dataset is None:
        # Extract existing classes.
        with open(f"{PROJECT_ROOT}/scripts/umpnet_data_split_new.json", "r") as f:
            data = json.load(f)

        for _, category_dict in data.items():
            for category, split_dict in category_dict.items():
                for train_or_test, id_list in split_dict.items():
                    for id in id_list:
                        id_to_cat[id] = f"{category}_{train_or_test}"

    else:
        with open(f"{PROJECT_ROOT}/scripts/umpnet_object_list.json", "r") as f:
            data = json.load(f)
        for split in ["train-train", "train-test"]:
            # for split in ["train-test"]:
            for id in toy_dataset[split]:
                id_to_cat[id] = split
    return id_to_cat


def load_obj_and_link(id_to_cat):
    # with open("./scripts/umpnet_object_list.json", "r") as f:
    # with open(f"{PROJECT_ROOT}/scripts/movable_links_005.json", "r") as f:
    with open(f"{PROJECT_ROOT}/scripts/movable_links_fullset_000.json", "r") as f:
        object_link_json = json.load(f)
    for id in id_to_cat.keys():
        if id not in object_link_json.keys():
            object_link_json[id] = []
    return object_link_json


inference_module_class = {
    "diffuser_pn++": FlowTrajectoryDiffuserSimulationModule_PN2,
    "diffuser_dgdit": FlowTrajectoryDiffuserSimulationModule_DGDiT,
    "diffuser_dit": FlowTrajectoryDiffuserSimulationModule_DiT,
    "diffuser_pndit": FlowTrajectoryDiffuserSimulationModule_PNDiT,
}


@hydra.main(config_path="../configs", config_name="eval_sim", version_base="1.3")
def main(cfg):
    ######################################################################
    # Torch settings.
    ######################################################################

    # Make deterministic + reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Since most of us are training on 3090s+, we can use mixed precision.
    torch.set_float32_matmul_precision("highest")

    # Global seed for reproducibility.
    L.seed_everything(20030208)
    np.random.seed(20030208)
    torch.manual_seed(20030208)

    ######################################################################
    # Create the datamodule.
    # Should be the same one as in training, but we're gonna use val+test
    # dataloaders.
    ######################################################################

    if cfg.dataset.dataset_type == "full-dataset":
        # Full dataset
        toy_dataset = None
    else:
        # Door dataset
        toy_dataset = {
            "id": "door-full-new-noslide",
            "train-train": [
                "8877",
                "8893",
                "8897",
                "8903",
                "8919",
                "8930",
                "8961",
                "8997",
                "9016",
                # "9032",   # has slide
                "9035",
                "9041",
                "9065",
                "9070",
                "9107",
                "9117",
                "9127",
                "9128",
                "9148",
                "9164",
                "9168",
                "9277",
                "9280",
                "9281",
                "9288",
                "9386",
                "9388",
                "9410",
            ],
            "train-test": ["8867", "8983", "8994", "9003", "9263", "9393"],
            "test": ["8867", "8983", "8994", "9003", "9263", "9393"],
        }

    id_to_cat = load_obj_id_to_category(toy_dataset)
    object_to_link = load_obj_and_link(id_to_cat)
    ######################################################################
    # Set up logging in WandB.
    # This is a different job type (eval), but we want it all grouped
    # together. Notice that we use our own logging here (not lightning).
    ######################################################################

    # Create a run.
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        dir=cfg.wandb.save_dir,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        job_type=cfg.job_type,
        save_code=True,  # This just has the main script.
        group=cfg.wandb.group,
    )

    # Log the code.
    wandb.run.log_code(
        root=PROJECT_ROOT,
        include_fn=match_fn(
            dirs=["configs", "scripts", "src"],
            extensions=[".py", ".yaml"],
        ),
    )

    ######################################################################
    # Create the network(s) which will be evaluated (same as training).
    # You might want to put this into a "create_network" function
    # somewhere so train and eval can be the same.
    #
    # We'll also load the weights.
    ######################################################################

    # If we still need original diffusion method
    if "his" not in cfg.model.name:
        trajectory_len = cfg.inference.trajectory_len
        if "diffuser" in cfg.model.name:
            if "pn++" in cfg.model.name:
                in_channels = (
                    3 * cfg.inference.trajectory_len + cfg.model.time_embed_dim
                )
            else:
                in_channels = (
                    3 * cfg.inference.trajectory_len
                )  # Will add 3 as input channel in diffuser
        else:
            in_channels = 1 if cfg.inference.mask_input_channel else 0

        if "pn++" in cfg.model.name:
            network = pnp.PN2Dense(
                in_channels=in_channels,
                out_channels=3 * trajectory_len,
                p=pnp.PN2DenseParams(),
            ).cuda()
        elif "dgdit" in cfg.model.name:
            network = DGDiT(
                in_channels=in_channels,
                depth=5,
                hidden_size=128,
                patch_size=1,
                num_heads=4,
                n_points=cfg.dataset.n_points,
            ).cuda()
        elif "pndit" in cfg.model.name:
            network = PN2DiT(
                in_channels=in_channels,
                depth=5,
                hidden_size=128,
                patch_size=1,
                num_heads=4,
                n_points=cfg.dataset.n_points,
            ).cuda()
        elif "dit" in cfg.model.name:
            network = DiT(
                in_channels=in_channels + 3,
                depth=5,
                hidden_size=128,
                num_heads=4,
                learn_sigma=True,
            ).cuda()

        # # Get the checkpoint file. If it's a wandb reference, download.
        # # Otherwise look to disk.
        # checkpoint_reference = cfg.checkpoint.reference
        # if checkpoint_reference.startswith(cfg.wandb.entity):
        #     # download checkpoint locally (if not already cached)
        #     artifact_dir = cfg.wandb.artifact_dir
        #     artifact = run.use_artifact(checkpoint_reference, type="model")
        #     ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
        # else:
        #     ckpt_file = checkpoint_reference
        ckpt_file = "TO BE SPECIFIED"

        # # Load the network weights.
        # ckpt = torch.load(ckpt_file)
        # network.load_state_dict(
        #     {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items()}
        # )
        model = inference_module_class[cfg.model.name](
            network, inference_cfg=cfg.inference, model_cfg=cfg.model
        ).cuda()
        model.load_from_ckpt(ckpt_file)
        model.eval()

    # History model
    if "hispndit" in cfg.model.name:
        network = {
            "DiT": PN2HisDiT(
                history_embed_dim=cfg.model.history_dim,
                in_channels=3,
                depth=5,
                hidden_size=128,
                num_heads=4,
                learn_sigma=True,
            ).cuda(),
            "History": HistoryEncoder(
                history_dim=cfg.model.history_dim,
                history_len=cfg.model.history_len,
                batch_norm=cfg.model.batch_norm,
                transformer=False,
                repeat_dim=False,
            ).cuda(),
        }
        history_model = FlowTrajectoryDiffuserSimulationModule_HisPNDiT(
            network, inference_cfg=cfg.inference, model_cfg=cfg.model
        ).cuda()
    elif "hisdit" in cfg.model.name:
        network = {
            "DiT": DiT(
                in_channels=3 + 3 + 128,
                depth=5,
                hidden_size=128,
                num_heads=4,
                learn_sigma=True,
            ).cuda(),
            "History": HistoryEncoder(
                history_dim=128, history_len=1, batch_norm=False
            ).cuda(),
        }
        history_model = FlowTrajectoryDiffuserSimulationModule_HisDiT(
            network, inference_cfg=cfg.inference, model_cfg=cfg.model
        ).cuda()
        
    ckpt_file = "/home/yishu/FlowBotHD/logs/train_trajectory_diffuser_hispndit/2024-11-02/22-44-51/checkpoints/epoch=64-step=2795-val_loss=0.00-weights-only.ckpt"#"TO BE SPECIFIED"
    history_model.load_from_ckpt(ckpt_file)
    history_model.eval()

    # Simulation and results.
    print("Simulating")
    instance_results_json = {}  # The success or not results
    flow_results_json = {}  # The flow visualizations
    os.makedirs("./logs/flow_vis")

    if cfg.website:
        # Visualization html
        os.makedirs("./logs/simu_eval/video_assets/")
        doc = html.PlotlyWebsiteBuilder("Simulation Visualizations")
    obj_cats = list(set(id_to_cat.values()))
    metric_df = pd.DataFrame(
        np.zeros((len(set(id_to_cat.values())), 4)),
        index=obj_cats,
        columns=["obj_cat", "count", "success_rate", "norm_dist"],
    )
    category_counts = {}
    sim_trajectories = []
    link_names = []

    # Create the evaluate object lists
    repeat_time = 5
    obj_ids = []
    for obj_id, obj_cat in tqdm.tqdm(list(id_to_cat.items())):
        if "test" not in obj_cat:
            continue
        if not os.path.exists(f"/home/yishu/datasets/partnet-mobility/raw/{obj_id}"):
            continue
        available_links = object_to_link[obj_id]
        if len(available_links) == 0:
            continue
        obj_ids.append(obj_id)
    obj_ids = obj_ids * repeat_time

    import random

    random.shuffle(obj_ids)

    # for obj_id, obj_cat in tqdm.tqdm(list(id_to_cat.items())):
    for obj_id in tqdm.tqdm(obj_ids):
        obj_cat = id_to_cat[obj_id]
        # if "test" not in obj_cat:
        #     continue
        # if not os.path.exists(f"/home/yishu/datasets/partnet-mobility/raw/{obj_id}"):
        #     continue
        available_links = object_to_link[obj_id]
        # if len(available_links) == 0:
        #     continue
        print(f"OBJ {obj_id} of {obj_cat}")
        trial_figs, trial_results, sim_trajectory = trial_with_diffuser_history(
            obj_id=obj_id,
            # model=model,
            model=history_model,  # All history model!!!
            history_model=history_model,
            n_step=30,
            gui=False,
            website=cfg.website,
            all_joint=True,
            available_joints=available_links,
            consistency_check=cfg.consistency_check,
            history_filter=cfg.history_filter,
        )
        sim_trajectories += sim_trajectory
        link_names += [f"{obj_id}_{link}" for link in available_links]

        # Wandb table
        if obj_cat not in category_counts.keys():
            category_counts[obj_cat] = 0
        category_counts[obj_cat] += len(trial_results)

        # for result in trial_results:
        for result, link_name in zip(
            trial_results, [f"{obj_id}_{link}" for link in available_links]
        ):
            instance_results_json[
                link_name
            ] = result.metric  # record the normalized distance
            metric_df.loc[obj_cat]["success_rate"] += result.success
            metric_df.loc[obj_cat]["norm_dist"] += result.metric

        if cfg.website:
            # Website visualization
            for id, (joint_name, fig) in enumerate(trial_figs.items()):
                tag = f"{obj_id}_{joint_name}"
                if fig is not None:
                    doc.add_plot(obj_cat, tag, fig)
                    with open(
                        f"./logs/flow_vis/{tag}.pkl", "wb"
                    ) as f:  # Save the flow visualization (not in website, but for demo)
                        pkl.dump(fig, f)

                    doc.add_video(
                        obj_cat,
                        f"{tag}{'_NO CONTACT' if not trial_results[id].contact else ''}",
                        f"http://128.2.178.238:{cfg.website_port}/video_assets/{tag}.mp4",
                    )
            # print(trial_results)
            doc.write_site("./logs/simu_eval")

        if category_counts[obj_cat] == 0:
            continue
        wandb_df = metric_df.copy(deep=True)
        for obj_cat in category_counts.keys():
            wandb_df.loc[obj_cat]["success_rate"] /= category_counts[obj_cat]
            wandb_df.loc[obj_cat]["norm_dist"] /= category_counts[obj_cat]
            wandb_df.loc[obj_cat]["count"] = category_counts[obj_cat]
            wandb_df.loc[obj_cat]["obj_cat"] = 0 if "train" in obj_cat else 1

        table = wandb.Table(dataframe=wandb_df.reset_index())
        run.log({f"simulation_metric_table": table})

        # Save/update the instance result json
        with open("./logs/instance_result.json", "w") as f:
            json.dump(instance_results_json, f)

    print(wandb_df)

    traces = []
    xs = list(range(31))
    for id, sim_trajectory in enumerate(sim_trajectories):
        traces.append(
            go.Scatter(x=xs, y=sim_trajectory, mode="lines", name=link_names[id])
        )

    layout = go.Layout(title="Simulation Trajectory Figure")
    fig = go.Figure(data=traces, layout=layout)
    wandb.log({"sim_traj_figure": wandb.Plotly(fig)})


if __name__ == "__main__":
    main()
