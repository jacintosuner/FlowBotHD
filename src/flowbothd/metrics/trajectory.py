import torch


def normalize_trajectory(pred):  # pred: bs * 1200, traj_len, 3
    return pred
    pred = pred.reshape(-1, 1200, pred.shape[1], pred.shape[2])
    norm = pred.norm(p=2, dim=-1)
    norm = torch.max(norm, dim=1).values + 1e-6
    pred = pred / norm[:, None, :, None]
    return torch.flatten(pred, start_dim=0, end_dim=1)  # bs * 1200, traj_len, 3


def flow_metrics(
    pred_flow, gt_flow, reduce=True
):  # if reduce = True, return mean, else return everything
    with torch.no_grad():
        # RMSE
        rmse = (pred_flow - gt_flow).norm(p=2, dim=-1)  # .mean()

        # Cosine similarity, normalized.
        # Only compute cosine similarity where gt_flow is non-zero
        gt_flow_norm = gt_flow.norm(dim=-1)
        nonzero_mask = gt_flow_norm > 0
        
        if nonzero_mask.any():
            gt_flow_nz = gt_flow[nonzero_mask]
            pred_flow_nz = pred_flow[nonzero_mask]
            cos_dist = torch.cosine_similarity(pred_flow_nz, gt_flow_nz, dim=-1)
        else:
            # If no non-zero ground truth flows, return 0 cosine similarity
            cos_dist = torch.zeros_like(rmse)

        # Magnitude
        mag_error = (
            pred_flow.norm(p=2, dim=-1) - gt_flow.norm(p=2, dim=-1)
        ).abs()  # .mean()

    if reduce:
        return rmse.mean(), cos_dist.mean(), mag_error.mean()
    else:
        return rmse, cos_dist, mag_error


def artflownet_loss(
    f_pred: torch.Tensor,
    f_target: torch.Tensor,
    n_nodes: torch.Tensor,
    reduce: bool = True,
) -> torch.Tensor:  # if reduce = True, return mean, else return everything
    # f_pred = normalize_trajectory(f_pred)

    # Flow loss, per-point.
    raw_se = ((f_pred - f_target) ** 2).sum(dim=-1)

    if reduce:
        weights = (1 / n_nodes).repeat_interleave(n_nodes)
        l_se = (raw_se * weights[:, None]).sum() / f_pred.shape[1]  # Trajectory length

        # Full loss, averaged across the batch.
        loss: torch.Tensor = l_se / len(n_nodes)

        return loss

    else:
        return raw_se
