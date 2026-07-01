import logging
import os
import random
from typing import List, Sequence

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from shared.model import ConvAE3D, composite_loss


def seed_everything(seed: int) -> None:
    """Best-effort reproducibility across Python/NumPy/PyTorch."""
    if seed is None:
        return

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_parameters(model):
    return [v.detach().cpu().numpy().copy() for _, v in model.state_dict().items()]


def set_parameters(model, params):
    keys = list(model.state_dict().keys())
    state = {k: torch.tensor(v, dtype=model.state_dict()[k].dtype) for k, v in zip(keys, params)}
    model.load_state_dict(state, strict=True)


def flatten_params(params: Sequence[np.ndarray]) -> np.ndarray:
    if not params:
        return np.array([], dtype=np.float64)
    return np.concatenate([p.astype(np.float64, copy=False).ravel() for p in params])


def params_diff(params_a: Sequence[np.ndarray], params_b: Sequence[np.ndarray]) -> List[np.ndarray]:
    return [a - b for a, b in zip(params_a, params_b)]


def vector_l2_norm(vec: np.ndarray) -> float:
    if vec.size == 0:
        return 0.0
    return float(np.linalg.norm(vec))


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray, eps: float = 1e-12) -> float:
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + eps
    return float(np.dot(vec_a, vec_b) / denom)


def parse_malicious_ids(malicious_cids) -> set[int]:
    if isinstance(malicious_cids, (list, tuple)):
        return {int(cid) for cid in malicious_cids if str(cid).strip() != ""}
    if isinstance(malicious_cids, str):
        return {int(cid.strip()) for cid in malicious_cids.split(",") if cid.strip() != ""}
    return set()


def per_patch_mse(xhat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 5:
        x = x.squeeze(1)
    if xhat.dim() == 5:
        xhat = xhat.squeeze(1)
    return ((xhat - x) ** 2).mean(dim=(1, 2, 3))


def per_patch_sam(xhat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.dim() == 5:
        x = x.squeeze(1)
    if xhat.dim() == 5:
        xhat = xhat.squeeze(1)

    dot = (xhat * x).sum(dim=1)
    xhat_norm = torch.sqrt((xhat * xhat).sum(dim=1) + eps)
    x_norm = torch.sqrt((x * x).sum(dim=1) + eps)

    cos = dot / (xhat_norm * x_norm + eps)
    cos = torch.clamp(cos, -1.0, 1.0)
    ang = torch.acos(cos)
    return ang.mean(dim=(1, 2))


class HSIClient(fl.client.NumPyClient):
    def __init__(self, cid, x_data, split, batch_size=4, seed: int = 42):
        self.cid = int(cid)
        # Make per-client RNG deterministic but distinct.
        seed_everything(int(seed) + self.cid)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvAE3D().to(self.device)

        tr_idx, va_idx = split
        x_train = torch.from_numpy(x_data[tr_idx]).unsqueeze(1)
        x_val = torch.from_numpy(x_data[va_idx]).unsqueeze(1)

        g = torch.Generator()
        g.manual_seed(int(seed) + self.cid)
        self.trainloader = DataLoader(
            TensorDataset(x_train),
            batch_size=batch_size,
            shuffle=True,
            generator=g,
        )
        self.valloader = DataLoader(TensorDataset(x_val), batch_size=batch_size, shuffle=False)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        global_params = [p.copy() for p in parameters]
        set_parameters(self.model, parameters)

        lr = float(config.get("lr", 3e-4))
        local_epochs = int(config.get("local_epochs", 1))
        clipnorm = float(config.get("clipnorm", 1.0))
        lambda_sam = float(config.get("lambda_sam", 0.01))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        total_loss, total_mse, total_sam, seen = 0.0, 0.0, 0.0, 0

        nan_detected = False
        nan_stage = "none"

        for epoch in range(local_epochs):
            if nan_detected:
                break

            for batch_idx, (xb,) in enumerate(self.trainloader):
                xb = xb.to(self.device)
                optimizer.zero_grad()
                xhat = self.model(xb)
                loss, mse_v, sam_v = composite_loss(xhat, xb, lambda_sam=lambda_sam)

                if not torch.isfinite(loss):
                    nan_detected = True
                    nan_stage = "loss"
                    logging.warning(
                        "[Client %s] Non-finite loss detected at epoch=%s batch=%s",
                        self.cid,
                        epoch,
                        batch_idx,
                    )
                    break

                loss.backward()

                grad_finite = True
                for param in self.model.parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        grad_finite = False
                        break

                if not grad_finite:
                    nan_detected = True
                    nan_stage = "grad"
                    logging.warning(
                        "[Client %s] Non-finite gradient detected at epoch=%s batch=%s",
                        self.cid,
                        epoch,
                        batch_idx,
                    )
                    break

                if clipnorm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clipnorm)
                optimizer.step()

                param_finite = True
                for param in self.model.parameters():
                    if not torch.isfinite(param.data).all():
                        param_finite = False
                        break

                if not param_finite:
                    nan_detected = True
                    nan_stage = "param"
                    logging.warning(
                        "[Client %s] Non-finite parameter detected after step at epoch=%s batch=%s",
                        self.cid,
                        epoch,
                        batch_idx,
                    )
                    break

                bs = xb.size(0)
                total_loss += loss.item() * bs
                total_mse += mse_v.item() * bs
                total_sam += sam_v.item() * bs
                seen += bs

        updated_params = get_parameters(self.model)
        malicious_ids = parse_malicious_ids(config.get("malicious_cids", ""))
        is_malicious = self.cid in malicious_ids
        gaussian_mean = float(config.get("gaussian_noise_mean", 0.0))
        gaussian_std = float(config.get("gaussian_noise_std", 1.0))

        honest_update = params_diff(updated_params, global_params)
        honest_update_vec = flatten_params(honest_update)
        pre_update_norm = vector_l2_norm(honest_update_vec)

        sent_params = updated_params
        if is_malicious:
            poisoned_params: List[np.ndarray] = []
            for global_param in global_params:
                noise = np.random.normal(
                    loc=gaussian_mean,
                    scale=gaussian_std,
                    size=global_param.shape,
                ).astype(global_param.dtype, copy=False)
                poisoned_params.append(global_param + noise)
            sent_params = poisoned_params

        sent_update = params_diff(sent_params, global_params)
        sent_update_vec = flatten_params(sent_update)
        post_update_norm = vector_l2_norm(sent_update_vec)
        update_cosine = cosine_similarity(honest_update_vec, sent_update_vec)

        logging.info(
            "[Client %s] malicious=%s g_mean=%.3f g_std=%.3f pre_norm=%.6f post_norm=%.6f cosine=%.6f nan_detected=%s nan_stage=%s",
            self.cid,
            is_malicious,
            gaussian_mean if is_malicious else 0.0,
            gaussian_std if is_malicious else 0.0,
            pre_update_norm,
            post_update_norm,
            update_cosine,
            nan_detected,
            nan_stage,
        )

        return (
            sent_params,
            len(self.trainloader.dataset),
            {
                "train_loss": float(total_loss / max(seen, 1)),
                "train_mse": float(total_mse / max(seen, 1)),
                "train_sam": float(total_sam / max(seen, 1)),
                "cid_metric": float(self.cid),
                "is_malicious": float(is_malicious),
                "attack_alpha": float(0.0),
                "gaussian_noise_mean": float(gaussian_mean if is_malicious else 0.0),
                "gaussian_noise_std": float(gaussian_std if is_malicious else 0.0),
                "pre_update_norm": float(pre_update_norm),
                "post_update_norm": float(post_update_norm),
                "update_cosine": float(update_cosine),
                "nan_detected": float(nan_detected),
            },
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()

        lambda_sam = float(config.get("lambda_sam", 0.01))
        anomaly_pct = float(config.get("anomaly_pct", 95))
        server_round = int(config.get("server_round", -1))
        save_scores = bool(config.get("save_scores", True))
        out_dir = str(config.get("score_dir", "client_scores"))

        total_loss, total_mse, total_sam, seen = 0.0, 0.0, 0.0, 0
        all_patch_mse = []
        all_patch_sam = []

        with torch.no_grad():
            for (xb,) in self.valloader:
                xb = xb.to(self.device)
                xhat = self.model(xb)
                loss, mse_v, sam_v = composite_loss(xhat, xb, lambda_sam=lambda_sam)

                bs = xb.size(0)
                total_loss += loss.item() * bs
                total_mse += mse_v.item() * bs
                total_sam += sam_v.item() * bs
                seen += bs

                all_patch_mse.append(per_patch_mse(xhat, xb).cpu())
                all_patch_sam.append(per_patch_sam(xhat, xb).cpu())

        val_loss = float(total_loss / max(seen, 1))
        val_mse = float(total_mse / max(seen, 1))
        val_sam = float(total_sam / max(seen, 1))
        patch_mse = torch.cat(all_patch_mse).numpy() if all_patch_mse else np.array([])
        patch_sam = torch.cat(all_patch_sam).numpy() if all_patch_sam else np.array([])

        if patch_mse.size > 0:
            thr_mse = float(np.percentile(patch_mse, anomaly_pct))
            frac_mse = float((patch_mse > thr_mse).mean())
        else:
            thr_mse, frac_mse = 0.0, 0.0

        if patch_sam.size > 0:
            thr_sam = float(np.percentile(patch_sam, anomaly_pct))
            frac_sam = float((patch_sam > thr_sam).mean())
        else:
            thr_sam, frac_sam = 0.0, 0.0

        if save_scores:
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, f"client{self.cid}_val_patch_mse.npy"), patch_mse)
            np.save(os.path.join(out_dir, f"client{self.cid}_val_patch_sam.npy"), patch_sam)

        metrics = {
            "val_mse": val_mse,
            "val_sam": val_sam,
            "mse_p95": thr_mse,
            "sam_p95": thr_sam,
            "anom_frac_mse": frac_mse,
            "anom_frac_sam": frac_sam,
        }

        logging.info(
            "[Client %s] eval round=%s val_loss=%.6f val_mse=%.6f val_sam=%.6f mse_p95=%.6f sam_p95=%.6f anom_frac_mse=%.6f anom_frac_sam=%.6f",
            self.cid,
            server_round,
            val_loss,
            val_mse,
            val_sam,
            thr_mse,
            thr_sam,
            frac_mse,
            frac_sam,
        )

        return val_loss, len(self.valloader.dataset), metrics
