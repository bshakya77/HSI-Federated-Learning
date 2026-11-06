from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Union

import flwr as fl
import numpy as np
import torch
from flwr.common import Metrics
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from shared.model import ConvAE3D


# Reproducibility
def seed_everything(seed: int) -> None:
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


# Metrics aggregation
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}

    total = sum(num_examples for num_examples, _ in metrics)
    if total == 0:
        return {}

    agg: Dict[str, float] = {}
    keys = set()

    for _, metric in metrics:
        keys.update(metric.keys())

    for key in keys:
        score = 0.0
        for num_examples, metric in metrics:
            if key in metric:
                score += float(metric[key]) * (num_examples / total)
        agg[key] = score

    return agg


# Parameter / update utilities
def _flatten_ndarrays(arrs: List[np.ndarray]) -> np.ndarray:
    if not arrs:
        return np.array([], dtype=np.float64)

    return np.concatenate(
        [a.astype(np.float64, copy=False).ravel() for a in arrs]
    )


def _unflatten_ndarrays(flat: np.ndarray, template: List[np.ndarray]) -> List[np.ndarray]:
    out = []
    pointer = 0

    for arr in template:
        size = arr.size
        out.append(flat[pointer:pointer + size].reshape(arr.shape).astype(arr.dtype, copy=False))
        pointer += size

    return out


def _params_to_update(client_params: List[np.ndarray], global_params: List[np.ndarray]) -> List[np.ndarray]:
    return [
        client_layer - global_layer
        for client_layer, global_layer in zip(client_params, global_params)
    ]


def _apply_update(global_params: List[np.ndarray], update: List[np.ndarray]) -> List[np.ndarray]:
    return [
        global_layer + update_layer
        for global_layer, update_layer in zip(global_params, update)
    ]


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / denom)


def _l2_vec(a: np.ndarray) -> float:
    return float(np.linalg.norm(a)) if a.size else 0.0


def _update_norm(update: List[np.ndarray]) -> float:
    return _l2_vec(_flatten_ndarrays(update))


# Server-side clipping defenses
def _clip_update(update: List[np.ndarray], clip_norm: float) -> List[np.ndarray]:
    norm = _update_norm(update)

    if clip_norm <= 0 or norm <= clip_norm:
        return update

    scale = clip_norm / (norm + 1e-12)

    return [
        layer * scale
        for layer in update
    ]


def _clip_updates_fixed(
    updates: List[List[np.ndarray]],
    fixed_clip_norm: float,
) -> tuple[List[List[np.ndarray]], float]:
    clipped = [
        _clip_update(update, fixed_clip_norm)
        for update in updates
    ]

    return clipped, fixed_clip_norm


def _clip_updates_adaptive(
    updates: List[List[np.ndarray]],
    multiplier: float = 1.5,
) -> tuple[List[List[np.ndarray]], float]:
    norms = np.array([_update_norm(update) for update in updates], dtype=np.float64)

    if norms.size == 0:
        return updates, 0.0

    median_norm = float(np.median(norms))
    clip_norm = float(multiplier * median_norm)

    clipped = [
        _clip_update(update, clip_norm)
        for update in updates
    ]

    return clipped, clip_norm

# DP noise addition
def _add_noise_to_update(
    aggregated_update: List[np.ndarray],
    dp_mode: str,
    clip_norm: float,
    noise_std: float,
    noise_multiplier: float,
    num_clients: int,
    seed: int,
    server_round: int,
) -> tuple[List[np.ndarray], float]:
    """
    Add deterministic Gaussian noise to the aggregated server update.

    Modes:
    - none:
        no noise

    - standalone:
        add direct Gaussian noise with std = DP_NOISE_STD.
        This is useful as a noise-only robustness baseline.
        It should not be described as formal DP because updates are not clipped.

    - clipped:
        add Gaussian noise calibrated to the clipping norm:
            std = DP_NOISE_MULTIPLIER * clip_norm / num_clients
        This is the DP-style clipped update setting.
    """

    dp_mode = str(dp_mode).lower()

    if dp_mode == "none":
        return aggregated_update, 0.0

    if dp_mode == "standalone":
        if noise_std <= 0.0:
            raise ValueError(
                "DP_MODE='standalone' requires DP_NOISE_STD > 0."
            )

        final_noise_std = float(noise_std)

    elif dp_mode == "clipped":
        if clip_norm <= 0.0:
            raise ValueError(
                "DP_MODE='clipped' requires server-side clipping. "
                "Set SERVER_CLIPPING_METHOD to 'fixed' or 'adaptive'."
            )

        if noise_multiplier <= 0.0:
            raise ValueError(
                "DP_MODE='clipped' requires DP_NOISE_MULTIPLIER > 0."
            )

        if num_clients <= 0:
            raise ValueError("num_clients must be positive.")

        final_noise_std = float(noise_multiplier * clip_norm / num_clients)

    else:
        raise ValueError(
            f"Unknown DP_MODE={dp_mode}. Use one of: none, standalone, clipped."
        )

    rng = np.random.default_rng(int(seed) + int(server_round) * 100_000)

    noisy_update = []

    for layer in aggregated_update:
        noise = rng.normal(
            loc=0.0,
            scale=final_noise_std,
            size=layer.shape,
        ).astype(layer.dtype, copy=False)

        noisy_update.append(layer + noise)

    return noisy_update, final_noise_std

# Aggregators
def _fedavg_updates(
    updates: List[List[np.ndarray]],
    num_examples: List[int],
) -> List[np.ndarray]:
    total_examples = float(sum(num_examples))

    if total_examples <= 0:
        weights = np.ones(len(updates), dtype=np.float64) / max(len(updates), 1)
    else:
        weights = np.array(num_examples, dtype=np.float64) / total_examples

    aggregated = []

    for layer_idx in range(len(updates[0])):
        layer_sum = np.zeros_like(updates[0][layer_idx])

        for client_idx, update in enumerate(updates):
            layer_sum += weights[client_idx] * update[layer_idx]

        aggregated.append(layer_sum)

    return aggregated


def _weighted_geometric_median(
    points: np.ndarray,
    weights: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
    eps: float = 1e-12,
) -> np.ndarray:
    weights = weights.astype(np.float64)
    weights = weights / (weights.sum() + eps)

    current = np.average(points, axis=0, weights=weights)

    for _ in range(max_iter):
        distances = np.linalg.norm(points - current, axis=1)
        distances = np.maximum(distances, eps)

        adjusted_weights = weights / distances
        next_point = np.sum(points * adjusted_weights[:, None], axis=0) / np.sum(adjusted_weights)

        shift = np.linalg.norm(next_point - current)

        current = next_point

        if shift < tol:
            break

    return current


def _rfa_updates(
    updates: List[List[np.ndarray]],
    num_examples: List[int],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> List[np.ndarray]:
    flat_updates = np.stack(
        [_flatten_ndarrays(update) for update in updates],
        axis=0,
    )

    total_examples = float(sum(num_examples))

    if total_examples <= 0:
        weights = np.ones(len(updates), dtype=np.float64) / max(len(updates), 1)
    else:
        weights = np.array(num_examples, dtype=np.float64) / total_examples

    median_flat = _weighted_geometric_median(
        flat_updates,
        weights=weights,
        max_iter=max_iter,
        tol=tol,
    )

    return _unflatten_ndarrays(median_flat, updates[0])


# Robust strategy
class RobustAggregationStrategy(fl.server.strategy.FedAvg):
    """
    Flower strategy supporting:

    - FedAvg
    - RFA / geometric median
    - no clipping
    - fixed server-side update clipping
    - adaptive median-based server-side update clipping

    Important:
    This performs clipping on client model updates at the server.
    This is different from local gradient clipping inside each client.
    """

    def __init__(
        self,
        *args,
        aggregation_method: str = "fedavg",
        server_clipping_method: str = "none",
        server_fixed_clip_norm: float = 5.0,
        server_adaptive_clip_multiplier: float = 1.5,
        rfa_max_iter: int = 100,
        rfa_tol: float = 1e-6,
        dp_mode: str = "none",
        dp_noise_std: float = 0.0,
        dp_noise_multiplier: float = 0.0,
        dp_seed: int = 42,
        client_metrics_logger: logging.Logger | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.aggregation_method = aggregation_method.lower()
        self.server_clipping_method = server_clipping_method.lower()
        self.server_fixed_clip_norm = float(server_fixed_clip_norm)
        self.server_adaptive_clip_multiplier = float(server_adaptive_clip_multiplier)
        self.rfa_max_iter = int(rfa_max_iter)
        self.rfa_tol = float(rfa_tol)
        self.dp_mode = str(dp_mode).lower()
        self.dp_noise_std = float(dp_noise_std)
        self.dp_noise_multiplier = float(dp_noise_multiplier)
        self.dp_seed = int(dp_seed)

        self._client_metrics_logger = client_metrics_logger
        self._last_global_ndarrays: List[np.ndarray] | None = None
        self._last_honest_mean_vec: np.ndarray | None = None
        self._last_mal_mean_vec: np.ndarray | None = None
        self._latest_aggregated_parameters = None

    def configure_fit(self, server_round, parameters, client_manager):
        try:
            self._last_global_ndarrays = [
                a.copy()
                for a in parameters_to_ndarrays(parameters)
            ]
        except Exception:
            self._last_global_ndarrays = None

        self._last_honest_mean_vec = None
        self._last_mal_mean_vec = None

        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        if self._last_global_ndarrays is None:
            return super().aggregate_fit(server_round, results, failures)

        base = self._last_global_ndarrays

        client_updates: List[List[np.ndarray]] = []
        num_examples: List[int] = []

        honest_vecs = []
        malicious_vecs = []

        per_client_log = []

        # Convert returned client weights to updates.
        for client_proxy, fit_res in results:
            client_params = parameters_to_ndarrays(fit_res.parameters)
            update = _params_to_update(client_params, base)
            update_vec = _flatten_ndarrays(update)
            update_norm = _l2_vec(update_vec)

            m = fit_res.metrics or {}

            metric_cid = int(m.get("cid_metric", -1)) if "cid_metric" in m else -1
            proxy_cid = getattr(client_proxy, "cid", "unknown")

            is_mal = bool(float(m.get("is_malicious", 0.0))) if "is_malicious" in m else False
            alpha = float(m.get("attack_alpha", 0.0)) if "attack_alpha" in m else 0.0

            client_updates.append(update)
            num_examples.append(int(fit_res.num_examples))

            if is_mal:
                malicious_vecs.append(update_vec)
            else:
                honest_vecs.append(update_vec)

            per_client_log.append(
                {
                    "proxy_cid": str(proxy_cid),
                    "metric_cid": metric_cid,
                    "is_malicious": is_mal,
                    "attack_alpha": alpha,
                    "pre_update_norm": float(m.get("pre_update_norm", float("nan"))),
                    "post_update_norm": float(m.get("post_update_norm", float("nan"))),
                    "server_delta_norm": update_norm,
                    "update_cosine": float(m.get("update_cosine", float("nan"))),
                    "train_loss": float(m.get("train_loss", float("nan"))),
                    "train_mse": float(m.get("train_mse", float("nan"))),
                    "train_sam": float(m.get("train_sam", float("nan"))),
                    "num_examples": int(fit_res.num_examples),
                }
            )

        pre_clip_norms = np.array(
            [_update_norm(update) for update in client_updates],
            dtype=np.float64,
        )

        # Store honest / malicious means before clipping for attack diagnostics.
        honest_mean_vec = (
            np.mean(np.stack(honest_vecs), axis=0)
            if honest_vecs
            else np.array([], dtype=np.float64)
        )
        mal_mean_vec = (
            np.mean(np.stack(malicious_vecs), axis=0)
            if malicious_vecs
            else np.array([], dtype=np.float64)
        )

        self._last_honest_mean_vec = honest_mean_vec if honest_vecs else None
        self._last_mal_mean_vec = mal_mean_vec if malicious_vecs else None

        honest_mal_cos = (
            _cosine(honest_mean_vec, mal_mean_vec)
            if honest_vecs and malicious_vecs
            else float("nan")
        )

        # Server-side clipping defense.
        clip_norm_used = 0.0

        if self.server_clipping_method == "none":
            clipped_updates = client_updates
            clip_norm_used = 0.0

        elif self.server_clipping_method == "fixed":
            clipped_updates, clip_norm_used = _clip_updates_fixed(
                client_updates,
                self.server_fixed_clip_norm,
            )

        elif self.server_clipping_method == "adaptive":
            clipped_updates, clip_norm_used = _clip_updates_adaptive(
                client_updates,
                multiplier=self.server_adaptive_clip_multiplier,
            )

        else:
            raise ValueError(
                f"Unknown SERVER_CLIPPING_METHOD={self.server_clipping_method}. "
                "Use one of: none, fixed, adaptive."
            )

        post_clip_norms = np.array(
            [_update_norm(update) for update in clipped_updates],
            dtype=np.float64,
        )

        # Aggregate clipped updates.
        if self.aggregation_method == "fedavg":
            aggregated_update = _fedavg_updates(
                clipped_updates,
                num_examples=num_examples,
            )

        elif self.aggregation_method == "rfa":
            aggregated_update = _rfa_updates(
                clipped_updates,
                num_examples=num_examples,
                max_iter=self.rfa_max_iter,
                tol=self.rfa_tol,
            )
        
        else:
            raise ValueError(
                f"Unknown AGGREGATION_METHOD={self.aggregation_method}. "
                "Use one of: fedavg, rfa."
            )

        aggregated_update, dp_noise_std_used = _add_noise_to_update(
            aggregated_update=aggregated_update,
            dp_mode=self.dp_mode,
            clip_norm=clip_norm_used,
            noise_std=self.dp_noise_std,
            noise_multiplier=self.dp_noise_multiplier,
            num_clients=len(clipped_updates),
            seed=self.dp_seed,
            server_round=server_round,
        )


        new_global = _apply_update(base, aggregated_update)
        aggregated_parameters = ndarrays_to_parameters(new_global)
        self._latest_aggregated_parameters = aggregated_parameters

        aggregated_update_vec = _flatten_ndarrays(aggregated_update)
        global_shift_norm = _l2_vec(aggregated_update_vec)

        cos_to_honest = (
            _cosine(aggregated_update_vec, honest_mean_vec)
            if honest_vecs
            else float("nan")
        )
        cos_to_mal = (
            _cosine(aggregated_update_vec, mal_mean_vec)
            if malicious_vecs
            else float("nan")
        )

        if self._client_metrics_logger is not None:
            self._client_metrics_logger.info(
                "[ROUND %s] defense_summary aggregation=%s clipping=%s clip_norm=%.6f "
                "dp_mode=%s dp_noise_std_config=%.12f dp_noise_multiplier=%.6f "
                "dp_noise_std_used=%.12f "
                "pre_clip_mean=%.6f pre_clip_max=%.6f "
                "post_clip_mean=%.6f post_clip_max=%.6f "
                "honest_vs_mal_cos=%.6f global_shift_norm=%.6f "
                "cos_to_honest=%.6f cos_to_mal=%.6f",
                server_round,
                self.aggregation_method,
                self.server_clipping_method,
                clip_norm_used,
                self.dp_mode,
                self.dp_noise_std,
                self.dp_noise_multiplier,
                dp_noise_std_used,
                float(np.mean(pre_clip_norms)) if pre_clip_norms.size else 0.0,
                float(np.max(pre_clip_norms)) if pre_clip_norms.size else 0.0,
                float(np.mean(post_clip_norms)) if post_clip_norms.size else 0.0,
                float(np.max(post_clip_norms)) if post_clip_norms.size else 0.0,
                honest_mal_cos,
                global_shift_norm,
                cos_to_honest,
                cos_to_mal,
            )

            for i, c in enumerate(per_client_log):
                self._client_metrics_logger.info(
                    "[ROUND %s] client proxy_cid=%s metric_cid=%s malicious=%s alpha=%.3f "
                    "num_examples=%s pre_update_norm=%.6f post_update_norm=%.6f "
                    "server_pre_clip_norm=%.6f server_post_clip_norm=%.6f update_cosine=%.6f "
                    "train_loss=%.6f train_mse=%.6f train_sam=%.6f",
                    server_round,
                    c["proxy_cid"],
                    c["metric_cid"],
                    c["is_malicious"],
                    c["attack_alpha"],
                    c["num_examples"],
                    c["pre_update_norm"],
                    c["post_update_norm"],
                    float(pre_clip_norms[i]),
                    float(post_clip_norms[i]),
                    c["update_cosine"],
                    c["train_loss"],
                    c["train_mse"],
                    c["train_sam"],
                )

        aggregated_metrics = {}

        if self.fit_metrics_aggregation_fn is not None:
            fit_metrics = [
                (fit_res.num_examples, fit_res.metrics)
                for _, fit_res in results
            ]
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)

        return aggregated_parameters, aggregated_metrics


# Strategy factory
def make_strategy(
    num_clients: int = 3,
    lr: float = 3e-4,
    local_epochs: int = 1,
    clipnorm: float = 0.0,
    lambda_sam: float = 0.01,
    seed: int = 42,
    malicious_cids: str = "",
    gaussian_noise_mean: float = 0.0,
    gaussian_noise_std: float = 1.0,
    attack_mode: str = "none",
    attack_seed: int = 42,
    sign_flip_alpha: float = 1.0,
    anomaly_pct: float = 95.0,
    save_scores: bool = True,
    score_dir: str = "client_scores",
    client_metrics_log_path: str | None = None,
    aggregation_method: str = "fedavg",
    server_clipping_method: str = "none",
    server_fixed_clip_norm: float = 5.0,
    server_adaptive_clip_multiplier: float = 1.5,
    rfa_max_iter: int = 100,
    rfa_tol: float = 1e-6,
    dp_mode: str = "none",
    dp_noise_std: float = 0.0,
    dp_noise_multiplier: float = 0.0,
    dp_seed: int = 42,
) -> fl.server.strategy.FedAvg:
    seed_everything(int(seed))

    model = ConvAE3D()

    initial_parameters = ndarrays_to_parameters(
        [
            v.detach().cpu().numpy().copy()
            for _, v in model.state_dict().items()
        ]
    )

    def fit_config(server_round: int) -> Dict[str, Union[int, float, str]]:
        return {
            "lr": lr,
            "local_epochs": local_epochs,
            "clipnorm": clipnorm,
            "lambda_sam": lambda_sam,

            # Attack controls
            "attack_mode": attack_mode,
            "malicious_cids": malicious_cids,
            "gaussian_noise_mean": gaussian_noise_mean,
            "gaussian_noise_std": gaussian_noise_std,
            "attack_seed": attack_seed,
            "sign_flip_alpha": sign_flip_alpha,
            "server_round": server_round,
        }

    def evaluate_config(server_round: int) -> Dict[str, Union[int, float, str, bool]]:
        return {
            "lambda_sam": lambda_sam,
            "anomaly_pct": anomaly_pct,
            "save_scores": save_scores,
            "score_dir": score_dir,
            "server_round": server_round,
        }

    client_metrics_logger: logging.Logger | None = None

    if client_metrics_log_path:
        client_metrics_logger = logging.getLogger("client-metrics")
        client_metrics_logger.setLevel(logging.INFO)

        if not any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", "") == str(Path(client_metrics_log_path))
            for h in client_metrics_logger.handlers
        ):
            fh = logging.FileHandler(client_metrics_log_path, encoding="utf-8")
            fh.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            client_metrics_logger.addHandler(fh)

    logging.info(
        "Initialized FL strategy: aggregation=%s, server_clipping=%s, "
        "fixed_clip_norm=%.6f, adaptive_multiplier=%.6f,"
        "dp_noise_multiplier=%.6f, dp_seed=%s, "
        "malicious_cids=%s, alpha=%.3f",
        aggregation_method,
        server_clipping_method,
        server_fixed_clip_norm,
        server_adaptive_clip_multiplier,
        dp_noise_multiplier,
        dp_seed,
        malicious_cids,
        sign_flip_alpha,
    )

    return RobustAggregationStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
        client_metrics_logger=client_metrics_logger,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        aggregation_method=aggregation_method,
        server_clipping_method=server_clipping_method,
        server_fixed_clip_norm=server_fixed_clip_norm,
        server_adaptive_clip_multiplier=server_adaptive_clip_multiplier,
        rfa_max_iter=rfa_max_iter,
        rfa_tol=rfa_tol,
        dp_mode=dp_mode,
        dp_noise_std=dp_noise_std,
        dp_noise_multiplier=dp_noise_multiplier,
        dp_seed=dp_seed,
    )