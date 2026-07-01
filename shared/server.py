from __future__ import annotations

import logging
import os
import random
from typing import Dict, List, Tuple, Union

import flwr as fl
from flwr.common import Metrics
from flwr.common import ndarrays_to_parameters
from flwr.common import parameters_to_ndarrays

import numpy as np
import torch
from pathlib import Path

from shared.model import ConvAE3D


def seed_everything(seed: int) -> None:
    """Best-effort reproducibility across Python/NumPy/PyTorch."""
    if seed is None:
        return

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    # Helps CUDA GEMM determinism (must be set before CUDA context init to fully apply).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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


def _flatten_ndarrays(arrs: List[np.ndarray]) -> np.ndarray:
    if not arrs:
        return np.array([], dtype=np.float64)
    return np.concatenate([a.astype(np.float64, copy=False).ravel() for a in arrs])


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / denom)


def _l2(a: np.ndarray) -> float:
    return float(np.linalg.norm(a)) if a.size else 0.0


class FedAvgWithClientMetrics(fl.server.strategy.FedAvg):
    """FedAvg with additional per-round client/update diagnostics logging."""

    def __init__(self, *args, client_metrics_logger: logging.Logger | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._client_metrics_logger = client_metrics_logger
        self._last_global_ndarrays: List[np.ndarray] | None = None
        self._last_honest_mean_vec: np.ndarray | None = None
        self._last_mal_mean_vec: np.ndarray | None = None
        self._latest_aggregated_parameters = None

    def configure_fit(self, server_round, parameters, client_manager):
        # Keep a copy of the current global parameters to compute client deltas.
        try:
            self._last_global_ndarrays = [a.copy() for a in parameters_to_ndarrays(parameters)]
        except Exception:
            self._last_global_ndarrays = None
        self._last_honest_mean_vec = None
        self._last_mal_mean_vec = None
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        # Compute diagnostics BEFORE aggregation (deltas relative to current global).
        if self._client_metrics_logger is not None and self._last_global_ndarrays is not None and results:
            base = self._last_global_ndarrays

            per_client = []
            honest_vecs = []
            malicious_vecs = []

            for client_proxy, fit_res in results:
                cid = getattr(client_proxy, "cid", "unknown")
                client_params = parameters_to_ndarrays(fit_res.parameters)
                delta = [cp - bp for cp, bp in zip(client_params, base)]
                dvec = _flatten_ndarrays(delta)
                dnorm = _l2(dvec)

                m = fit_res.metrics or {}
                is_mal = bool(float(m.get("is_malicious", 0.0))) if "is_malicious" in m else False
                g_mean = float(m.get("gaussian_noise_mean", 0.0)) if "gaussian_noise_mean" in m else 0.0
                g_std = float(m.get("gaussian_noise_std", 0.0)) if "gaussian_noise_std" in m else 0.0

                per_client.append(
                    {
                        "cid": str(cid),
                        "delta_norm": float(dnorm),
                        "is_malicious": bool(is_mal),
                        "gaussian_noise_mean": float(g_mean),
                        "gaussian_noise_std": float(g_std),
                        "pre_update_norm": float(m.get("pre_update_norm", float("nan"))),
                        "post_update_norm": float(m.get("post_update_norm", float("nan"))),
                        "update_cosine": float(m.get("update_cosine", float("nan"))),
                        "train_loss": float(m.get("train_loss", float("nan"))),
                        "train_mse": float(m.get("train_mse", float("nan"))),
                        "train_sam": float(m.get("train_sam", float("nan"))),
                    }
                )

                if is_mal:
                    malicious_vecs.append(dvec)
                else:
                    honest_vecs.append(dvec)

            # Pairwise cosine similarities (summaries)
            delta_vecs = []
            for client_proxy, fit_res in results:
                client_params = parameters_to_ndarrays(fit_res.parameters)
                delta = [cp - bp for cp, bp in zip(client_params, base)]
                delta_vecs.append(_flatten_ndarrays(delta))

            pairwise = []
            for i in range(len(delta_vecs)):
                for j in range(i + 1, len(delta_vecs)):
                    pairwise.append(_cosine(delta_vecs[i], delta_vecs[j]))

            pairwise_mean = float(np.mean(pairwise)) if pairwise else 0.0

            honest_mean_vec = np.mean(np.stack(honest_vecs), axis=0) if honest_vecs else np.array([], dtype=np.float64)
            mal_mean_vec = np.mean(np.stack(malicious_vecs), axis=0) if malicious_vecs else np.array([], dtype=np.float64)
            self._last_honest_mean_vec = honest_mean_vec if honest_vecs else None
            self._last_mal_mean_vec = mal_mean_vec if malicious_vecs else None

            honest_mal_dir = _cosine(honest_mean_vec, mal_mean_vec) if (honest_vecs and malicious_vecs) else float("nan")

            self._client_metrics_logger.info(
                "[ROUND %s] client_update_summary: n=%d pairwise_cos_mean=%.6f honest_vs_mal_cos=%.6f honest_mean_norm=%.6f mal_mean_norm=%.6f",
                server_round,
                len(per_client),
                pairwise_mean,
                honest_mal_dir,
                _l2(honest_mean_vec),
                _l2(mal_mean_vec),
            )
            for c in per_client:
                self._client_metrics_logger.info(
                    "[ROUND %s] client=%s malicious=%s g_mean=%.3f g_std=%.3f delta_norm=%.6f pre_norm=%.6f post_norm=%.6f upd_cos=%.6f train_loss=%.6f train_mse=%.6f train_sam=%.6f",
                    server_round,
                    c["cid"],
                    c["is_malicious"],
                    c["gaussian_noise_mean"],
                    c["gaussian_noise_std"],
                    c["delta_norm"],
                    c["pre_update_norm"],
                    c["post_update_norm"],
                    c["update_cosine"],
                    c["train_loss"],
                    c["train_mse"],
                    c["train_sam"],
                )

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self._latest_aggregated_parameters = aggregated_parameters

        # Global update shift AFTER aggregation (new_global - old_global)
        if (
            self._client_metrics_logger is not None
            and self._last_global_ndarrays is not None
            and aggregated_parameters is not None
        ):
            new_global = parameters_to_ndarrays(aggregated_parameters)
            base = self._last_global_ndarrays
            gdelta = [ng - bg for ng, bg in zip(new_global, base)]
            gvec = _flatten_ndarrays(gdelta)
            cos_to_honest = (
                _cosine(gvec, self._last_honest_mean_vec) if self._last_honest_mean_vec is not None else float("nan")
            )
            cos_to_mal = (
                _cosine(gvec, self._last_mal_mean_vec) if self._last_mal_mean_vec is not None else float("nan")
            )
            self._client_metrics_logger.info(
                "[ROUND %s] global_shift: norm=%.6f cos_to_honest=%.6f cos_to_mal=%.6f",
                server_round,
                _l2(gvec),
                cos_to_honest,
                cos_to_mal,
            )

        return aggregated_parameters, aggregated_metrics


def make_strategy(
    num_clients: int = 3,
    lr: float = 3e-4,
    local_epochs: int = 1,
    clipnorm: float = 1.0,
    lambda_sam: float = 0.01,
    seed: int = 42,
    malicious_cids: str = "",
    gaussian_noise_mean: float = 0.0,
    gaussian_noise_std: float = 1.0,
    anomaly_pct: float = 95.0,
    save_scores: bool = True,
    score_dir: str = "client_scores",
    client_metrics_log_path: str | None = None,
) -> fl.server.strategy.FedAvg:
    seed_everything(int(seed))
    model = ConvAE3D()
    initial_parameters = ndarrays_to_parameters(
        [v.detach().cpu().numpy().copy() for _, v in model.state_dict().items()]
    )

    def fit_config(server_round: int) -> Dict[str, Union[int, float, str]]:
        return {
            "lr": lr,
            "local_epochs": local_epochs,
            "clipnorm": clipnorm,
            "lambda_sam": lambda_sam,
            "malicious_cids": malicious_cids,
            "gaussian_noise_mean": gaussian_noise_mean,
            "gaussian_noise_std": gaussian_noise_std,
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
        # Avoid duplicate handlers if make_strategy is called twice.
        if not any(
            isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(Path(client_metrics_log_path))
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

    return FedAvgWithClientMetrics(
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
    )
