import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import flwr as fl
import torch
from flwr.common import parameters_to_ndarrays

from shared.model import ConvAE3D
from shared.server import make_strategy


def _setup_run_logging() -> tuple[Path, str]:
    raw_log_dir = os.environ.get("LOG_DIR", "/app/logs")
    logs_dir = Path(raw_log_dir)
    if not logs_dir.is_absolute():
        logs_dir = Path("/app") / logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"server_{ts}.log"
    client_metrics_path = logs_dir / f"client_metrics_{ts}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.info("Log file initialized at %s", log_path)
    logging.info("Client metrics log initialized at %s", client_metrics_path)
    return client_metrics_path, ts


NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", "7"))
NUM_ROUNDS = int(os.environ.get("NUM_ROUNDS", "20"))
LR = float(os.environ.get("LR", "1e-4"))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", "5"))
CLIPNORM = float(os.environ.get("CLIPNORM", "0.0"))
LAMBDA_SAM = float(os.environ.get("LAMBDA_SAM", "0.05"))
SEED = int(os.environ.get("SEED", "42"))
MALICIOUS_CIDS = os.environ.get("MALICIOUS_CIDS", "")
GAUSSIAN_NOISE_MEAN = float(os.environ.get("GAUSSIAN_NOISE_MEAN", "0.0"))
GAUSSIAN_NOISE_STD = float(os.environ.get("GAUSSIAN_NOISE_STD", "1.0"))
ATTACK_MODE = os.environ.get("ATTACK_MODE", "none").lower()
ATTACK_SEED = int(os.environ.get("ATTACK_SEED", str(SEED)))
SIGN_FLIP_ALPHA = float(os.environ.get("SIGN_FLIP_ALPHA", "1.0"))
ANOMALY_PCT = float(os.environ.get("ANOMALY_PCT", "95.0"))
SCORE_DIR = os.environ.get("SCORE_DIR", "/app/outputs/client_scores")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8080"))

AGGREGATION_METHOD = os.environ.get("AGGREGATION_METHOD", "fedavg")
SERVER_CLIPPING_METHOD = os.environ.get("SERVER_CLIPPING_METHOD", "none")
SERVER_FIXED_CLIP_NORM = float(os.environ.get("SERVER_FIXED_CLIP_NORM", "5.0"))
SERVER_ADAPTIVE_CLIP_MULTIPLIER = float(os.environ.get("SERVER_ADAPTIVE_CLIP_MULTIPLIER", "1.5"))
RFA_MAX_ITER = int(os.environ.get("RFA_MAX_ITER", "100"))
RFA_TOL = float(os.environ.get("RFA_TOL", "1e-6"))

# DP parameters
DP_MODE = os.environ.get("DP_MODE", "none").lower()
DP_NOISE_STD = float(os.environ.get("DP_NOISE_STD", "0.0"))
DP_NOISE_MULTIPLIER = float(os.environ.get("DP_NOISE_MULTIPLIER", "0.0"))
DP_SEED = int(os.environ.get("DP_SEED", str(SEED)))

client_metrics_path, _run_ts = _setup_run_logging()

strategy = make_strategy(
    num_clients=NUM_CLIENTS,
    lr=LR,
    local_epochs=LOCAL_EPOCHS,
    clipnorm=CLIPNORM,
    lambda_sam=LAMBDA_SAM,
    seed=SEED,
    malicious_cids=MALICIOUS_CIDS,
    gaussian_noise_mean=GAUSSIAN_NOISE_MEAN,
    gaussian_noise_std=GAUSSIAN_NOISE_STD,
    attack_mode=ATTACK_MODE,
    attack_seed=ATTACK_SEED,
    sign_flip_alpha=SIGN_FLIP_ALPHA,
    anomaly_pct=ANOMALY_PCT,
    save_scores=True,
    score_dir=SCORE_DIR,
    client_metrics_log_path=str(client_metrics_path),

    aggregation_method=AGGREGATION_METHOD,
    server_clipping_method=SERVER_CLIPPING_METHOD,
    server_fixed_clip_norm=SERVER_FIXED_CLIP_NORM,
    server_adaptive_clip_multiplier=SERVER_ADAPTIVE_CLIP_MULTIPLIER,
    rfa_max_iter=RFA_MAX_ITER,
    rfa_tol=RFA_TOL,

    dp_mode=DP_MODE,
    dp_noise_std=DP_NOISE_STD,
    dp_noise_multiplier=DP_NOISE_MULTIPLIER,
    dp_seed=DP_SEED,
)

fl.server.start_server(
    server_address=f"0.0.0.0:{SERVER_PORT}",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)

# Save final checkpoint after all server rounds complete.
try:
    raw_ckpt_dir = os.environ.get("CHECKPOINT_DIR", "/app/checkpoints")
    ckpt_dir = Path(raw_ckpt_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = Path("/app") / ckpt_dir
    run_dir = ckpt_dir / _run_ts
    run_dir.mkdir(parents=True, exist_ok=True)

    final_params = getattr(strategy, "_latest_aggregated_parameters", None)
    if final_params is None:
        logging.warning("No aggregated parameters found; skipping checkpoint save.")
    else:
        model = ConvAE3D()
        nds = parameters_to_ndarrays(final_params)
        keys = list(model.state_dict().keys())
        state = {k: torch.tensor(v, dtype=model.state_dict()[k].dtype) for k, v in zip(keys, nds)}
        model.load_state_dict(state, strict=True)
        ckpt_path = run_dir / "global_model_final.pt"
        torch.save(
            {
                "run_ts": _run_ts,
                "num_rounds": NUM_ROUNDS,
                "state_dict": model.state_dict(),
            },
            ckpt_path,
        )
        logging.info("Saved final checkpoint to %s", ckpt_path)
except Exception:
    logging.exception("Failed to save final checkpoint")
