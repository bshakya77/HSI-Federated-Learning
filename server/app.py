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
ANOMALY_PCT = float(os.environ.get("ANOMALY_PCT", "95.0"))
SCORE_DIR = os.environ.get("SCORE_DIR", "/app/outputs/client_scores")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8080"))

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
    anomaly_pct=ANOMALY_PCT,
    save_scores=True,
    score_dir=SCORE_DIR,
    client_metrics_log_path=str(client_metrics_path),
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
